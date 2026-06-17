from __future__ import annotations

"""Launch multiple standalone RWKV infer services on idle GPUs."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.eval.scheduler.process import list_idle_gpus


_ENGINE_MODE_CHOICES = ("rwkv-lightning", "lightning", "classic")


def _default_engine_mode() -> str:
    value = os.environ.get("RWKV_INFER_ENGINE_MODE", "rwkv-lightning").strip().lower()
    if value in _ENGINE_MODE_CHOICES:
        return value
    return "rwkv-lightning"


@dataclass(frozen=True, slots=True)
class InferServiceSpec:
    model_path: Path
    model_name: str
    gpu: str
    port: int
    max_batch_size: int
    log_path: Path
    state_db_path: Path | None = None
    model_index: int = 0
    replica_index: int = 0
    replica_count: int = 1

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/healthz"


@dataclass(frozen=True, slots=True)
class RunningInferService:
    spec: InferServiceSpec
    pid: int


@dataclass(frozen=True, slots=True)
class InferRouterSpec:
    host: str
    port: int
    routes: tuple[str, ...]
    timeout_s: float
    log_level: str
    log_path: Path

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/healthz"


@dataclass(frozen=True, slots=True)
class RunningInferRouter:
    spec: InferRouterSpec
    pid: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch RWKV infer services on idle GPUs")
    parser.add_argument("--model-paths", "--models", nargs="+", required=True, help="RWKV weight paths to deploy")
    parser.add_argument(
        "--model-names",
        nargs="+",
        help="Public model names; must match --model-paths length when provided",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host passed to each infer service")
    parser.add_argument("--base-port", type=int, default=18081, help="First service port; later services increment")
    parser.add_argument("--api-key", default="", help="Bearer token required by each infer service")
    parser.add_argument(
        "--engine-mode",
        choices=_ENGINE_MODE_CHOICES,
        default=_default_engine_mode(),
        help="Inference backend implementation; rwkv-lightning is the formal server backend",
    )
    parser.add_argument(
        "--infer-auto-config",
        choices=("off", "balanced", "throughput"),
        help="Startup auto-config mode passed to each infer service; omit to use run_infer_server default",
    )
    parser.add_argument("--state-db-dir", help="Directory for per-model lightning sqlite state caches")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max queued requests per infer batch")
    parser.add_argument(
        "--max-batch-sizes",
        nargs="+",
        type=int,
        help="Per-model infer batch sizes; length must match --models. Overrides --max-batch-size.",
    )
    parser.add_argument("--batch-collect-ms", type=int, default=10, help="Batch collection window")
    parser.add_argument(
        "--replicas-per-model",
        type=int,
        default=1,
        help="Number of infer service replicas to launch for each model on its assigned GPU",
    )
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    parser.add_argument("--log-dir", default="logs/infer", help="Directory for child service logs")
    parser.add_argument("--manifest-path", default="logs/infer/fleet.json", help="JSON manifest output path")
    parser.add_argument("--gpu-idle-max-mem", type=int, default=1000, help="GPU idle threshold in MB")
    parser.add_argument("--poll-seconds", type=float, default=10.0, help="Polling interval while waiting for GPUs")
    parser.add_argument(
        "--startup-stagger-s",
        type=float,
        default=2.0,
        help="Sleep after launching one service before probing the next GPU",
    )
    parser.add_argument(
        "--no-wait-for-gpus",
        action="store_true",
        help="Launch on currently idle GPUs only; fail if some models cannot be deployed",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Start child services in new sessions and exit after writing the manifest",
    )
    parser.add_argument(
        "--router-port",
        type=int,
        help="When provided, launch run_infer_router on this port after all infer services start",
    )
    parser.add_argument("--router-host", help="Router bind host; defaults to --host")
    parser.add_argument("--router-timeout-s", type=float, default=600.0, help="Router backend request timeout")
    parser.add_argument("--router-log-level", help="Router uvicorn log level; defaults to --log-level")
    parser.add_argument("--router-log-path", help="Router child process log path; defaults under --log-dir")
    return parser.parse_args(argv)


def resolve_model_names(model_paths: Sequence[Path], model_names: Sequence[str] | None) -> tuple[str, ...]:
    if model_names is None:
        return tuple(path.stem for path in model_paths)
    if len(model_names) != len(model_paths):
        raise ValueError("--model-names length must match --model-paths length")
    cleaned = tuple(str(name).strip() for name in model_names)
    if any(not name for name in cleaned):
        raise ValueError("--model-names cannot contain empty values")
    return cleaned


def resolve_max_batch_sizes(
    model_paths: Sequence[Path],
    *,
    max_batch_size: int,
    max_batch_sizes: Sequence[int] | None,
) -> tuple[int, ...]:
    if max_batch_sizes is None:
        return tuple(max(1, int(max_batch_size)) for _path in model_paths)
    if len(max_batch_sizes) != len(model_paths):
        raise ValueError("--max-batch-sizes length must match --model-paths")
    return tuple(max(1, int(value)) for value in max_batch_sizes)


def plan_deployments(
    *,
    model_paths: Sequence[Path],
    model_names: Sequence[str],
    max_batch_sizes: Sequence[int],
    idle_gpus: Sequence[str],
    assigned_gpus: set[str],
    base_port: int,
    log_dir: Path,
    state_db_dir: Path | None,
    launched_count: int,
    replicas_per_model: int = 1,
) -> list[InferServiceSpec]:
    available_gpus = [gpu for gpu in idle_gpus if gpu not in assigned_gpus]
    specs: list[InferServiceSpec] = []
    replica_count = max(1, int(replicas_per_model))
    gpu_cursor = 0
    for model_index, (model_path, model_name, max_batch_size) in enumerate(
        zip(model_paths, model_names, max_batch_sizes, strict=False)
    ):
        if len(available_gpus) - gpu_cursor < replica_count:
            break
        safe_name = _safe_name(model_name)
        for replica_index in range(replica_count):
            gpu = available_gpus[gpu_cursor]
            gpu_cursor += 1
            port = int(base_port) + int(launched_count) + len(specs)
            replica_suffix = "" if replica_count == 1 else f".r{replica_index + 1}"
            state_db_path = None
            if state_db_dir is not None:
                state_db_path = state_db_dir / f"{safe_name}{replica_suffix}.sqlite3"
            specs.append(
                InferServiceSpec(
                    model_path=model_path,
                    model_name=model_name,
                    gpu=str(gpu),
                    port=port,
                    max_batch_size=max(1, int(max_batch_size)),
                    log_path=log_dir / f"{safe_name}{replica_suffix}.port{port}.log",
                    state_db_path=state_db_path,
                    model_index=model_index,
                    replica_index=replica_index,
                    replica_count=replica_count,
                )
            )
    return specs


def build_command(
    spec: InferServiceSpec,
    *,
    host: str,
    api_key: str,
    engine_mode: str,
    infer_auto_config: str | None,
    batch_collect_ms: int,
    log_level: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.bin.run_infer_server",
        "--model-path",
        str(spec.model_path),
        "--model-name",
        spec.model_name,
        "--device",
        "cuda:0",
        "--engine-mode",
        engine_mode,
        "--host",
        host,
        "--port",
        str(spec.port),
        "--max-batch-size",
        str(int(spec.max_batch_size)),
        "--batch-collect-ms",
        str(int(batch_collect_ms)),
        "--log-level",
        log_level,
    ]
    if infer_auto_config:
        command.extend(["--infer-auto-config", str(infer_auto_config)])
    if api_key:
        command.extend(["--api-key", api_key])
    if spec.state_db_path is not None:
        command.extend(["--state-db-path", str(spec.state_db_path)])
    return command


def build_router_command(spec: InferRouterSpec) -> list[str]:
    if not spec.routes:
        raise ValueError("router requires at least one route")
    command = [
        sys.executable,
        "-m",
        "src.bin.run_infer_router",
        "--host",
        spec.host,
        "--port",
        str(int(spec.port)),
        "--timeout-s",
        str(float(spec.timeout_s)),
        "--log-level",
        spec.log_level,
    ]
    for route in spec.routes:
        command.extend(["--route", route])
    return command


def launch_service(
    spec: InferServiceSpec,
    *,
    host: str,
    api_key: str,
    engine_mode: str,
    infer_auto_config: str | None,
    batch_collect_ms: int,
    log_level: str,
    detach: bool,
) -> subprocess.Popen[bytes]:
    command = build_command(
        spec,
        host=host,
        api_key=api_key,
        engine_mode=engine_mode,
        infer_auto_config=infer_auto_config,
        batch_collect_ms=batch_collect_ms,
        log_level=log_level,
    )
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.state_db_path is not None:
        spec.state_db_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu)
    with spec.log_path.open("ab", buffering=0) as stream:
        stream.write(f"\n$ {' '.join(command)}\nCUDA_VISIBLE_DEVICES={spec.gpu}\n".encode("utf-8"))
        return subprocess.Popen(
            command,
            stdout=stream,
            stderr=stream,
            env=env,
            start_new_session=detach,
        )


def launch_router(spec: InferRouterSpec, *, detach: bool) -> subprocess.Popen[bytes]:
    command = build_router_command(spec)
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.log_path.open("ab", buffering=0) as stream:
        stream.write(f"\n$ {' '.join(command)}\n".encode("utf-8"))
        return subprocess.Popen(
            command,
            stdout=stream,
            stderr=stream,
            start_new_session=detach,
        )


def build_routes_by_model(services: Sequence[RunningInferService]) -> dict[str, list[str]]:
    routes_by_model: dict[str, list[str]] = {}
    for service in services:
        routes_by_model.setdefault(service.spec.model_name, []).append(service.spec.base_url)
    return routes_by_model


def build_router_routes(services: Sequence[RunningInferService]) -> list[str]:
    routes_by_model = build_routes_by_model(services)
    return [
        f"{model_name}={base_url}"
        for model_name, base_urls in routes_by_model.items()
        for base_url in base_urls
    ]


def write_manifest(
    manifest_path: Path,
    *,
    services: Sequence[RunningInferService],
    host: str,
    api_key_set: bool,
    router: RunningInferRouter | None = None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    routes_by_model = build_routes_by_model(services)
    router_routes = build_router_routes(services)
    payload = {
        "host": host,
        "api_key_set": bool(api_key_set),
        "routes_by_model": routes_by_model,
        "router_routes": router_routes,
        "router": None
        if router is None
        else {
            "host": router.spec.host,
            "port": router.spec.port,
            "base_url": router.spec.base_url,
            "health_url": router.spec.health_url,
            "routes": list(router.spec.routes),
            "timeout_s": router.spec.timeout_s,
            "log_level": router.spec.log_level,
            "log_path": str(router.spec.log_path),
            "pid": router.pid,
        },
        "services": [
            {
                "model_path": str(service.spec.model_path),
                "model_name": service.spec.model_name,
                "gpu": service.spec.gpu,
                "port": service.spec.port,
                "max_batch_size": service.spec.max_batch_size,
                "replica_index": service.spec.replica_index,
                "replica_count": service.spec.replica_count,
                "log_path": str(service.spec.log_path),
                "state_db_path": None if service.spec.state_db_path is None else str(service.spec.state_db_path),
                "base_url": service.spec.base_url,
                "health_url": service.spec.health_url,
                "pid": service.pid,
            }
            for service in services
        ],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _terminate_pids(pids: Sequence[int]) -> None:
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.time() + 20
    for pid in pids:
        while time.time() < deadline:
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        else:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def terminate_services(services: Sequence[RunningInferService]) -> None:
    _terminate_pids([service.pid for service in services])


def terminate_running(services: Sequence[RunningInferService], router: RunningInferRouter | None = None) -> None:
    pids = [service.pid for service in services]
    if router is not None:
        pids.append(router.pid)
    _terminate_pids(pids)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_paths = tuple(Path(path).expanduser().resolve() for path in args.model_paths)
    for path in model_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    model_names = resolve_model_names(model_paths, args.model_names)
    max_batch_sizes = resolve_max_batch_sizes(
        model_paths,
        max_batch_size=int(args.max_batch_size),
        max_batch_sizes=args.max_batch_sizes,
    )
    pending_paths = list(model_paths)
    pending_names = list(model_names)
    pending_batch_sizes = list(max_batch_sizes)
    log_dir = Path(args.log_dir).expanduser()
    manifest_path = Path(args.manifest_path).expanduser()
    state_db_dir = None if not args.state_db_dir else Path(args.state_db_dir).expanduser()
    wait_for_gpus = not bool(args.no_wait_for_gpus)

    assigned_gpus: set[str] = set()
    services: list[RunningInferService] = []
    processes: dict[int, subprocess.Popen[bytes]] = {}
    router: RunningInferRouter | None = None

    try:
        while pending_paths:
            idle_gpus = list_idle_gpus(int(args.gpu_idle_max_mem))
            specs = plan_deployments(
                model_paths=pending_paths,
                model_names=pending_names,
                max_batch_sizes=pending_batch_sizes,
                idle_gpus=idle_gpus,
                assigned_gpus=assigned_gpus,
                base_port=int(args.base_port),
                log_dir=log_dir,
                state_db_dir=state_db_dir,
                launched_count=len(services),
                replicas_per_model=int(args.replicas_per_model),
            )
            if not specs:
                if not wait_for_gpus:
                    missing = ", ".join(pending_names)
                    raise RuntimeError(f"no idle GPU available for pending models: {missing}")
                print(
                    f"waiting for idle GPU; pending={len(pending_paths)}, "
                    f"assigned={sorted(assigned_gpus)}, threshold={args.gpu_idle_max_mem}MB",
                    flush=True,
                )
                time.sleep(max(float(args.poll_seconds), 1.0))
                continue

            for spec in specs:
                print(
                    f"launch {spec.model_name} on gpu={spec.gpu} port={spec.port} log={spec.log_path}",
                    flush=True,
                )
                process = launch_service(
                    spec,
                    host=str(args.host),
                    api_key=str(args.api_key or ""),
                    engine_mode=str(args.engine_mode),
                    infer_auto_config=args.infer_auto_config,
                    batch_collect_ms=int(args.batch_collect_ms),
                    log_level=str(args.log_level),
                    detach=bool(args.detach),
                )
                assigned_gpus.add(spec.gpu)
                services.append(RunningInferService(spec=spec, pid=int(process.pid)))
                processes[int(process.pid)] = process
                write_manifest(
                    manifest_path,
                    services=services,
                    host=str(args.host),
                    api_key_set=bool(args.api_key),
                    router=router,
                )
                time.sleep(max(float(args.startup_stagger_s), 0.0))
            consumed_models = 1 + max(spec.model_index for spec in specs)
            del pending_paths[:consumed_models]
            del pending_names[:consumed_models]
            del pending_batch_sizes[:consumed_models]

        if args.router_port is not None:
            router_log_path = (
                Path(args.router_log_path).expanduser()
                if args.router_log_path
                else log_dir / f"router.port{int(args.router_port)}.log"
            )
            router_spec = InferRouterSpec(
                host=str(args.router_host or args.host),
                port=int(args.router_port),
                routes=tuple(build_router_routes(services)),
                timeout_s=float(args.router_timeout_s),
                log_level=str(args.router_log_level or args.log_level),
                log_path=router_log_path,
            )
            print(
                f"launch router port={router_spec.port} routes={len(router_spec.routes)} log={router_spec.log_path}",
                flush=True,
            )
            router_process = launch_router(router_spec, detach=bool(args.detach))
            router = RunningInferRouter(spec=router_spec, pid=int(router_process.pid))
            processes[int(router_process.pid)] = router_process

        write_manifest(
            manifest_path,
            services=services,
            host=str(args.host),
            api_key_set=bool(args.api_key),
            router=router,
        )
        print(f"manifest written: {manifest_path}", flush=True)
        if args.detach:
            return 0

        while processes:
            for pid, process in list(processes.items()):
                rc = process.poll()
                if rc is None:
                    continue
                if router is not None and pid == router.pid:
                    print(f"infer router exited: pid={pid} returncode={rc}", flush=True)
                    processes.pop(pid)
                    if rc != 0:
                        terminate_services(services)
                        return int(rc)
                    continue
                service = next(item for item in services if item.pid == pid)
                print(f"infer service exited: model={service.spec.model_name} pid={pid} returncode={rc}", flush=True)
                processes.pop(pid)
                if rc != 0:
                    remaining_services = [item for item in services if item.pid != pid]
                    terminate_running(remaining_services, router)
                    return int(rc)
            time.sleep(2)
        return 0
    except KeyboardInterrupt:
        terminate_running(services, router)
        return 130


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("._") or "model"


__all__ = [
    "InferRouterSpec",
    "InferServiceSpec",
    "RunningInferRouter",
    "RunningInferService",
    "build_command",
    "build_router_command",
    "build_router_routes",
    "build_routes_by_model",
    "launch_service",
    "launch_router",
    "main",
    "parse_args",
    "plan_deployments",
    "resolve_max_batch_sizes",
    "resolve_model_names",
    "terminate_running",
    "write_manifest",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
