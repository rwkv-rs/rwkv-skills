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


@dataclass(frozen=True, slots=True)
class InferServiceSpec:
    model_path: Path
    model_name: str
    gpu: str
    port: int
    max_batch_size: int
    log_path: Path
    state_db_path: Path | None = None

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
        choices=("classic", "lightning"),
        default="classic",
        help="Local inference engine implementation to use",
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
) -> list[InferServiceSpec]:
    available_gpus = [gpu for gpu in idle_gpus if gpu not in assigned_gpus]
    specs: list[InferServiceSpec] = []
    for offset, (model_path, model_name, max_batch_size, gpu) in enumerate(
        zip(model_paths, model_names, max_batch_sizes, available_gpus, strict=False)
    ):
        port = int(base_port) + int(launched_count) + offset
        safe_name = _safe_name(model_name)
        state_db_path = None
        if state_db_dir is not None:
            state_db_path = state_db_dir / f"{safe_name}.sqlite3"
        specs.append(
            InferServiceSpec(
                model_path=model_path,
                model_name=model_name,
                gpu=str(gpu),
                port=port,
                max_batch_size=max(1, int(max_batch_size)),
                log_path=log_dir / f"{safe_name}.port{port}.log",
                state_db_path=state_db_path,
            )
        )
    return specs


def build_command(
    spec: InferServiceSpec,
    *,
    host: str,
    api_key: str,
    engine_mode: str,
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
    if api_key:
        command.extend(["--api-key", api_key])
    if spec.state_db_path is not None:
        command.extend(["--state-db-path", str(spec.state_db_path)])
    return command


def launch_service(
    spec: InferServiceSpec,
    *,
    host: str,
    api_key: str,
    engine_mode: str,
    batch_collect_ms: int,
    log_level: str,
    detach: bool,
) -> subprocess.Popen[bytes]:
    command = build_command(
        spec,
        host=host,
        api_key=api_key,
        engine_mode=engine_mode,
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


def write_manifest(
    manifest_path: Path,
    *,
    services: Sequence[RunningInferService],
    host: str,
    api_key_set: bool,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": host,
        "api_key_set": bool(api_key_set),
        "services": [
            {
                "model_path": str(service.spec.model_path),
                "model_name": service.spec.model_name,
                "gpu": service.spec.gpu,
                "port": service.spec.port,
                "max_batch_size": service.spec.max_batch_size,
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


def terminate_services(services: Sequence[RunningInferService]) -> None:
    for service in services:
        try:
            os.kill(service.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.time() + 20
    for service in services:
        while time.time() < deadline:
            try:
                os.kill(service.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        else:
            try:
                os.kill(service.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


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
                    batch_collect_ms=int(args.batch_collect_ms),
                    log_level=str(args.log_level),
                    detach=bool(args.detach),
                )
                assigned_gpus.add(spec.gpu)
                services.append(RunningInferService(spec=spec, pid=int(process.pid)))
                processes[int(process.pid)] = process
                pending_paths.pop(0)
                pending_names.pop(0)
                pending_batch_sizes.pop(0)
                write_manifest(manifest_path, services=services, host=str(args.host), api_key_set=bool(args.api_key))
                time.sleep(max(float(args.startup_stagger_s), 0.0))

        write_manifest(manifest_path, services=services, host=str(args.host), api_key_set=bool(args.api_key))
        print(f"manifest written: {manifest_path}", flush=True)
        if args.detach:
            return 0

        while processes:
            for pid, process in list(processes.items()):
                rc = process.poll()
                if rc is None:
                    continue
                service = next(item for item in services if item.pid == pid)
                print(f"infer service exited: model={service.spec.model_name} pid={pid} returncode={rc}", flush=True)
                processes.pop(pid)
                if rc != 0:
                    terminate_services([item for item in services if item.pid != pid])
                    return int(rc)
            time.sleep(2)
        return 0
    except KeyboardInterrupt:
        terminate_services(services)
        return 130


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("._") or "model"


__all__ = [
    "InferServiceSpec",
    "RunningInferService",
    "build_command",
    "launch_service",
    "main",
    "parse_args",
    "plan_deployments",
    "resolve_max_batch_sizes",
    "resolve_model_names",
    "write_manifest",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
