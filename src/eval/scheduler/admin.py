from __future__ import annotations

"""HTTP/admin control shell for the scheduler."""

from dataclasses import asdict, dataclass, field, replace
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, cast
import uuid

from src.eval.benchmark_registry import BenchmarkField
from src.eval.evaluating import RunMode, collect_benchmark_dataset_slugs

from .actions import DispatchOptions, action_dispatch
from .config import (
    DEFAULT_ADMIN_STATE_DIR,
    DEFAULT_DISPATCH_POLL_SECONDS,
    DEFAULT_GPU_IDLE_MAX_MEM,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_GLOBS,
    DEFAULT_PID_DIR,
    DEFAULT_RUN_LOG_DIR,
)
from .control import (
    DesiredState,
    ObservedStatus,
    SchedulerProgressSnapshot,
    SchedulerRuntimeControl,
    SchedulerRuntimeFile,
)
from .dataset_utils import canonical_slug, canonicalize_benchmark_list
from .jobs import JOB_CATALOGUE, JOB_ORDER
from .models import MODEL_SELECT_CHOICES


_KNOWN_DATASET_SLUGS: tuple[str, ...] = tuple(
    sorted({canonical_slug(slug) for spec in JOB_CATALOGUE.values() for slug in spec.dataset_slugs})
)


class SchedulerAdminError(RuntimeError):
    status_code: int
    message: str

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(slots=True)
class SchedulerStartRequest:
    log_dir: str = str(DEFAULT_LOG_DIR)
    pid_dir: str = str(DEFAULT_PID_DIR)
    run_log_dir: str = str(DEFAULT_RUN_LOG_DIR)
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODEL_GLOBS))
    infer_base_url: str = ""
    infer_models: list[str] = field(default_factory=list)
    infer_api_key: str = ""
    infer_timeout_s: float = 600.0
    infer_max_workers: int = 32
    distributed_claims: bool = False
    scheduler_node_id: str = ""
    lease_duration_s: int = 900
    model_regex: list[str] = field(default_factory=list)
    model_select: str = "latest-data"
    min_param_b: float | None = None
    max_param_b: float | None = None
    only_jobs: list[str] = field(default_factory=list)
    skip_jobs: list[str] = field(default_factory=list)
    job_order: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    benchmark_fields: list[str] = field(default_factory=list)
    extra_benchmarks: list[str] = field(default_factory=list)
    only_datasets: list[str] = field(default_factory=list)
    skip_datasets: list[str] = field(default_factory=list)
    enable_param_search: bool = False
    run_mode: str = RunMode.AUTO.value
    dispatch_poll_seconds: int = DEFAULT_DISPATCH_POLL_SECONDS
    gpu_idle_max_mem: int = DEFAULT_GPU_IDLE_MAX_MEM
    skip_missing_dataset: bool = False
    clean_param_swap: bool = False
    batch_cache: str | None = None
    overwrite: bool = False
    disable_checker: bool = False
    max_concurrent_jobs: int | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SchedulerStartRequest":
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown fields: {', '.join(unknown)}")
        return cls(**payload)

    def copy(self) -> "SchedulerStartRequest":
        return replace(
            self,
            models=list(self.models),
            infer_models=list(self.infer_models),
            model_regex=list(self.model_regex),
            only_jobs=list(self.only_jobs),
            skip_jobs=list(self.skip_jobs),
            job_order=list(self.job_order),
            domains=list(self.domains),
            benchmark_fields=list(self.benchmark_fields),
            extra_benchmarks=list(self.extra_benchmarks),
            only_datasets=list(self.only_datasets),
            skip_datasets=list(self.skip_datasets),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dispatch_options(self) -> DispatchOptions:
        if self.model_select not in MODEL_SELECT_CHOICES:
            raise ValueError(f"unknown model_select={self.model_select!r}")

        job_list = _resolve_job_list(self.only_jobs, self.skip_jobs, self.domains)
        if not job_list:
            raise ValueError("no schedulable jobs remain after filtering")

        skip_dataset_slugs = _canonicalize_slugs(self.skip_datasets)
        benchmark_fields = tuple(BenchmarkField(value) for value in self.benchmark_fields)
        selected: set[str] = set()
        if benchmark_fields or self.extra_benchmarks:
            selected.update(
                collect_benchmark_dataset_slugs(
                    fields=benchmark_fields,
                    extra_benchmark_names=tuple(self.extra_benchmarks),
                )
            )
        selected.update(_canonicalize_slugs(self.only_datasets))
        only_dataset_slugs = tuple(sorted(selected))
        model_patterns = _compile_model_patterns(self.model_regex)
        job_priority = _resolve_job_priority(self.job_order, job_list)

        explicit = self.run_mode
        if self.overwrite and explicit not in (RunMode.AUTO.value, RunMode.RERUN.value):
            raise ValueError("--overwrite only supports run_mode auto/rerun")
        run_mode = RunMode.RERUN if self.overwrite else RunMode.parse(explicit)

        infer_base_url = str(self.infer_base_url or "").strip() or None
        infer_models = tuple(str(item).strip() for item in self.infer_models if str(item).strip())
        if infer_base_url or infer_models:
            if not infer_base_url:
                raise ValueError("remote inference mode requires infer_base_url")
            if not infer_models:
                raise ValueError("remote inference mode requires infer_models")
            model_globs: tuple[str, ...] = ()
        else:
            model_globs = tuple(self.models)

        batch_cache_path = Path(self.batch_cache) if self.batch_cache else None
        return DispatchOptions(
            log_dir=Path(self.log_dir),
            pid_dir=Path(self.pid_dir),
            run_log_dir=Path(self.run_log_dir),
            job_order=job_list,
            job_priority=job_priority,
            model_select=self.model_select,
            min_param_b=self.min_param_b,
            max_param_b=self.max_param_b,
            skip_dataset_slugs=skip_dataset_slugs,
            model_globs=model_globs,
            only_dataset_slugs=only_dataset_slugs,
            model_name_patterns=model_patterns,
            enable_param_search=self.enable_param_search,
            run_mode=run_mode,
            infer_base_url=infer_base_url,
            infer_models=infer_models,
            infer_api_key=str(self.infer_api_key or ""),
            infer_timeout_s=float(self.infer_timeout_s),
            infer_max_workers=int(self.infer_max_workers),
            distributed_claims=bool(self.distributed_claims),
            scheduler_node_id=(str(self.scheduler_node_id or "").strip() or None),
            lease_duration_s=int(self.lease_duration_s),
            dispatch_poll_seconds=int(self.dispatch_poll_seconds),
            gpu_idle_max_mem=int(self.gpu_idle_max_mem),
            skip_missing_dataset=self.skip_missing_dataset,
            clean_param_swap=self.clean_param_swap,
            batch_cache_path=batch_cache_path,
            disable_checker=self.disable_checker,
            max_concurrent_jobs=self.max_concurrent_jobs,
        )


@dataclass(slots=True)
class ActiveSchedulerRun:
    run_id: str
    request: SchedulerStartRequest
    options: DispatchOptions
    run_dir: Path
    runtime_control: SchedulerRuntimeControl
    thread: threading.Thread


@dataclass(slots=True)
class SchedulerRunSnapshot:
    run_id: str
    config_path: str
    control_path: str
    runtime_path: str
    desired_state: DesiredState
    runtime: SchedulerRuntimeFile
    request: SchedulerStartRequest


DispatchRunner = Callable[..., None]


class SchedulerAdminController:
    def __init__(
        self,
        *,
        state_dir: Path = DEFAULT_ADMIN_STATE_DIR,
        default_request: SchedulerStartRequest | None = None,
        dispatch_runner: DispatchRunner = action_dispatch,
    ) -> None:
        self._state_dir = state_dir
        self._default_request = default_request or SchedulerStartRequest()
        self._dispatch_runner = dispatch_runner
        self._lock = threading.Lock()
        self._active: ActiveSchedulerRun | None = None

    def draft(self) -> dict[str, Any]:
        return self._default_request.to_dict()

    def start(self, request: SchedulerStartRequest | None = None) -> SchedulerRunSnapshot:
        payload = (request or self._default_request).copy()
        options = payload.to_dispatch_options()

        with self._lock:
            active = self._active
            if active is not None:
                snapshot = self._snapshot_locked(active)
                if not snapshot.runtime.status_enum().is_terminal():
                    raise SchedulerAdminError(HTTPStatus.CONFLICT, "an evaluation task is already active")

            run_id = time.strftime("run-%Y%m%dT%H%M%S") + f"-{uuid.uuid4().hex[:8]}"
            run_dir = self._state_dir / run_id
            runtime_control = SchedulerRuntimeControl.from_dir(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "request.json"
            config_path.write_text(
                json.dumps(payload.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            runtime_control.write_desired_state(DesiredState.RUNNING)
            runtime_control.write_status(ObservedStatus.STARTING)

            thread = threading.Thread(
                target=self._run_dispatch,
                name=f"rwkv-scheduler-{run_id}",
                args=(options, runtime_control),
                daemon=True,
            )
            active = ActiveSchedulerRun(
                run_id=run_id,
                request=payload,
                options=options,
                run_dir=run_dir,
                runtime_control=runtime_control,
                thread=thread,
            )
            self._active = active
            thread.start()
            return self._snapshot_locked(active)

    def pause(self) -> SchedulerRunSnapshot:
        return self._set_desired_state(DesiredState.PAUSED)

    def resume(self) -> SchedulerRunSnapshot:
        return self._set_desired_state(DesiredState.RUNNING)

    def cancel(self) -> SchedulerRunSnapshot:
        return self._set_desired_state(DesiredState.CANCELLED)

    def snapshot(self) -> SchedulerRunSnapshot | None:
        with self._lock:
            active = self._active
            if active is None:
                return None
            return self._snapshot_locked(active)

    def _set_desired_state(self, desired_state: DesiredState) -> SchedulerRunSnapshot:
        with self._lock:
            active = self._active
            if active is None:
                raise SchedulerAdminError(HTTPStatus.NOT_FOUND, "no scheduler run has been started")
            snapshot = self._snapshot_locked(active)
            if snapshot.runtime.status_enum().is_terminal():
                raise SchedulerAdminError(
                    HTTPStatus.CONFLICT,
                    f"scheduler is already {snapshot.runtime.observed_status}",
                )
            active.runtime_control.write_desired_state(desired_state)
            return self._snapshot_locked(active)

    def _snapshot_locked(self, active: ActiveSchedulerRun) -> SchedulerRunSnapshot:
        desired_state, runtime = active.runtime_control.snapshot()
        if runtime is None:
            runtime = active.runtime_control.write_status(ObservedStatus.STARTING)
        return SchedulerRunSnapshot(
            run_id=active.run_id,
            config_path=str(active.run_dir / "request.json"),
            control_path=str(active.runtime_control.control_path),
            runtime_path=str(active.runtime_control.runtime_path),
            desired_state=desired_state,
            runtime=runtime,
            request=active.request,
        )

    def _run_dispatch(self, options: DispatchOptions, runtime_control: SchedulerRuntimeControl) -> None:
        try:
            self._dispatch_runner(options, runtime_control=runtime_control)
            runtime = runtime_control.read_runtime_file()
            if runtime is None or not runtime.status_enum().is_terminal():
                desired_state = runtime_control.desired_state()
                terminal = ObservedStatus.CANCELLED if desired_state is DesiredState.CANCELLED else ObservedStatus.COMPLETED
                runtime_control.write_status(
                    terminal,
                    progress=_progress_from_runtime(runtime),
                )
        except Exception as exc:  # noqa: BLE001
            runtime_control.write_status(
                ObservedStatus.FAILED,
                error=str(exc),
                progress=_progress_from_runtime(runtime_control.read_runtime_file()),
            )


class SchedulerAdminHttpServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        controller: SchedulerAdminController,
        *,
        api_key: str | None = None,
    ) -> None:
        super().__init__(server_address, SchedulerAdminRequestHandler)
        self.controller = controller
        self.api_key = api_key


class SchedulerAdminRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path in {"/", "/admin"}:
                server = cast(SchedulerAdminHttpServer, self.server)
                self._write_html(HTTPStatus.OK, _render_admin_shell(server.controller.draft()))
                return
            if self.path == "/api":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "name": "rwkv-skills-scheduler-admin",
                        "routes": [
                            "/api/health",
                            "/api/v1/admin/health",
                            "/api/v1/admin/eval/draft",
                            "/api/v1/admin/eval/status",
                            "/api/v1/admin/eval/start",
                            "/api/v1/admin/eval/pause",
                            "/api/v1/admin/eval/resume",
                            "/api/v1/admin/eval/cancel",
                        ],
                    },
                )
                return
            if self.path == "/api/health":
                self._write_json(HTTPStatus.OK, {"status": "ok"})
                return
            if self.path == "/api/v1/admin/health":
                if not self._check_auth():
                    return
                server = cast(SchedulerAdminHttpServer, self.server)
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "active": server.controller.snapshot() is not None,
                    },
                )
                return
            if self.path == "/api/v1/admin/eval/draft":
                if not self._check_auth():
                    return
                server = cast(SchedulerAdminHttpServer, self.server)
                self._write_json(HTTPStatus.OK, server.controller.draft())
                return
            if self.path == "/api/v1/admin/eval/status":
                if not self._check_auth():
                    return
                server = cast(SchedulerAdminHttpServer, self.server)
                self._write_json(HTTPStatus.OK, build_status_response(server.controller.snapshot()))
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except SchedulerAdminError as exc:
            self._write_json(exc.status_code, {"error": exc.message})

    def do_POST(self) -> None:  # noqa: N802
        if not self._check_auth():
            return
        try:
            server = cast(SchedulerAdminHttpServer, self.server)
            if self.path == "/api/v1/admin/eval/start":
                payload = self._read_json_body()
                request = SchedulerStartRequest.from_payload(payload)
                snapshot = server.controller.start(request)
                self._write_json(HTTPStatus.OK, build_status_response(snapshot))
                return
            if self.path == "/api/v1/admin/eval/pause":
                self._write_json(HTTPStatus.OK, build_status_response(server.controller.pause()))
                return
            if self.path == "/api/v1/admin/eval/resume":
                self._write_json(HTTPStatus.OK, build_status_response(server.controller.resume()))
                return
            if self.path == "/api/v1/admin/eval/cancel":
                self._write_json(HTTPStatus.OK, build_status_response(server.controller.cancel()))
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except SchedulerAdminError as exc:
            self._write_json(exc.status_code, {"error": exc.message})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _check_auth(self) -> bool:
        api_key = cast(SchedulerAdminHttpServer, self.server).api_key
        if not api_key:
            return True
        header = self.headers.get("Authorization", "")
        if header == f"Bearer {api_key}":
            return True
        self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        return False

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return {}
        length = int(raw_length)
        if length <= 0:
            return {}
        payload = self.rfile.read(length)
        if not payload:
            return {}
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return cast(dict[str, Any], data)

    def _write_json(self, status: int | HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_html(self, status: int | HTTPStatus, content: str) -> None:
        body = content.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_status_response(snapshot: SchedulerRunSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return {
            "status": "idle",
            "desired_state": None,
            "config_path": None,
            "control_path": None,
            "runtime_path": None,
            "started_at_unix_ms": None,
            "updated_at_unix_ms": None,
            "finished_at_unix_ms": None,
            "error": None,
            "pending_jobs": 0,
            "running_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "tasks_total": 0,
            "progress_percent": 0.0,
            "queue_head": [],
            "active_jobs": [],
            "available_gpus": [],
            "request": None,
            "run_id": None,
        }

    runtime = snapshot.runtime
    tasks_total = runtime.pending_jobs + runtime.running_jobs + runtime.completed_jobs + runtime.failed_jobs
    progress_numerator = runtime.completed_jobs + runtime.failed_jobs
    progress_percent = 0.0 if tasks_total == 0 else min(progress_numerator / tasks_total, 1.0)
    return {
        "status": runtime.observed_status,
        "desired_state": snapshot.desired_state.value,
        "config_path": snapshot.config_path,
        "control_path": snapshot.control_path,
        "runtime_path": snapshot.runtime_path,
        "started_at_unix_ms": runtime.started_at_unix_ms,
        "updated_at_unix_ms": runtime.updated_at_unix_ms,
        "finished_at_unix_ms": runtime.finished_at_unix_ms,
        "error": runtime.error,
        "pending_jobs": runtime.pending_jobs,
        "running_jobs": runtime.running_jobs,
        "completed_jobs": runtime.completed_jobs,
        "failed_jobs": runtime.failed_jobs,
        "tasks_total": tasks_total,
        "progress_percent": progress_percent,
        "queue_head": runtime.queue_head,
        "active_jobs": runtime.active_jobs,
        "available_gpus": runtime.available_gpus,
        "request": snapshot.request.to_dict(),
        "run_id": snapshot.run_id,
    }


def serve_scheduler_admin(
    *,
    host: str,
    port: int,
    controller: SchedulerAdminController,
    api_key: str | None = None,
) -> None:
    server = SchedulerAdminHttpServer((host, int(port)), controller, api_key=api_key)
    print(f"🌐 Scheduler admin listening on http://{host}:{port}")
    server.serve_forever()


def _resolve_job_list(
    include: list[str],
    exclude: list[str],
    domains: list[str],
) -> tuple[str, ...]:
    order = list(JOB_ORDER)
    if domains:
        allowed_domains = set(domains)
        order = [job for job in order if JOB_CATALOGUE[job].domain in allowed_domains]
    if include:
        allowed_jobs = set(include)
        order = [job for job in order if job in allowed_jobs]
    if exclude:
        blocked_jobs = set(exclude)
        order = [job for job in order if job not in blocked_jobs]
    return tuple(order)


def _canonicalize_slugs(slugs: list[str]) -> tuple[str, ...]:
    if not slugs:
        return tuple()
    return canonicalize_benchmark_list(slugs, known_slugs=_KNOWN_DATASET_SLUGS)


def _compile_model_patterns(patterns: list[str]) -> tuple[re.Pattern[str], ...]:
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        compiled.append(re.compile(raw))
    return tuple(compiled)


def _resolve_job_priority(priority: list[str], available: tuple[str, ...]) -> tuple[str, ...] | None:
    if not priority:
        return None
    allowed = set(available)
    ordered: list[str] = []
    for job in priority:
        if job in allowed and job not in ordered:
            ordered.append(job)
    return tuple(ordered) if ordered else None


def _progress_from_runtime(runtime: SchedulerRuntimeFile | None) -> SchedulerProgressSnapshot:
    if runtime is None:
        return SchedulerProgressSnapshot()
    return SchedulerProgressSnapshot(
        pending_jobs=runtime.pending_jobs,
        running_jobs=runtime.running_jobs,
        completed_jobs=runtime.completed_jobs,
        failed_jobs=runtime.failed_jobs,
        queue_head=tuple(runtime.queue_head),
        active_jobs=tuple(runtime.active_jobs),
        available_gpus=tuple(runtime.available_gpus),
    )


def _render_admin_shell(draft_payload: dict[str, Any]) -> str:
    payload = html.escape(json.dumps(draft_payload, ensure_ascii=False, indent=2, sort_keys=True))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RWKV Scheduler Admin</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #22c55e;
      --danger: #ef4444;
      --border: #334155;
      --mono: "Iosevka", "SFMono-Regular", Consolas, monospace;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #020617 0%, #111827 100%);
      color: var(--text);
      font: 15px/1.5 system-ui, sans-serif;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: 1.2fr 0.8fr;
    }}
    .card {{
      background: rgba(17, 24, 39, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    p {{
      margin: 0 0 12px;
      color: var(--muted);
    }}
    textarea, input, pre {{
      width: 100%;
      box-sizing: border-box;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      padding: 12px;
      font: 13px/1.5 var(--mono);
    }}
    textarea {{
      min-height: 420px;
      resize: vertical;
    }}
    input {{
      margin-bottom: 12px;
    }}
    .row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 12px 0 0;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      font-weight: 700;
      cursor: pointer;
    }}
    .primary {{ background: var(--accent); color: #052e16; }}
    .secondary {{ background: #eab308; color: #422006; }}
    .danger {{ background: var(--danger); color: white; }}
    .ghost {{ background: #334155; color: var(--text); }}
    .status {{
      min-height: 420px;
      white-space: pre-wrap;
      overflow: auto;
    }}
    .hint {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      textarea, .status {{ min-height: 300px; }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="card" style="margin-bottom:16px">
      <h1>RWKV Scheduler Admin</h1>
      <p>集中式 scheduler 状态机和 HTTP/admin 控制壳。通过 JSON payload 启动 dispatch，后续用 pause / resume / cancel 控制。</p>
    </div>
    <div class="grid">
      <section class="card">
        <input id="token" type="password" placeholder="Bearer token，可留空">
        <textarea id="payload">{payload}</textarea>
        <div class="row">
          <button class="primary" id="start">Start</button>
          <button class="secondary" id="pause">Pause</button>
          <button class="ghost" id="resume">Resume</button>
          <button class="danger" id="cancel">Cancel</button>
          <button class="ghost" id="refresh">Refresh</button>
        </div>
        <div class="hint">`/api/v1/admin/eval/start` 接收的就是上面的 JSON。</div>
      </section>
      <section class="card">
        <pre class="status" id="status"></pre>
      </section>
    </div>
  </main>
  <script>
    const tokenInput = document.getElementById('token');
    const payloadInput = document.getElementById('payload');
    const statusEl = document.getElementById('status');

    async function api(path, method = 'GET', body = null) {{
      const headers = {{ 'Content-Type': 'application/json' }};
      if (tokenInput.value.trim()) {{
        headers['Authorization'] = `Bearer ${{tokenInput.value.trim()}}`;
      }}
      const resp = await fetch(path, {{
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
      }});
      const text = await resp.text();
      let data = null;
      try {{
        data = text ? JSON.parse(text) : null;
      }} catch (error) {{
        data = {{ error: text }};
      }}
      if (!resp.ok) {{
        throw new Error(JSON.stringify(data, null, 2));
      }}
      return data;
    }}

    async function refresh() {{
      try {{
        const data = await api('/api/v1/admin/eval/status');
        statusEl.textContent = JSON.stringify(data, null, 2);
      }} catch (error) {{
        statusEl.textContent = String(error);
      }}
    }}

    async function callControl(path) {{
      try {{
        const data = await api(path, 'POST');
        statusEl.textContent = JSON.stringify(data, null, 2);
      }} catch (error) {{
        statusEl.textContent = String(error);
      }}
    }}

    document.getElementById('start').onclick = async () => {{
      try {{
        const body = JSON.parse(payloadInput.value || '{{}}');
        const data = await api('/api/v1/admin/eval/start', 'POST', body);
        statusEl.textContent = JSON.stringify(data, null, 2);
      }} catch (error) {{
        statusEl.textContent = String(error);
      }}
    }};
    document.getElementById('pause').onclick = () => callControl('/api/v1/admin/eval/pause');
    document.getElementById('resume').onclick = () => callControl('/api/v1/admin/eval/resume');
    document.getElementById('cancel').onclick = () => callControl('/api/v1/admin/eval/cancel');
    document.getElementById('refresh').onclick = refresh;
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>"""


__all__ = [
    "SchedulerAdminController",
    "SchedulerAdminError",
    "SchedulerAdminHttpServer",
    "SchedulerRunSnapshot",
    "SchedulerStartRequest",
    "build_status_response",
    "serve_scheduler_admin",
]
