"""FastAPI admin layer for the RWKV Skills evaluation scheduler.

Wraps the existing :class:`SchedulerAdminController` (draft / start / pause /
resume / cancel / snapshot) and exposes it under ``/api/admin/*`` so the Next
frontend can drive evaluation runs through the FastAPI JSON API. Two
gap-fillers are added on top of the controller:

  * ``/api/admin/eval/options`` — valid choices for the start form (jobs,
    domains, model-select, worker-profile, protocol, run-mode).
  * ``/api/admin/backpressure`` — live GPU / remote-worker telemetry, proxied
    from the infer server's ``/v1/backpressure`` endpoint plus the local GPUs
    reported by the active scheduler run.

Auth is optional: set ``RWKV_ADMIN_API_KEY`` to require an
``Authorization: Bearer <key>`` header on every ``/api/admin/*`` request.
"""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path
import threading
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, Header, HTTPException, Query

# The scheduler / infer modules below pull in heavy dependencies (torch, the
# full eval stack) and rely on sibling modules that may be absent in a trimmed
# checkout. We import them *lazily* — inside ``_deps()`` — so that merely
# importing this module, and thus building the dashboard app, never fails. A
# genuinely missing dependency degrades only the ``/api/admin/*`` routes
# (HTTP 503), leaving the leaderboard / eval routes fully functional.

_deps_lock = threading.Lock()
_deps_cache: SimpleNamespace | None = None
_deps_error: str | None = None


def _deps() -> SimpleNamespace:
    """Import the scheduler/infer symbols on first use; cache the result.

    Raises ``HTTPException(503)`` (not ``ImportError``) when the scheduler stack
    cannot be imported, so a missing dependency surfaces as a clean per-route
    error instead of crashing the whole dashboard at import time.
    """
    global _deps_cache, _deps_error
    with _deps_lock:
        if _deps_cache is None and _deps_error is None:
            try:
                from src.eval.evaluating.task_persistence import RunMode
                from src.eval.scheduler.admin import (
                    SchedulerAdminController,
                    SchedulerAdminError,
                    SchedulerStartRequest,
                    build_status_response,
                )
                from src.eval.scheduler.backpressure import (
                    RemoteBackpressureError,
                    fetch_remote_backpressure,
                )
                from src.eval.scheduler.jobs import JOB_CATALOGUE, JOB_ORDER
                from src.eval.scheduler.models import MODEL_SELECT_CHOICES
                from src.eval.scheduler.remote_slots import INFER_WORKER_PROFILE_CHOICES
                from src.infer.backend import REMOTE_INFERENCE_PROTOCOL_CHOICES

                _deps_cache = SimpleNamespace(
                    RunMode=RunMode,
                    SchedulerAdminController=SchedulerAdminController,
                    SchedulerAdminError=SchedulerAdminError,
                    SchedulerStartRequest=SchedulerStartRequest,
                    build_status_response=build_status_response,
                    RemoteBackpressureError=RemoteBackpressureError,
                    fetch_remote_backpressure=fetch_remote_backpressure,
                    JOB_CATALOGUE=JOB_CATALOGUE,
                    JOB_ORDER=JOB_ORDER,
                    MODEL_SELECT_CHOICES=MODEL_SELECT_CHOICES,
                    INFER_WORKER_PROFILE_CHOICES=INFER_WORKER_PROFILE_CHOICES,
                    REMOTE_INFERENCE_PROTOCOL_CHOICES=REMOTE_INFERENCE_PROTOCOL_CHOICES,
                )
            except Exception as exc:  # noqa: BLE001
                _deps_error = f"{type(exc).__name__}: {exc}"
        if _deps_cache is None:
            raise HTTPException(
                status_code=503,
                detail=f"调度器 admin 不可用（依赖缺失）：{_deps_error}",
            )
        return _deps_cache


# Process-wide controller, built lazily on first admin request.
_controller_lock = threading.Lock()
_controller: Any = None


def _get_controller() -> Any:
    global _controller
    deps = _deps()
    with _controller_lock:
        if _controller is None:
            _controller = deps.SchedulerAdminController()
        return _controller


def _admin_api_key() -> str:
    # Read on each call so a key placed in .env (loaded after import) is honored.
    return (os.environ.get("RWKV_ADMIN_API_KEY") or "").strip()


def _check_auth(authorization: str | None) -> None:
    key = _admin_api_key()
    if not key:
        return
    if authorization == f"Bearer {key}":
        return
    raise HTTPException(status_code=401, detail="unauthorized")


def _options_payload() -> dict[str, Any]:
    deps = _deps()
    catalogue = deps.JOB_CATALOGUE
    domains = sorted({spec.domain for spec in catalogue.values()})
    jobs = [
        {"name": name, "domain": catalogue[name].domain}
        for name in deps.JOB_ORDER
        if name in catalogue
    ]
    return {
        "jobs": jobs,
        "domains": domains,
        "model_select": list(deps.MODEL_SELECT_CHOICES),
        "worker_profile": list(deps.INFER_WORKER_PROFILE_CHOICES),
        "protocol": list(deps.REMOTE_INFERENCE_PROTOCOL_CHOICES),
        "run_mode": [mode.value for mode in deps.RunMode],
    }


def _draft_payload() -> dict[str, Any]:
    payload = dict(_get_controller().draft())
    infer_base_url = (os.environ.get("RWKV_ADMIN_INFER_BASE_URL") or "").strip()
    infer_models = [
        item.strip()
        for item in (os.environ.get("RWKV_ADMIN_INFER_MODELS") or "").split(",")
        if item.strip()
    ]
    if infer_base_url and infer_models:
        payload["infer_base_url"] = infer_base_url
        payload["infer_models"] = infer_models
        payload["models"] = []
        return payload

    configured = [str(item) for item in payload.get("models", []) if str(item).strip()]
    if any(glob(pattern) for pattern in configured):
        return payload

    override = (os.environ.get("RWKV_ADMIN_MODEL_GLOB") or "").strip()
    candidates = [override] if override else []
    candidates.append(str(Path.home() / "weights" / "rwkv7-*.pth"))
    for pattern in candidates:
        if pattern and glob(pattern):
            payload["models"] = [pattern]
            break
    return payload


def _validate_start_payload(payload: dict[str, Any]) -> dict[str, Any]:
    deps = _deps()
    request = deps.SchedulerStartRequest.from_payload(payload)
    options = request.to_dispatch_options()
    base_url = str(request.infer_base_url or "").rstrip("/")
    warnings: list[str] = []

    if base_url:
        headers = {"Accept": "application/json"}
        if request.infer_api_key:
            headers["Authorization"] = f"Bearer {request.infer_api_key}"
        try:
            import json

            with urlopen(Request(f"{base_url}/models", headers=headers), timeout=3.0) as response:
                body = json.loads(response.read().decode("utf-8"))
            available_models = [str(item.get("id")) for item in body.get("data", []) if item.get("id")]
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise ValueError(f"推理服务不可用：{exc}") from exc
        missing = sorted(set(request.infer_models) - set(available_models))
        if missing:
            raise ValueError(f"推理服务中不存在模型：{', '.join(missing)}")
        mode = "remote"
        models = list(request.infer_models)
    else:
        matches = sorted({path for pattern in request.models for path in glob(pattern)})
        if not matches:
            raise ValueError("本地模型路径没有匹配到任何权重文件")
        mode = "local"
        models = matches
        if len(matches) > 8:
            warnings.append(f"本地路径匹配到 {len(matches)} 个模型，启动任务可能较多")

    if len(options.job_order) > 10:
        warnings.append(f"当前配置包含 {len(options.job_order)} 个任务类型，建议先选择 domain 或 only_jobs")

    return {
        "valid": True,
        "mode": mode,
        "models": models,
        "job_count": len(options.job_order),
        "jobs": list(options.job_order),
        "run_mode": request.run_mode,
        "warnings": warnings,
    }


def register_admin_routes(app: FastAPI) -> None:
    """Mount the ``/api/admin/*`` routes onto an existing FastAPI app.

    Kept as an explicit registration helper so tests and alternative app
    factories can share the same route wiring.
    """

    @app.get("/api/admin/health")
    def admin_health(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _check_auth(authorization)
        # Health must never 503: report scheduler availability instead.
        try:
            snapshot = _get_controller().snapshot()
        except HTTPException as exc:
            return {
                "status": "unavailable",
                "active": False,
                "auth_required": bool(_admin_api_key()),
                "detail": exc.detail,
            }
        return {
            "status": "ok",
            "active": snapshot is not None,
            "auth_required": bool(_admin_api_key()),
        }

    @app.get("/api/admin/eval/options")
    def admin_options(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _check_auth(authorization)
        return _options_payload()

    @app.get("/api/admin/eval/draft")
    def admin_draft(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _check_auth(authorization)
        return _draft_payload()

    @app.get("/api/admin/eval/status")
    def admin_status(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _check_auth(authorization)
        deps = _deps()
        return deps.build_status_response(_get_controller().snapshot())

    @app.post("/api/admin/eval/start")
    def admin_start(
        payload: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(authorization)
        deps = _deps()
        try:
            request = deps.SchedulerStartRequest.from_payload(payload)
            snapshot = _get_controller().start(request)
        except deps.SchedulerAdminError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return deps.build_status_response(snapshot)

    @app.post("/api/admin/eval/validate")
    def admin_validate(
        payload: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(authorization)
        try:
            return _validate_start_payload(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _control(action: str) -> dict[str, Any]:
        deps = _deps()
        controller = _get_controller()
        try:
            if action == "pause":
                snapshot = controller.pause()
            elif action == "resume":
                snapshot = controller.resume()
            else:
                snapshot = controller.cancel()
        except deps.SchedulerAdminError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
        return deps.build_status_response(snapshot)

    @app.post("/api/admin/eval/pause")
    def admin_pause(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _check_auth(authorization)
        return _control("pause")

    @app.post("/api/admin/eval/resume")
    def admin_resume(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _check_auth(authorization)
        return _control("resume")

    @app.post("/api/admin/eval/cancel")
    def admin_cancel(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _check_auth(authorization)
        return _control("cancel")

    @app.get("/api/admin/backpressure")
    def admin_backpressure(
        infer_base_url: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(authorization)
        deps = _deps()
        snapshot = _get_controller().snapshot()
        request = snapshot.request if snapshot is not None else None

        # Local GPUs reported by the active run.
        available_gpus = list(snapshot.runtime.available_gpus) if snapshot is not None else []

        base_url = (infer_base_url or (request.infer_base_url if request else "") or "").strip()
        api_key = request.infer_api_key if request else ""
        timeout_s = float(request.infer_backpressure_timeout_s) if request else 2.0

        models: list[dict[str, Any]] = []
        error: str | None = None
        if base_url:
            try:
                signals = deps.fetch_remote_backpressure(
                    base_url=base_url, api_key=api_key, timeout_s=timeout_s
                )
                models = [asdict(sig) for sig in signals.values()]
            except (deps.RemoteBackpressureError, ValueError) as exc:
                error = str(exc)

        return {
            "infer_base_url": base_url,
            "available_gpus": available_gpus,
            "models": models,
            "error": error,
        }


__all__ = ["register_admin_routes"]
