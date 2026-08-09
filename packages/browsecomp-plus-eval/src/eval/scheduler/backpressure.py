from __future__ import annotations

"""Remote inference backpressure signals and scheduler-owned launch budgets."""

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from .dataset_utils import safe_slug
from .remote_slots import infer_workers_for_model, parse_remote_model_slots


class RemoteBackpressureError(RuntimeError):
    pass


@dataclass(slots=True)
class RemoteModelBackpressure:
    model: str
    model_slug: str
    status: str
    route_count: int = 0
    ok_route_count: int = 0
    pending_queue: int = 0
    prefill_reserved_bsz: int = 0
    max_batch_size: int | None = None
    failed_batches: int = 0
    last_total_tok_s: float | None = None
    error: str | None = None


@dataclass(slots=True)
class RemoteConcurrencyBudget:
    model: str
    model_slug: str
    infer_max_workers: int
    remote_batch_size: int | None
    launch_allowed: bool = True
    reason: str = "static"
    pending_queue: int | None = None
    prefill_reserved_bsz: int | None = None
    max_batch_size: int | None = None
    source_status: str | None = None
    error: str | None = None

    def to_event_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "model_slug": self.model_slug,
            "infer_max_workers": int(self.infer_max_workers),
            "launch_allowed": bool(self.launch_allowed),
            "reason": self.reason,
        }
        if self.remote_batch_size is not None:
            payload["remote_batch_size"] = int(self.remote_batch_size)
        if self.pending_queue is not None:
            payload["pending_queue"] = int(self.pending_queue)
        if self.prefill_reserved_bsz is not None:
            payload["prefill_reserved_bsz"] = int(self.prefill_reserved_bsz)
        if self.max_batch_size is not None:
            payload["max_batch_size"] = int(self.max_batch_size)
        if self.source_status:
            payload["source_status"] = self.source_status
        if self.error:
            payload["error"] = self.error
        return payload


def fetch_remote_backpressure(
    *,
    base_url: str,
    api_key: str = "",
    timeout_s: float = 2.0,
) -> dict[str, RemoteModelBackpressure]:
    url = f"{_normalize_api_base(base_url)}/backpressure"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib_request.Request(url, method="GET", headers=headers)
    try:
        with urllib_request.urlopen(req, timeout=max(float(timeout_s), 0.1)) as resp:
            raw = resp.read()
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RemoteBackpressureError(f"HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RemoteBackpressureError(str(exc.reason)) from exc
    except TimeoutError as exc:
        raise RemoteBackpressureError("request timed out") from exc
    except OSError as exc:
        raise RemoteBackpressureError(str(exc)) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteBackpressureError("backpressure response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RemoteBackpressureError("backpressure response must be a JSON object")
    return parse_remote_backpressure(payload)


def parse_remote_backpressure(payload: Mapping[str, Any]) -> dict[str, RemoteModelBackpressure]:
    models = payload.get("models")
    if not isinstance(models, Mapping):
        return {}
    parsed: dict[str, RemoteModelBackpressure] = {}
    for raw_model, raw_entry in models.items():
        if not isinstance(raw_entry, Mapping):
            continue
        model = str(raw_entry.get("model") or raw_model).strip()
        if not model:
            continue
        aggregate = raw_entry.get("aggregate")
        aggregate_map = aggregate if isinstance(aggregate, Mapping) else {}
        model_slug = safe_slug(model)
        parsed[model_slug] = RemoteModelBackpressure(
            model=model,
            model_slug=model_slug,
            status=str(raw_entry.get("status") or aggregate_map.get("status") or "unknown"),
            route_count=_int_or_zero(raw_entry.get("route_count", aggregate_map.get("route_count"))),
            ok_route_count=_int_or_zero(aggregate_map.get("ok_route_count")),
            pending_queue=_int_or_zero(aggregate_map.get("pending_queue")),
            prefill_reserved_bsz=_int_or_zero(aggregate_map.get("prefill_reserved_bsz")),
            max_batch_size=_optional_int(aggregate_map.get("max_batch_size")),
            failed_batches=_int_or_zero(aggregate_map.get("failed_batches")),
            last_total_tok_s=_optional_float(aggregate_map.get("last_total_tok_s")),
            error=_optional_str(raw_entry.get("error") or aggregate_map.get("error")),
        )
    return parsed


def compute_remote_concurrency_budgets(
    *,
    infer_models: Sequence[str],
    backpressure: Mapping[str, RemoteModelBackpressure],
    default_infer_max_workers: int,
    default_remote_batch_size: int | None,
    pending_high_watermark: int = 0,
    min_infer_max_workers: int = 1,
    infer_worker_profile: str = "fixed",
) -> dict[str, RemoteConcurrencyBudget]:
    budgets: dict[str, RemoteConcurrencyBudget] = {}
    default_workers = max(1, int(default_infer_max_workers))
    min_workers = max(1, int(min_infer_max_workers))
    high_watermark = max(0, int(pending_high_watermark))
    for slot in parse_remote_model_slots(infer_models):
        model_name = slot.model
        slot_slug = slot.slot_slug
        source_model_slug = slot.model_slug
        slot_workers = infer_workers_for_model(
            model_name,
            default_workers=default_workers,
            profile=infer_worker_profile,
        )
        signal = backpressure.get(source_model_slug)
        if signal is None:
            budgets[slot_slug] = RemoteConcurrencyBudget(
                model=model_name,
                model_slug=slot_slug,
                infer_max_workers=max(min_workers, slot_workers),
                remote_batch_size=_positive_or_none(default_remote_batch_size),
                reason="static_no_backpressure",
            )
            continue

        launch_allowed = True
        reason = "backpressure_ok"
        error = signal.error
        if signal.ok_route_count <= 0 and signal.route_count > 0:
            launch_allowed = False
            reason = "no_healthy_backend"
        elif signal.pending_queue > high_watermark:
            launch_allowed = False
            reason = "backend_queue_pending"
        else:
            prefill_high_watermark = _prefill_reserved_high_watermark(signal, high_watermark)
            if signal.prefill_reserved_bsz > prefill_high_watermark:
                launch_allowed = False
                reason = "backend_prefill_reserved"

        worker_cap = _positive_or_none(signal.max_batch_size)
        infer_max_workers = max(min_workers, slot_workers)

        remote_batch_size = _positive_or_none(default_remote_batch_size)
        if remote_batch_size is not None and worker_cap is not None:
            remote_batch_size = max(1, min(remote_batch_size, worker_cap))

        budgets[slot_slug] = RemoteConcurrencyBudget(
            model=model_name,
            model_slug=slot_slug,
            infer_max_workers=infer_max_workers,
            remote_batch_size=remote_batch_size,
            launch_allowed=launch_allowed,
            reason=reason,
            pending_queue=signal.pending_queue,
            prefill_reserved_bsz=signal.prefill_reserved_bsz,
            max_batch_size=signal.max_batch_size,
            source_status=signal.status,
            error=error,
        )
    return budgets


def static_remote_concurrency_budgets(
    *,
    infer_models: Sequence[str],
    default_infer_max_workers: int,
    default_remote_batch_size: int | None,
    reason: str = "static",
    infer_worker_profile: str = "fixed",
) -> dict[str, RemoteConcurrencyBudget]:
    budgets: dict[str, RemoteConcurrencyBudget] = {}
    for slot in parse_remote_model_slots(infer_models):
        model_name = slot.model
        model_slug = slot.slot_slug
        budgets[model_slug] = RemoteConcurrencyBudget(
            model=model_name,
            model_slug=model_slug,
            infer_max_workers=infer_workers_for_model(
                model_name,
                default_workers=max(1, int(default_infer_max_workers)),
                profile=infer_worker_profile,
            ),
            remote_batch_size=_positive_or_none(default_remote_batch_size),
            reason=reason,
        )
    return budgets


def _positive_or_none(value: int | None) -> int | None:
    if value is None:
        return None
    return max(1, int(value))


def _prefill_reserved_high_watermark(signal: RemoteModelBackpressure, pending_high_watermark: int) -> int:
    if pending_high_watermark > 0:
        return int(pending_high_watermark)
    if signal.max_batch_size is not None and signal.max_batch_size > 0:
        return max(16, int(signal.max_batch_size) // 5)
    return 16


def _normalize_api_base(base_url: str) -> str:
    base = str(base_url or "").strip()
    if not base:
        raise ValueError("infer base URL cannot be empty")
    if "://" not in base:
        base = f"http://{base}"
    base = base.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: object) -> int:
    parsed = _optional_int(value)
    return 0 if parsed is None else parsed


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


__all__ = [
    "RemoteBackpressureError",
    "RemoteConcurrencyBudget",
    "RemoteModelBackpressure",
    "compute_remote_concurrency_budgets",
    "fetch_remote_backpressure",
    "parse_remote_backpressure",
    "static_remote_concurrency_budgets",
]
