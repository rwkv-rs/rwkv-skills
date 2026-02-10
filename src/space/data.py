"""Helpers to load score records from SQL and normalise them for the space dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from src.db.eval_db_service import EvalDbService
from src.db.orm import init_orm
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import detect_job_from_dataset


ARCH_VERSIONS = ("rwkv7", "rwkv7a", "rwkv7b")
DATA_VERSIONS = (
    "g0",
    "g0a",
    "g0a2",
    "g0a3",
    "g0a4",
    "g0b",
    "g0c",
    "g1",
    "g1a",
    "g1a2",
    "g1a3",
    "g1a4",
    "g1b",
    "g1c",
    "g1d",
)
NUM_PARAMS = ("0_1b", "0_4b", "1_5b", "2_9b", "7_2b", "13_3b")
DB_PLACEHOLDER_PATH = Path("<db>")


def _record_error(message: str, errors: list[str] | None) -> None:
    print(f"[space] {message}", file=sys.stderr)
    if errors is not None:
        errors.append(message)


def _canonical_pass_key(key: str) -> str | None:
    token = key.strip().lower().replace("@", "").replace("_", "").replace("-", "").replace(" ", "")
    if not token.startswith("pass"):
        return None
    suffix = token[len("pass") :]
    if suffix.startswith("at"):
        suffix = suffix[2:]
    if suffix.isdigit():
        return f"pass@{int(suffix)}"
    return None


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("value", "score", "accuracy"):
            nested = value.get(key)
            if isinstance(nested, (int, float, str)) and not isinstance(nested, bool):
                parsed = _numeric(nested)
                if parsed is not None:
                    return parsed
    return None


def _collect_pass_metrics(source: Any) -> dict[str, float]:
    """Extract pass@k style metrics from loosely structured payloads."""

    collected: dict[str, float] = {}
    if source is None or isinstance(source, bool):
        return collected
    if isinstance(source, (int, float)):
        collected["pass@1"] = float(source)
        return collected
    if not isinstance(source, dict):
        return collected

    for key, value in source.items():
        canonical = _canonical_pass_key(str(key))
        number = _numeric(value)
        if canonical and number is not None:
            collected[canonical] = number
            continue
        if str(key).lower() == "score" and number is not None:
            collected.setdefault("pass@1", number)
            continue
        if isinstance(value, dict):
            nested = _collect_pass_metrics(value)
            for nested_key, nested_value in nested.items():
                collected.setdefault(nested_key, nested_value)
    return collected


def _normalize_metrics(payload: dict[str, Any], *, dataset: str, is_cot: bool, task: str | None) -> dict[str, Any]:
    raw_metrics = payload.get("metrics") if isinstance(payload, dict) else None
    metrics: dict[str, Any] = {}
    if isinstance(raw_metrics, dict):
        for key, value in raw_metrics.items():
            canonical = _canonical_pass_key(str(key))
            if canonical:
                parsed = _numeric(value)
                metrics[canonical] = parsed if parsed is not None else value
            else:
                metrics[key] = value

    job_name = detect_job_from_dataset(dataset, is_cot=is_cot)
    slug = canonical_slug(dataset)
    normalized_slug = "".join(ch for ch in slug if ch.isalnum()).lower()
    code_like = job_name in {"code_human_eval", "code_mbpp", "code_livecodebench"} or (
        task and task.startswith("code")
    ) or (
        "humaneval" in normalized_slug or "mbpp" in normalized_slug or "livecodebench" in normalized_slug
    )
    if not code_like:
        return metrics

    for source in (raw_metrics, payload.get("score"), payload.get("scores"), payload):
        for key, value in _collect_pass_metrics(source).items():
            metrics.setdefault(key, value)
    return metrics


@dataclass(slots=True, frozen=True)
class ModelSignature:
    arch: str | None
    data: str | None
    params: str | None
    arch_rank: int | None
    data_rank: int | None
    param_rank: int | None

    def data_key(self) -> int:
        return self.data_rank if self.data_rank is not None else -1


@dataclass(slots=True, frozen=True)
class ScoreEntry:
    dataset: str
    model: str
    metrics: dict[str, Any]
    samples: int
    problems: int | None
    created_at: datetime
    log_path: str
    cot: bool
    task: str | None
    task_details: dict[str, Any] | None
    path: Path
    relative_path: Path
    domain: str
    extra: dict[str, Any]
    arch_version: str | None
    data_version: str | None
    num_params: str | None


def _normalize_token(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _match_rank(text: str, ordered: tuple[str, ...]) -> tuple[str | None, int | None]:
    normalized = _normalize_token(text)
    best_rank: int | None = None
    best_token: str | None = None
    for idx, token in enumerate(ordered):
        tok_norm = _normalize_token(token)
        if tok_norm and tok_norm in normalized:
            if best_rank is None or idx > best_rank:
                best_rank = idx
                best_token = token.upper()
    return best_token, best_rank


def parse_model_signature(model: str) -> ModelSignature:
    arch, arch_rank = _match_rank(model, ARCH_VERSIONS)
    data, data_rank = _match_rank(model, DATA_VERSIONS)
    params, param_rank = _match_rank(model, NUM_PARAMS)
    return ModelSignature(
        arch=arch,
        data=data,
        params=params,
        arch_rank=arch_rank,
        data_rank=data_rank,
        param_rank=param_rank,
    )


def _parse_created_at(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=None)
    if isinstance(raw, str):
        try:
            cleaned = raw.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(cleaned)
            return parsed.replace(tzinfo=None)
        except ValueError:
            pass
    return datetime.utcnow()


def _parse_int(
    value: Any,
    *,
    field: str,
    default: int | None,
    errors: list[str] | None,
) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        _record_error(f"数据库字段 {field} 无法解析为整数: {value!r}", errors)
        return default


def _infer_domain(dataset_slug: str, *, is_cot: bool, task: str | None) -> str:
    slug = canonical_slug(dataset_slug)
    if slug.startswith("mmlu"):
        return "mmlu系列"
    job = detect_job_from_dataset(slug, is_cot=is_cot)
    if job in {"code_human_eval", "code_mbpp", "code_livecodebench"}:
        return "coding系列"
    if job == "instruction_following":
        return "instruction following系列"
    if job in {"free_response", "free_response_judge"}:
        return "math reasoning系列"
    if job in {"multi_choice_plain", "multi_choice_cot"}:
        return "multi-choice系列"
    if task:
        if "code" in task:
            return "coding系列"
        if "instruction" in task:
            return "instruction following系列"
    return "其他"


def _score_entry_from_db(payload: dict[str, Any], errors: list[str] | None) -> ScoreEntry | None:
    dataset = canonical_slug(str(payload.get("dataset", "")).strip())
    model = str(payload.get("model", "")).strip()
    if not dataset or not model:
        return None

    is_cot = bool(payload.get("cot", False))
    task = str(payload.get("task")).strip() if payload.get("task") else None
    metrics = _normalize_metrics(payload, dataset=dataset, is_cot=is_cot, task=task)
    created_at = _parse_created_at(payload.get("created_at"))
    samples = _parse_int(payload.get("samples"), field="samples", default=0, errors=errors)
    problems = _parse_int(payload.get("problems"), field="problems", default=None, errors=errors)
    log_path = str(payload.get("log_path") or "")
    task_details = payload.get("task_details") if isinstance(payload.get("task_details"), dict) else None
    domain = _infer_domain(dataset, is_cot=is_cot, task=task)

    sig = parse_model_signature(model)

    known_keys = {
        "task_id",
        "dataset",
        "model",
        "metrics",
        "samples",
        "problems",
        "created_at",
        "log_path",
        "task",
        "task_details",
        "cot",
        "is_param_search",
    }
    extra = {k: v for k, v in payload.items() if k not in known_keys}
    task_id = payload.get("task_id")
    relative = Path(str(task_id)) if task_id is not None else DB_PLACEHOLDER_PATH

    return ScoreEntry(
        dataset=dataset,
        model=model,
        metrics=metrics,
        samples=samples if samples is not None else 0,
        problems=problems,
        created_at=created_at,
        log_path=log_path,
        cot=is_cot,
        task=task,
        task_details=task_details,
        path=DB_PLACEHOLDER_PATH,
        relative_path=relative,
        domain=domain,
        extra=extra,
        arch_version=sig.arch,
        data_version=sig.data,
        num_params=sig.params,
    )


def load_scores(errors: list[str] | None = None) -> list[ScoreEntry]:
    try:
        init_orm(DEFAULT_DB_CONFIG)
    except Exception as exc:  # noqa: BLE001
        _record_error(f"初始化数据库失败: {exc}", errors)
        return []

    try:
        rows = EvalDbService().list_latest_scores_for_space(include_param_search=False)
    except Exception as exc:  # noqa: BLE001
        _record_error(f"读取数据库分数失败: {exc}", errors)
        return []

    entries: list[ScoreEntry] = []
    for row in rows:
        payload = dict(row) if isinstance(row, Mapping) else None
        if payload is None:
            continue
        entry = _score_entry_from_db(payload, errors)
        if entry is not None:
            entries.append(entry)
    return entries


def pick_latest_model(entries: Iterable[ScoreEntry]) -> str | None:
    by_model: dict[str, list[ScoreEntry]] = {}
    for entry in entries:
        by_model.setdefault(entry.model, []).append(entry)

    latest_model: str | None = None
    best_key: tuple[int, float] | None = None
    for model, items in by_model.items():
        sig = parse_model_signature(model)
        newest_time = max(item.created_at for item in items).timestamp()
        key = (sig.data_key(), newest_time)
        if best_key is None or key > best_key:
            best_key = key
            latest_model = model
    return latest_model


def latest_entries_for_model(entries: Iterable[ScoreEntry], model: str | None) -> list[ScoreEntry]:
    if not model:
        return []
    latest: dict[tuple[str, bool, str | None], ScoreEntry] = {}
    for entry in entries:
        if entry.model != model:
            continue
        key = (entry.dataset, entry.cot, entry.task)
        previous = latest.get(key)
        if previous is None or entry.created_at > previous.created_at:
            latest[key] = entry
    return sorted(
        latest.values(),
        key=lambda item: (item.domain, item.dataset, item.task or ""),
    )


def list_models(entries: Iterable[ScoreEntry]) -> list[str]:
    return sorted({entry.model for entry in entries})


def list_domains(entries: Iterable[ScoreEntry]) -> list[str]:
    return sorted({entry.domain for entry in entries})


__all__ = [
    "ScoreEntry",
    "list_domains",
    "list_models",
    "load_scores",
    "pick_latest_model",
    "latest_entries_for_model",
    "parse_model_signature",
]
