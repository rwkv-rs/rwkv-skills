"""Helpers to load score artifacts and normalise them for the space dashboard."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.eval.results.layout import SCORES_ROOT
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import detect_job_from_dataset


ARCH_VERSIONS = ("rwkv7", "rwkv7a", "rwkv7b")
DATA_VERSIONS = ("g0", "g0a", "g0a2", "g0a3", "g0a4", "g0b", "g0c", "g1", "g1a", "g1a2", "g1a3", "g1a4", "g1b", "g1c")
NUM_PARAMS = ("0_1b", "0_4b", "1_5b", "2_9b", "7_2b", "13_3b")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _expand_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def _score_root_candidates() -> list[Path]:
    override = _expand_path(os.environ.get("RWKV_SKILLS_SPACE_SCORES_ROOT"))
    if override:
        return [override]

    candidates: list[Path] = []
    seen: set[str] = set()

    def _push(path: Path | None) -> None:
        if path is None:
            return
        candidate = Path(path).expanduser()
        key = str(candidate)
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    run_log_dir = _expand_path(os.environ.get("RUN_LOG_DIR"))
    _push(run_log_dir)

    run_results_dir = _expand_path(os.environ.get("RUN_RESULTS_DIR"))
    if run_results_dir:
        _push(run_results_dir / "scores")

    results_root = _expand_path(os.environ.get("RWKV_SKILLS_RESULTS_ROOT"))
    if results_root:
        _push(results_root / "results" / "scores")
        _push(results_root / "scores")
        _push(results_root)

    bundle = _REPO_ROOT / "rwkv-skills-results"
    _push(bundle / "results" / "scores")
    _push(bundle / "scores")
    _push(bundle)

    _push(SCORES_ROOT)
    return candidates


def _has_score_files(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
        next(path.rglob("*.json"))
        return True
    except StopIteration:
        return False
    except OSError:
        return False


def _resolve_scores_root() -> Path:
    candidates = _score_root_candidates()
    for path in candidates:
        if _has_score_files(path):
            return path
    for path in candidates:
        try:
            if path.exists():
                return path
        except OSError:
            continue
    return candidates[-1] if candidates else SCORES_ROOT


SPACE_SCORES_ROOT = _resolve_scores_root()


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
            # Older dumps sometimes used ``score`` as the primary field.
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
    code_like = job_name in {"code_human_eval", "code_mbpp"} or (task and task.startswith("code")) or (
        "humaneval" in normalized_slug or "mbpp" in normalized_slug
    )
    if not code_like:
        return metrics

    # Pull pass@k style fields from legacy shapes such as ``score`` / nested dicts.
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


def _parse_created_at(raw: Any, path: Path) -> datetime:
    if isinstance(raw, str):
        try:
            cleaned = raw.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(cleaned)
            return parsed.replace(tzinfo=None)
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return datetime.utcnow()


def _infer_domain(dataset_slug: str, *, is_cot: bool, task: str | None) -> str:
    slug = canonical_slug(dataset_slug)
    if slug.startswith("mmlu"):
        return "mmlu系列"
    job = detect_job_from_dataset(slug, is_cot=is_cot)
    if job in {"code_human_eval", "code_mbpp"}:
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


def load_scores(scores_root: str | Path = SPACE_SCORES_ROOT, errors: list[str] | None = None) -> list[ScoreEntry]:
    root = Path(scores_root)
    if not root.exists():
        return []
    entries: list[ScoreEntry] = []
    for path in sorted(root.rglob("*.json")):
        try:
            relative_path = path.relative_to(root)
        except ValueError:
            relative_path = Path(path.name)
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - best effort ingest
            msg = f"无法读取分数文件 {path}: {exc}"
            print(f"[space] {msg}", file=sys.stderr)
            if errors is not None:
                errors.append(msg)
            continue

        dataset = canonical_slug(str(payload.get("dataset", "")).strip())
        model = str(payload.get("model", "")).strip()
        if not dataset or not model:
            continue

        is_cot = bool(payload.get("cot", False))
        task = str(payload.get("task")).strip() if payload.get("task") else None
        metrics = _normalize_metrics(payload, dataset=dataset, is_cot=is_cot, task=task)
        created_at = _parse_created_at(payload.get("created_at"), path)
        raw_samples = payload.get("samples")
        try:
            samples = int(raw_samples) if raw_samples is not None else 0
        except (TypeError, ValueError):
            msg = f"分数文件 {path} 的 samples 字段无法解析: {raw_samples!r}，已按 0 处理"
            print(f"[space] {msg}", file=sys.stderr)
            if errors is not None:
                errors.append(msg)
            samples = 0
        raw_problems = payload.get("problems")
        problems: int | None
        if raw_problems is None:
            problems = None
        else:
            try:
                problems = int(raw_problems)
            except (TypeError, ValueError):
                msg = f"分数文件 {path} 的 problems 字段无法解析: {raw_problems!r}"
                print(f"[space] {msg}", file=sys.stderr)
                if errors is not None:
                    errors.append(msg)
                problems = None
        log_path = str(payload.get("log_path") or "")
        task_details = payload.get("task_details") if isinstance(payload.get("task_details"), dict) else None
        domain = _infer_domain(dataset, is_cot=is_cot, task=task)

        sig = parse_model_signature(model)

        known_keys = {
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
        }
        extra = {k: v for k, v in payload.items() if k not in known_keys}

        entries.append(
            ScoreEntry(
                dataset=dataset,
                model=model,
                metrics=metrics,
                samples=samples,
                problems=problems,
                created_at=created_at,
                log_path=log_path,
                cot=is_cot,
                task=task,
                task_details=task_details,
                path=path,
                relative_path=relative_path,
                domain=domain,
                extra=extra,
                arch_version=sig.arch,
                data_version=sig.data,
                num_params=sig.params,
            )
        )
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
    "SPACE_SCORES_ROOT",
]
