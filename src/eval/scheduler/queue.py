from __future__ import annotations

"""Queue construction helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Mapping, Sequence, Pattern

from src.eval.param_search.cot_grid import grid_size_by_mode

from .config import RESULTS_ROOT
from .dataset_utils import canonical_slug, safe_slug
from .jobs import JOB_CATALOGUE, JobSpec, detect_job_from_dataset
from .models import expand_model_paths, filter_model_paths
from .naming import build_run_slug
from .state import CompletedKey, RunningEntry


@dataclass(slots=True)
class QueueItem:
    job_name: str
    job_id: str
    dataset_slug: str
    model_path: Path
    model_slug: str
    extra_args: tuple[str, ...] = ()
    dataset_path: Path | None = None


_UNKNOWN_QUESTION_COUNT = 10**9
_EARLY_DATASET_SLUGS = frozenset(
    canonical_slug(slug)
    for slug in (
        "mmlu_test",
        "mmlu_pro_test",
        "gsm8k_test",
        "math_500_test",
        "human_eval_test",
        "mbpp_test",
        "ifeval_test",
        "ceval_test",
    )
)

_PARAM_SEARCH_BENCHMARKS = tuple(canonical_slug(slug) for slug in ("gsm8k_test", "math_500_test"))


def _normalize_param_search_scan_mode(mode: str | None) -> str:
    normalized = (mode or "both").strip().lower()
    if normalized not in {"both", "normal", "simple"}:
        raise ValueError(f"未知的 param-search scan mode: {mode!r} (expected: both/normal/simple)")
    return normalized


def _param_search_required_trial_indices(scan_mode: str) -> range:
    sizes = grid_size_by_mode()
    normal = int(sizes["normal"])
    simple = int(sizes["simple"])
    total = normal + simple
    mode = _normalize_param_search_scan_mode(scan_mode)
    if mode == "normal":
        return range(0, normal)
    if mode == "simple":
        return range(normal, total)
    return range(0, total)


def _param_search_score_dir(model_slug: str, dataset_slug: str) -> Path:
    return RESULTS_ROOT / "param_search" / "scores" / safe_slug(model_slug) / canonical_slug(dataset_slug)


def _param_search_trial_indices_present(model_slug: str, dataset_slug: str) -> set[int]:
    directory = _param_search_score_dir(model_slug, dataset_slug)
    if not directory.exists():
        return set()
    present: set[int] = set()
    for path in directory.glob("trial_*.json"):
        stem = path.stem
        if not stem.startswith("trial_"):
            continue
        suffix = stem.removeprefix("trial_")
        try:
            present.add(int(suffix))
        except ValueError:
            continue
    return present


def _param_search_done(model_slug: str, dataset_slug: str, *, scan_mode: str) -> bool:
    required = _param_search_required_trial_indices(scan_mode)
    present = _param_search_trial_indices_present(model_slug, dataset_slug)
    return all(idx in present for idx in required)


def build_queue(
    *,
    model_globs: Sequence[str],
    job_order: Sequence[str],
    completed: Collection[CompletedKey],
    failed: Collection[CompletedKey] | None = None,
    running: Collection[str],
    skip_dataset_slugs: Collection[str],
    only_dataset_slugs: Collection[str] | None,
    model_select: str,
    min_param_b: float | None,
    max_param_b: float | None,
    param_search_scan_mode: str = "both",
    model_name_patterns: Sequence[Pattern[str]] | None = None,
) -> list[QueueItem]:
    model_paths = expand_model_paths(model_globs)
    if not model_paths:
        return []
    filtered_models = filter_model_paths(model_paths, model_select, min_param_b, max_param_b)
    latest_2_9b_models = set(filter_model_paths(model_paths, "latest-data", 2.9, 2.9))

    pending: list[QueueItem] = []
    completed_set = set(completed)
    failed_set = set(failed or ())
    skip_datasets = {canonical_slug(slug) for slug in skip_dataset_slugs}
    only_datasets = {canonical_slug(slug) for slug in only_dataset_slugs or []}
    running_set = set(running)
    compiled_patterns = tuple(model_name_patterns or ())
    scan_mode = _normalize_param_search_scan_mode(param_search_scan_mode)

    param_search_enabled: dict[Path, bool] = {}
    if latest_2_9b_models:
        gsm_slug, math_slug = _PARAM_SEARCH_BENCHMARKS
        for model_path in filtered_models:
            if model_path not in latest_2_9b_models:
                continue
            model_slug = safe_slug(model_path.stem)
            gsm_job = detect_job_from_dataset(gsm_slug, True) or "free_response_judge"
            math_job = detect_job_from_dataset(math_slug, True) or "free_response"
            gsm_key = CompletedKey(job=gsm_job, model_slug=model_slug, dataset_slug=gsm_slug, is_cot=True)
            math_key = CompletedKey(job=math_job, model_slug=model_slug, dataset_slug=math_slug, is_cot=True)
            param_search_enabled[model_path] = (
                gsm_key not in completed_set
                and gsm_key not in failed_set
                and math_key not in completed_set
                and math_key not in failed_set
            )

    for job_name in job_order:
        spec = JOB_CATALOGUE.get(job_name)
        if spec is None:
            continue
        for dataset_slug in spec.dataset_slugs:
            canonical_dataset = canonical_slug(dataset_slug)
            if only_datasets and canonical_dataset not in only_datasets:
                continue
            if canonical_dataset in skip_datasets:
                continue
            for model_path in filtered_models:
                model_slug = safe_slug(model_path.stem)
                if compiled_patterns:
                    name = model_path.name
                    stem = model_path.stem
                    if not any(pattern.search(name) or pattern.search(stem) for pattern in compiled_patterns):
                        continue

                # Latest 2.9b: replace gsm8k + hendrycks_math eval runs with a param-search workflow.
                param_search_on = model_path in latest_2_9b_models and param_search_enabled.get(model_path, False)
                if (
                    param_search_on
                    and canonical_dataset in _PARAM_SEARCH_BENCHMARKS
                    and job_name in {"free_response", "free_response_judge"}
                ):
                    continue
                if job_name.startswith("param_search_"):
                    if not param_search_on:
                        continue
                    if job_name != "param_search_select":
                        if _param_search_done(model_slug, canonical_dataset, scan_mode=scan_mode):
                            continue
                    else:
                        if not all(_param_search_done(model_slug, slug, scan_mode=scan_mode) for slug in _PARAM_SEARCH_BENCHMARKS):
                            continue

                key = CompletedKey(
                    job=job_name,
                    model_slug=model_slug,
                    dataset_slug=canonical_dataset,
                    is_cot=spec.is_cot,
                )
                if key in completed_set or key in failed_set:
                    continue
                job_id = f"{spec.id_prefix}{build_run_slug(model_path, canonical_dataset, is_cot=spec.is_cot)}"
                if job_id in running_set:
                    continue
                extra_args: tuple[str, ...] = ()
                if job_name in {"param_search_free_response", "param_search_free_response_judge"} and scan_mode != "both":
                    extra_args = ("--scan-mode", scan_mode)
                pending.append(
                    QueueItem(
                        job_name=job_name,
                        job_id=job_id,
                        dataset_slug=canonical_dataset,
                        model_path=model_path,
                        model_slug=model_slug,
                        extra_args=extra_args,
                    )
                )
    return pending


def sort_queue_items(
    queue: list[QueueItem],
    *,
    question_counts: Mapping[str, int] | None = None,
    job_priority: Mapping[str, int] | None = None,
) -> list[QueueItem]:
    """Sort pending jobs so smaller datasets & nocot runs go first."""

    if not queue:
        return queue
    counts = question_counts or {}
    priorities = job_priority or {}

    def _key(item: QueueItem) -> tuple[int, int, int, int, str, str]:
        job = JOB_CATALOGUE.get(item.job_name)
        is_cot = job.is_cot if job else False
        questions = counts.get(item.dataset_slug, _UNKNOWN_QUESTION_COUNT)
        job_rank = priorities.get(item.job_name, len(priorities))
        # Prioritise specific datasets: non-CoT runs first, then CoT, then everything else.
        if item.dataset_slug in _EARLY_DATASET_SLUGS:
            dataset_rank = 0 if not is_cot else 1
        else:
            dataset_rank = 2
        nocot_rank = 0 if not is_cot else 1
        return (job_rank, dataset_rank, questions, nocot_rank, item.dataset_slug, item.job_id)

    queue.sort(key=_key)
    return queue


__all__ = ["QueueItem", "build_queue", "sort_queue_items"]
