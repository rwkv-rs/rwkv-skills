from __future__ import annotations

"""Queue construction helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Mapping, Sequence, Pattern

from src.eval.benchmark_registry import resolve_benchmark_metadata
from src.eval.param_search.cot_grid import grid_size

from .config import RESULTS_ROOT
from .dataset_utils import canonical_slug, make_dataset_slug, safe_slug
from .jobs import JOB_CATALOGUE
from .models import expand_model_paths, filter_model_names, filter_model_paths
from .naming import build_run_slug
from .state import CompletedKey


@dataclass(slots=True)
class QueueItem:
    job_name: str
    job_id: str
    dataset_slug: str
    model_path: Path | None
    model_slug: str
    model_name: str | None = None
    infer_base_url: str | None = None
    infer_model: str | None = None
    extra_args: tuple[str, ...] = ()
    dataset_path: Path | None = None

    def __post_init__(self) -> None:
        if self.model_name is None:
            if self.infer_model:
                self.model_name = str(self.infer_model)
            elif self.model_path is not None:
                self.model_name = self.model_path.stem
            else:
                raise ValueError("QueueItem requires model_name, infer_model, or model_path")
        if self.infer_base_url and not self.infer_model:
            self.infer_model = self.model_name

    @property
    def is_remote(self) -> bool:
        return bool(self.infer_base_url)


_UNKNOWN_QUESTION_COUNT = 10**9
_WARMUP_BENCHMARK_NAMES = (
    "mmlu",
    "mmlu_pro",
    "gsm8k",
    "math_500",
    "human_eval",
    "mbpp",
    "livecodebench",
    "ifeval",
    "ceval",
)


def _build_warmup_dataset_slugs() -> frozenset[str]:
    slugs: set[str] = set()
    for benchmark_name in _WARMUP_BENCHMARK_NAMES:
        metadata = resolve_benchmark_metadata(benchmark_name)
        slugs.add(canonical_slug(make_dataset_slug(metadata.dataset, metadata.default_split)))
    return frozenset(slugs)


def _build_param_search_target_jobs() -> dict[str, str]:
    targets: dict[str, str] = {}
    for job_name in JOB_CATALOGUE:
        if not job_name.startswith("param_search_") or job_name == "param_search_select":
            continue
        target_job = job_name.removeprefix("param_search_")
        if target_job in JOB_CATALOGUE:
            targets[job_name] = target_job
    return targets


_EARLY_DATASET_SLUGS = _build_warmup_dataset_slugs()
_PARAM_SEARCH_TARGET_JOBS = _build_param_search_target_jobs()
_PARAM_SEARCH_BENCHMARKS = tuple(
    sorted(
        {
            canonical_slug(dataset_slug)
            for job_name in _PARAM_SEARCH_TARGET_JOBS
            for dataset_slug in JOB_CATALOGUE[job_name].dataset_slugs
        }
    )
)
_PARAM_SEARCH_TRIAL_JOBS = frozenset(_PARAM_SEARCH_TARGET_JOBS.values())


def _param_search_required_trial_indices() -> range:
    return range(0, int(grid_size()))


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


def _param_search_done(model_slug: str, dataset_slug: str) -> bool:
    required = _param_search_required_trial_indices()
    present = _param_search_trial_indices_present(model_slug, dataset_slug)
    return all(idx in present for idx in required)


def _param_search_mode_enabled_for_model(
    *,
    model_slug: str,
    completed_set: Collection[CompletedKey],
    failed_set: Collection[CompletedKey],
) -> bool:
    for search_job_name, target_job_name in _PARAM_SEARCH_TARGET_JOBS.items():
        search_job = JOB_CATALOGUE[search_job_name]
        target_job = JOB_CATALOGUE[target_job_name]
        for dataset_slug in search_job.dataset_slugs:
            key = CompletedKey(
                job=target_job_name,
                model_slug=model_slug,
                dataset_slug=canonical_slug(dataset_slug),
                is_cot=target_job.is_cot,
            )
            if key in completed_set or key in failed_set:
                return False
    return bool(_PARAM_SEARCH_TARGET_JOBS)


def _should_replace_with_param_search(
    *,
    job_name: str,
    dataset_slug: str,
    param_search_enabled: bool,
) -> bool:
    return (
        param_search_enabled
        and canonical_slug(dataset_slug) in _PARAM_SEARCH_BENCHMARKS
        and job_name in _PARAM_SEARCH_TRIAL_JOBS
    )


def _param_search_queue_ready(
    *,
    job_name: str,
    model_slug: str,
    dataset_slug: str,
) -> bool:
    canonical_dataset = canonical_slug(dataset_slug)
    if job_name != "param_search_select":
        return not _param_search_done(model_slug, canonical_dataset)
    return all(_param_search_done(model_slug, slug) for slug in _PARAM_SEARCH_BENCHMARKS)


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
    enable_param_search: bool = False,
    model_name_patterns: Sequence[Pattern[str]] | None = None,
    infer_base_url: str | None = None,
    infer_models: Sequence[str] = (),
) -> list[QueueItem]:
    remote_base_url = str(infer_base_url or "").strip() or None
    remote_models = tuple(str(name).strip() for name in infer_models if str(name).strip())
    remote_mode = bool(remote_base_url or remote_models)

    resolved_models: list[tuple[str, Path | None]] = []
    latest_2_9b_models: set[str] = set()
    if remote_mode:
        if not remote_base_url or not remote_models:
            raise ValueError("远端调度必须同时提供 infer_base_url 和 infer_models。")
        filtered_model_names = filter_model_names(remote_models, model_select, min_param_b, max_param_b)
        latest_2_9b_models = set(filter_model_names(remote_models, "latest-data", 2.9, 2.9))
        resolved_models = [(model_name, None) for model_name in filtered_model_names]
    else:
        model_paths = expand_model_paths(model_globs)
        if not model_paths:
            return []
        filtered_models = filter_model_paths(model_paths, model_select, min_param_b, max_param_b)
        latest_2_9b_models = {path.stem for path in filter_model_paths(model_paths, "latest-data", 2.9, 2.9)}
        resolved_models = [(path.stem, path) for path in filtered_models]

    pending: list[QueueItem] = []
    completed_set = set(completed)
    failed_set = set(failed or ())
    skip_datasets = {canonical_slug(slug) for slug in skip_dataset_slugs}
    only_datasets = {canonical_slug(slug) for slug in only_dataset_slugs or []}
    running_set = set(running)
    compiled_patterns = tuple(model_name_patterns or ())

    param_search_enabled: dict[str, bool] = {}
    if enable_param_search and latest_2_9b_models:
        for model_name, _model_path in resolved_models:
            if model_name not in latest_2_9b_models:
                continue
            model_slug = safe_slug(model_name)
            param_search_enabled[model_name] = _param_search_mode_enabled_for_model(
                model_slug=model_slug,
                completed_set=completed_set,
                failed_set=failed_set,
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
            for model_name, model_path in resolved_models:
                model_slug = safe_slug(model_name)
                if compiled_patterns:
                    local_name = model_path.name if model_path is not None else model_name
                    if not any(pattern.search(local_name) or pattern.search(model_name) for pattern in compiled_patterns):
                        continue

                # Latest 2.9b: replace gsm8k + hendrycks_math eval runs with a param-search workflow.
                param_search_on = (
                    enable_param_search
                    and model_name in latest_2_9b_models
                    and param_search_enabled.get(model_name, False)
                )
                if _should_replace_with_param_search(
                    job_name=job_name,
                    dataset_slug=canonical_dataset,
                    param_search_enabled=param_search_on,
                ):
                    continue
                if job_name.startswith("param_search_"):
                    if not enable_param_search:
                        continue
                    if not param_search_on:
                        continue
                    if not _param_search_queue_ready(
                        job_name=job_name,
                        model_slug=model_slug,
                        dataset_slug=canonical_dataset,
                    ):
                        continue

                key = CompletedKey(
                    job=job_name,
                    model_slug=model_slug,
                    dataset_slug=canonical_dataset,
                    is_cot=spec.is_cot,
                )
                if key in completed_set or key in failed_set:
                    continue
                job_id = f"{spec.id_prefix}{build_run_slug(model_name, canonical_dataset, is_cot=spec.is_cot)}"
                if job_id in running_set:
                    continue
                extra_args: tuple[str, ...] = ()
                pending.append(
                    QueueItem(
                        job_name=job_name,
                        job_id=job_id,
                        dataset_slug=canonical_dataset,
                        model_path=model_path,
                        model_slug=model_slug,
                        model_name=model_name,
                        infer_base_url=remote_base_url,
                        infer_model=(model_name if remote_mode else None),
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
