from __future__ import annotations

"""Queue construction helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Sequence

from .dataset_utils import canonical_slug, safe_slug
from .jobs import JOB_CATALOGUE, JobSpec
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
    dataset_path: Path | None = None


def build_queue(
    *,
    model_globs: Sequence[str],
    job_order: Sequence[str],
    completed: Collection[CompletedKey],
    failed: Collection[CompletedKey] | None = None,
    running: Collection[str],
    skip_dataset_slugs: Collection[str],
    model_select: str,
    min_param_b: float | None,
    max_param_b: float | None,
) -> list[QueueItem]:
    model_paths = expand_model_paths(model_globs)
    if not model_paths:
        return []
    filtered_models = filter_model_paths(model_paths, model_select, min_param_b, max_param_b)

    pending: list[QueueItem] = []
    completed_set = set(completed)
    failed_set = set(failed or ())
    skip_datasets = {canonical_slug(slug) for slug in skip_dataset_slugs}
    running_set = set(running)

    for job_name in job_order:
        spec = JOB_CATALOGUE.get(job_name)
        if spec is None:
            continue
        for dataset_slug in spec.dataset_slugs:
            canonical_dataset = canonical_slug(dataset_slug)
            if canonical_dataset in skip_datasets:
                continue
            for model_path in filtered_models:
                model_slug = safe_slug(model_path.stem)
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
                pending.append(
                    QueueItem(
                        job_name=job_name,
                        job_id=job_id,
                        dataset_slug=canonical_dataset,
                        model_path=model_path,
                        model_slug=model_slug,
                    )
                )
    return pending


__all__ = ["QueueItem", "build_queue"]
