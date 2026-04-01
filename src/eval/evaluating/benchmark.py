from __future__ import annotations

"""Benchmark selection helpers aligned with rwkv-rs' evaluating layer."""

from dataclasses import dataclass
from typing import Sequence

from src.eval.benchmark_registry import ALL_BENCHMARKS, BENCHMARKS_BY_FIELD, BenchmarkField, BenchmarkMetadata
from src.eval.scheduler.dataset_utils import canonical_slug, make_dataset_slug, split_benchmark_and_split


_BENCHMARKS_BY_NAME: dict[str, BenchmarkMetadata] = {item.name: item for item in ALL_BENCHMARKS}


@dataclass(frozen=True, slots=True)
class SelectedBenchmark:
    metadata: BenchmarkMetadata
    dataset_slug: str

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def field(self) -> BenchmarkField:
        return self.metadata.field


def resolve_registered_benchmark_name(raw_name: str) -> str:
    slug = canonical_slug(raw_name)
    if slug in _BENCHMARKS_BY_NAME:
        return slug

    benchmark_name, _ = split_benchmark_and_split(slug)
    if benchmark_name in _BENCHMARKS_BY_NAME:
        return benchmark_name

    raise ValueError(f"unknown benchmark name: {raw_name}")


def benchmark_dataset_slug(metadata: BenchmarkMetadata) -> str:
    return make_dataset_slug(metadata.name, metadata.default_split)


def collect_benchmarks(
    *,
    fields: Sequence[BenchmarkField] | None = None,
    extra_benchmark_names: Sequence[str] | None = None,
) -> tuple[SelectedBenchmark, ...]:
    selected: dict[str, SelectedBenchmark] = {}

    for field in fields or ():
        for metadata in BENCHMARKS_BY_FIELD.get(field, ()):
            selected.setdefault(
                metadata.name,
                SelectedBenchmark(metadata=metadata, dataset_slug=benchmark_dataset_slug(metadata)),
            )

    for raw_name in extra_benchmark_names or ():
        resolved_name = resolve_registered_benchmark_name(raw_name)
        metadata = _BENCHMARKS_BY_NAME[resolved_name]
        selected[metadata.name] = SelectedBenchmark(
            metadata=metadata,
            dataset_slug=benchmark_dataset_slug(metadata),
        )

    if not selected and not fields and not extra_benchmark_names:
        for metadata in ALL_BENCHMARKS:
            selected.setdefault(
                metadata.name,
                SelectedBenchmark(metadata=metadata, dataset_slug=benchmark_dataset_slug(metadata)),
            )

    return tuple(sorted(selected.values(), key=lambda item: (item.field.value, item.name)))


def collect_benchmark_dataset_slugs(
    *,
    fields: Sequence[BenchmarkField] | None = None,
    extra_benchmark_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    return tuple(item.dataset_slug for item in collect_benchmarks(fields=fields, extra_benchmark_names=extra_benchmark_names))


__all__ = [
    "SelectedBenchmark",
    "benchmark_dataset_slug",
    "collect_benchmark_dataset_slugs",
    "collect_benchmarks",
    "resolve_registered_benchmark_name",
]
