from __future__ import annotations

"""GPU batch-size probing inspired by rwkv-eval's batch profiler."""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from .config import DEFAULT_PYTHON, REPO_ROOT
from .jobs import JobSpec


_BATCH_CANDIDATES_ENV = os.environ.get("RUN_BATCH_CANDIDATES") or os.environ.get(
    "RUN_COT_BATCH_CANDIDATES",
    # limit default probing to 2048 to avoid host fallback overhead when VRAM is flaky
    "2048,1024,512,256,128,64,32,16,8,4,2,1",
)
DEFAULT_COT_BATCH_CANDIDATES = tuple(
    int(value.strip()) for value in _BATCH_CANDIDATES_ENV.split(",") if value.strip()
)
DEFAULT_PROBE_MAX_GENERATE = int(os.environ.get("RUN_PROBE_MAX_GENERATE", "16"))


def _extract_cached_batch(record: Any) -> int | None:
    if isinstance(record, dict):
        value = record.get("batch")
    else:
        value = record
    if isinstance(value, bool):
        return int(value) if value else None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def load_batch_cache(cache_path: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, Mapping):
        return {}

    normalised: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for job_name, job_payload in data.items():
        if not isinstance(job_payload, Mapping):
            continue
        job_map: dict[str, dict[str, dict[str, Any]]] = {}
        for model_slug, model_payload in job_payload.items():
            if not isinstance(model_payload, Mapping):
                continue
            model_map: dict[str, dict[str, Any]] = {}
            for gpu_key, record in model_payload.items():
                entry: dict[str, Any] = {}
                batch_value = _extract_cached_batch(record)
                if batch_value is not None:
                    entry["batch"] = batch_value
                if isinstance(record, Mapping):
                    last_error = record.get("last_error")
                    last_probe = record.get("last_probe")
                    if isinstance(last_error, str) and last_error.strip():
                        entry["last_error"] = last_error.strip()
                    if isinstance(last_probe, (int, float)):
                        entry["last_probe"] = float(last_probe)
                if entry:
                    model_map[str(gpu_key)] = entry
            if model_map:
                job_map[model_slug] = model_map
        if job_map:
            normalised[job_name] = job_map
    return normalised


def save_batch_cache(cache_path: Path, data: Mapping[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(slots=True)
class BatchProbeResult:
    batch_size: int | None


@dataclass(slots=True)
class BatchProfiler:
    cache_path: Path
    candidates: tuple[int, ...] = field(default_factory=lambda: DEFAULT_COT_BATCH_CANDIDATES)
    probe_max_generate: int = DEFAULT_PROBE_MAX_GENERATE
    command_prefix: tuple[str, ...] = (DEFAULT_PYTHON, "-m")
    _cache: MutableMapping[str, dict[str, dict[str, dict[str, Any]]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cache = load_batch_cache(self.cache_path)

    def determine_batch_size(
        self,
        *,
        job: JobSpec,
        job_id: str,
        gpu: str,
        dataset_path: Path | None,
        model_path: Path,
        model_slug: str,
        env: dict[str, str],
    ) -> int | None:
        batch_flag = job.batch_flag
        probe_flag = job.probe_flag
        if not batch_flag or not probe_flag:
            return None

        candidates = self.candidates
        if not candidates:
            return None

        max_allowed_batch = max(candidates)

        job_cache = self._cache.setdefault(job.name, {})
        model_cache = job_cache.setdefault(model_slug, {})
        record = model_cache.get(gpu)
        cached_value = _extract_cached_batch(record)
        if cached_value is not None and cached_value > max_allowed_batch:
            cached_value = max_allowed_batch
            model_cache[gpu] = {
                "batch": cached_value,
                "last_probe": time.time(),
            }
            record = model_cache[gpu]
            save_batch_cache(self.cache_path, self._cache)
        record_is_dict = isinstance(record, Mapping)
        last_probe = record.get("last_probe") if record_is_dict else None
        last_error = record.get("last_error") if record_is_dict else None
        cache_is_trustworthy = (
            cached_value is not None and cached_value > 0 and isinstance(last_probe, (int, float)) and (not last_error)
        )
        if cache_is_trustworthy:
            print(f"🔁 Using cached batch size {cached_value} for {job_id} on cuda:{gpu}.")
            return cached_value

        if job.probe_dataset_required and not dataset_path:
            fallback = min(candidates)
            print(
                f"⚠️  Dataset path is required but missing for {job_id}; falling back to batch size {fallback}."
            )
            model_cache[gpu] = {
                "batch": fallback,
                "last_error": "dataset path unavailable",
                "last_probe": time.time(),
            }
            save_batch_cache(self.cache_path, self._cache)
            return fallback

        probe_env = env.copy()
        probe_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        probe_env["CUDA_VISIBLE_DEVICES"] = gpu

        joined = ",".join(str(value) for value in candidates)
        print(f"🔍 Probing batch size for {job_id} on cuda:{gpu} (candidates: {joined})")

        for candidate in candidates:
            command = list(self.command_prefix) + [job.module]
            command.extend(
                [
                    "--model-path",
                    str(model_path),
                    "--device",
                    "cuda:0",
                    batch_flag,
                    str(candidate),
                ]
            )
            if probe_flag:
                command.append(probe_flag)
                if probe_flag in {"--max-samples", "--max-tokens"}:
                    command.append("1")
            if dataset_path is not None:
                command.extend(["--dataset", str(dataset_path)])
            if job.probe_max_generate_flag:
                command.extend([job.probe_max_generate_flag, str(self.probe_max_generate)])
            if job.probe_extra_args:
                command.extend(job.probe_extra_args)
            if job.extra_args:
                command.extend(job.extra_args)

            start_time = time.time()
            proc = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                env=probe_env,
                capture_output=True,
                text=True,
            )
            elapsed = max(0.0, time.time() - start_time)

            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            combined_lower = f"{stdout}\n{stderr}".lower()

            if proc.returncode == 0:
                model_cache[gpu] = {
                    "batch": candidate,
                    "last_probe": time.time(),
                }
                save_batch_cache(self.cache_path, self._cache)
                print(f"✅ Batch size {candidate} works for {job_id} on cuda:{gpu} (probe {elapsed:.1f}s).")
                return candidate

            message = stderr or stdout or f"exit code {proc.returncode}"
            if "out of memory" in combined_lower or "cuda oom" in combined_lower:
                print(
                    f"⚠️  Batch size {candidate} hit OOM for {job_id} on cuda:{gpu}; trying smaller candidate."
                )
                model_cache[gpu] = {
                    "last_error": f"oom at {candidate}: {message[:200]}",
                    "last_probe": time.time(),
                }
                save_batch_cache(self.cache_path, self._cache)
                continue

            print(f"❌ Batch size {candidate} probe failed for {job_id} on cuda:{gpu} (exit {proc.returncode}).")
            if stdout:
                print("   probe stdout:")
                for line in stdout.splitlines():
                    print(f"     {line}")
            if stderr:
                print("   probe stderr:")
                for line in stderr.splitlines():
                    print(f"     {line}")
            raise RuntimeError(f"probe failed for {job_id} on cuda:{gpu}: {message}")


__all__ = ["BatchProfiler", "load_batch_cache", "save_batch_cache"]
