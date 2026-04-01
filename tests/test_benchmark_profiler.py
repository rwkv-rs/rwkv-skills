from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.eval.scheduler.jobs import JobSpec
from src.eval.scheduler.profiler import BatchProfiler, load_batch_cache
import src.eval.scheduler.profiler as profiler_module


def _job_spec() -> JobSpec:
    return JobSpec(
        name="free_response",
        module="src.bin.eval_free_response",
        dataset_slugs=("math_500_test",),
        is_cot=True,
        domain="free_response",
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=False,
    )


def test_select_probe_candidates_prefers_power_of_two_before_partial_batch(tmp_path: Path) -> None:
    profiler = BatchProfiler(
        tmp_path / "batch_cache.json",
        candidates=(4096, 2048, 1024, 512, 256, 128, 64),
    )

    assert profiler._select_probe_candidates(profiler.candidates, question_count=1319) == (
        1024,
        1319,
        512,
        256,
        128,
        64,
    )
    assert profiler._select_probe_candidates(profiler.candidates, question_count=1024) == (
        1024,
        512,
        256,
        128,
        64,
    )


def test_determine_batch_size_caches_first_success_after_oom(monkeypatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "batch_cache.json"
    profiler = BatchProfiler(cache_path, candidates=(8, 4, 2), command_prefix=("python3", "-m"))
    calls: list[int] = []

    def fake_run(command, cwd, env, capture_output, text):
        batch = int(command[command.index("--batch-size") + 1])
        calls.append(batch)
        if batch == 8:
            return SimpleNamespace(returncode=1, stdout="", stderr="CUDA out of memory")
        if batch == 4:
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        raise AssertionError(f"unexpected probe candidate: {batch}")

    monkeypatch.setattr(profiler_module.subprocess, "run", fake_run)

    result = profiler.determine_batch_size(
        job=_job_spec(),
        job_id="free_response__math_500_test",
        gpu="0",
        dataset_path=None,
        model_path=tmp_path / "weights" / "rwkv7-g1a-2.9b.pth",
        model_slug="rwkv7_g1a_2_9b",
        env={},
        dataset_questions=None,
    )

    assert result == 4
    assert calls == [8, 4]
    cache = load_batch_cache(cache_path)
    assert cache["free_response"]["rwkv7_g1a_2_9b"]["0"]["batch"] == 4
    assert "last_error" not in cache["free_response"]["rwkv7_g1a_2_9b"]["0"]

    monkeypatch.setattr(
        profiler_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cached probe should skip subprocess")),
    )

    cached = profiler.determine_batch_size(
        job=_job_spec(),
        job_id="free_response__math_500_test",
        gpu="0",
        dataset_path=None,
        model_path=tmp_path / "weights" / "rwkv7-g1a-2.9b.pth",
        model_slug="rwkv7_g1a_2_9b",
        env={},
        dataset_questions=None,
    )

    assert cached == 4
