from __future__ import annotations

import subprocess
from pathlib import Path

from src.eval.scheduler.jobs import JobSpec
from src.eval.scheduler.profiler import BatchProfiler


def test_batch_profiler_strips_session_env_for_probe(monkeypatch, tmp_path) -> None:
    captured_env: dict[str, str] = {}

    def fake_run(*_args, **kwargs):
        captured_env.update(kwargs["env"])
        return subprocess.CompletedProcess(args=kwargs.get("args", []), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.eval.scheduler.profiler.subprocess.run", fake_run)
    profiler = BatchProfiler(cache_path=tmp_path / "batch_cache.json", candidates=(1,))
    job = JobSpec(
        name="free_response_judge",
        module="src.bin.eval_free_response_judge",
        dataset_slugs=("math_500_test",),
        is_cot=True,
        domain="free_response",
        batch_flag="--batch-size",
        probe_flag="--probe-only",
    )

    batch = profiler.determine_batch_size(
        job=job,
        job_id="job-1",
        gpu="1",
        dataset_path=Path("data/math_500/test.jsonl"),
        model_path=Path("rwkv7-g1g-1.5b-20260526-ctx8192.pth"),
        model_slug="rwkv7_g1g_1_5b_20260526_ctx8192",
        env={
            "RWKV_SESSION_ID": "session-1",
            "RWKV_SESSION_TASK_ID": "42",
            "RWKV_SKILLS_TASK_ID": "42",
            "RWKV_SKILLS_VERSION_ID": "42",
        },
        dataset_questions=1,
    )

    assert batch == 1
    assert "RWKV_SESSION_ID" not in captured_env
    assert "RWKV_SESSION_TASK_ID" not in captured_env
    assert "RWKV_SKILLS_TASK_ID" not in captured_env
    assert "RWKV_SKILLS_VERSION_ID" not in captured_env
