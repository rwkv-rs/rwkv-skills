from __future__ import annotations

import subprocess
from pathlib import Path

from src.db import eval_db_service as service_module
from src.db.eval_db_service import EvalDbService


class _FakeRepo:
    def get_benchmark_id(self, **kwargs: object) -> int:  # noqa: ARG002
        return 1

    def get_benchmark_num_samples(self, **kwargs: object) -> int:  # noqa: ARG002
        return 1

    def get_model_id(self, **kwargs: object) -> int:  # noqa: ARG002
        return 2

    def find_tasks_by_identity(self, **kwargs: object) -> list[dict[str, object]]:  # noqa: ARG002
        return [
            {"task_id": 10, "status": "Completed"},
            {"task_id": 11, "status": "Failed"},
        ]

    def task_has_score(self, *, task_id: int) -> bool:
        return False

    def fetch_completion_keys(self, **kwargs: object) -> list[tuple[int, int, int]]:  # noqa: ARG002
        return [(0, 0, 0)]


def test_scoreless_completed_task_does_not_block_resume(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_get_cached_git_sha", lambda: "test-sha")
    service = EvalDbService.__new__(EvalDbService)
    service._repo = _FakeRepo()

    ctx = service.get_resume_context(
        dataset="aime25_test",
        model="rwkv7-g1g-13.3b-20260523-ctx8192",
        is_param_search=False,
        job_name="free_response",
        sampling_config={},
    )

    assert ctx.completed_task_ids == ()
    assert ctx.resumable_task_ids == (11,)
    assert ctx.task_id == 11
    assert ctx.can_resume is True


def test_git_sha_falls_back_to_stable_nogit_fingerprint(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "_GIT_SHA_CACHE", None)
    monkeypatch.setattr(service_module, "REPO_ROOT", Path("/home/rwkv/chase/rwkv-skills"))

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:  # noqa: ARG001
        return subprocess.CompletedProcess(args=["git"], returncode=128, stdout="", stderr="fatal")

    monkeypatch.setattr(service_module.subprocess, "run", _fake_run)

    assert service_module._get_cached_git_sha() == "nogit-ffee99691273"
