from __future__ import annotations

from pathlib import Path

from src.eval.tasks.agent_bench import deps as agent_bench_deps
from src.eval.tasks.agent_bench import tasks as agent_bench_tasks
from src.eval.tasks.function_calling import bfcl_ast, bfcl_exec


def test_moved_eval_task_helpers_resolve_repo_root(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bfcl_root = repo_root / "references" / "gorilla" / "berkeley-function-call-leaderboard"

    assert agent_bench_deps._REPO_ROOT == repo_root
    assert agent_bench_tasks.REPO_ROOT == repo_root
    assert bfcl_ast._repo_default_official_root() == bfcl_root

    monkeypatch.setattr(Path, "is_dir", lambda _self: False)
    assert bfcl_exec._default_bfcl_official_root() == bfcl_root
