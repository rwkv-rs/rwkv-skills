from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.dashboard.core.boards import BOARD_NAIVE
from src.dashboard.core.constants import AUTO_MODEL_LABEL
from src.dashboard.core.data import ScoreEntry
from src.dashboard.core.domains import DOMAIN_MMLU
from src.dashboard.core.selection import _prepare_selection
from src.dashboard.web.serialize import serialize_leaderboard


def _entry(
    *,
    task_id: int,
    dataset: str,
    model: str,
    task: str,
    created_at: datetime,
    prompt_profile: str | None,
) -> ScoreEntry:
    sampling_config: dict[str, object] = {}
    if prompt_profile is not None:
        sampling_config["prompt_profile"] = prompt_profile
    return ScoreEntry(
        task_id=task_id,
        dataset=dataset,
        model=model,
        metrics={"accuracy": 0.8},
        samples=100,
        problems=100,
        created_at=created_at,
        log_path="",
        cot=False,
        task=task,
        task_details=None,
        path=Path("<test>"),
        relative_path=Path("<test>"),
        domain=DOMAIN_MMLU,
        extra={"sampling_config": sampling_config, "cot_mode": "no_cot"},
        arch_version="RWKV7",
        data_version="G1G" if "g1g" in model else "G1F",
        num_params="1_5b",
    )


def _task_ids_from_rows(rows: list[dict[str, object]]) -> set[int]:
    task_ids: set[int] = set()
    for row in rows:
        for cell in row["cells"]:  # type: ignore[index]
            meta = cell.get("meta") if isinstance(cell, dict) else None
            if isinstance(meta, dict) and isinstance(meta.get("task_id"), int):
                task_ids.add(meta["task_id"])
            latest_meta = cell.get("latest_meta") if isinstance(cell, dict) else None
            if isinstance(latest_meta, dict) and isinstance(latest_meta.get("task_id"), int):
                task_ids.add(latest_meta["task_id"])
    return task_ids


def test_normal_and_naive_boards_resolve_independent_lineages() -> None:
    normal_task_id = 101
    naive_task_id = 202
    entries = [
        _entry(
            task_id=normal_task_id,
            dataset="mmlu_pro",
            model="rwkv7-g1f-1.5b-20260526-ctx8192",
            task="multi_choice_plain",
            created_at=datetime(2026, 5, 26),
            prompt_profile=None,
        ),
        _entry(
            task_id=naive_task_id,
            dataset="gpqa_main",
            model="rwkv7-g1g-1.5b-20260527-ctx8192",
            task="multi_choice_plain_naive",
            created_at=datetime(2026, 5, 27),
            prompt_profile="naive",
        ),
    ]
    selection = _prepare_selection(entries, AUTO_MODEL_LABEL)

    payload = serialize_leaderboard(
        selection,
        all_entries=entries,
        view_mode="benchmark_detail_latest",
    )

    knowledge = next(domain for domain in payload["domains"] if domain["key"] == "knowledge")
    naive_board = payload["naive_board"]

    assert payload["param_columns"][0]["latest_model"] == "rwkv7-g1f-1.5b-20260526-ctx8192"
    assert knowledge["param_columns"][0]["latest_model"] == "rwkv7-g1f-1.5b-20260526-ctx8192"
    assert naive_board["param_columns"][0]["latest_model"] == "rwkv7-g1g-1.5b-20260527-ctx8192"

    assert _task_ids_from_rows(knowledge["rows"]) == {normal_task_id}
    assert _task_ids_from_rows(naive_board["rows"]) == {naive_task_id}
    assert naive_board["key"] == BOARD_NAIVE

    assert {meta["task_id"] for meta in payload["interaction_meta"].values()} == {
        normal_task_id,
        naive_task_id,
    }


def test_normal_and_naive_cells_do_not_overwrite_click_metadata() -> None:
    normal_task_id = 101
    naive_task_id = 202
    model = "rwkv7-g1g-1.5b-20260527-ctx8192"
    entries = [
        _entry(
            task_id=normal_task_id,
            dataset="mmlu_pro",
            model=model,
            task="multi_choice_plain",
            created_at=datetime(2026, 5, 27),
            prompt_profile=None,
        ),
        _entry(
            task_id=naive_task_id,
            dataset="mmlu_pro",
            model=model,
            task="multi_choice_plain_naive",
            created_at=datetime(2026, 5, 27),
            prompt_profile="naive",
        ),
    ]
    selection = _prepare_selection(entries, AUTO_MODEL_LABEL)

    payload = serialize_leaderboard(
        selection,
        all_entries=entries,
        view_mode="benchmark_detail_latest",
    )

    assert len(payload["interaction_meta"]) == 2
    assert {meta["task_id"] for meta in payload["interaction_meta"].values()} == {
        normal_task_id,
        naive_task_id,
    }
