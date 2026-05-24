from __future__ import annotations

import importlib
import sys

from src.eval.scheduler import state


def test_state_module_does_not_eagerly_import_db_service() -> None:
    module_name = "src.eval.scheduler.state"
    db_module_name = "src.db.eval_db_service"
    db_init_module_name = "src.db.database"

    sys.modules.pop(module_name, None)
    sys.modules.pop(db_module_name, None)
    sys.modules.pop(db_init_module_name, None)

    importlib.import_module(module_name)

    assert db_module_name not in sys.modules
    assert db_init_module_name not in sys.modules


def test_newer_completed_record_prefers_latest_task_id_for_same_job() -> None:
    key = state.CompletedKey(
        job="function_toolalpaca",
        model_slug="rwkv7_g1f_7_2b_20260414_ctx8192",
        dataset_slug="toolalpaca_eval_real_test",
        is_cot=True,
    )
    older = state.CompletedRecord(
        job_id="function_toolalpaca__toolalpaca_eval_real_test_cot_rwkv7_g1f_7_2b_20260414_ctx8192",
        key=key,
        model_name="rwkv7-g1f-7.2b-20260414-ctx8192",
        version_id="274",
    )
    newer = state.CompletedRecord(
        job_id=older.job_id,
        key=key,
        model_name=older.model_name,
        version_id="832",
    )

    assert state._newer_completed_record(older, newer) is newer
    assert state._newer_completed_record(newer, older) is newer
