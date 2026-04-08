from __future__ import annotations

import importlib
import sys


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
