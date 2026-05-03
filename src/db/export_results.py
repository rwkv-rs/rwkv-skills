from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .eval_db_service import EvalDbService


def export_version_results(
    service: "EvalDbService",
    *,
    task_id: str,
) -> None:
    """Results are persisted in the database; skip duplicate JSON exports."""
    return


__all__ = ["export_version_results"]
