"""Dashboard DB adapter.

This mirrors the scoreboard-store boundary used by the helicopter dashboard:
routes and API services depend on this small adapter instead of constructing
the project DB service directly.
"""

from __future__ import annotations

from typing import Any

from src.db.eval_db_service import EvalDbService

from ..core.score_index import write_score_index_entries


class DashboardStore:
    def __init__(self, service: EvalDbService | None = None) -> None:
        self._service = service

    @property
    def service(self) -> EvalDbService:
        if self._service is None:
            self._service = EvalDbService()
        return self._service

    def rebuild_score_index_from_db(self, *, include_param_search: bool = False) -> int:
        rows = self.service.list_latest_scores_for_space(
            include_param_search=include_param_search,
        )
        _, count = write_score_index_entries(rows)
        return count

    def list_eval_records_for_space(
        self,
        *,
        task_id: str,
        only_wrong: bool,
        limit: int | None = None,
        offset: int = 0,
        include_context: bool = True,
        include_preview: bool = False,
    ) -> list[dict[str, Any]]:
        return self.service.list_eval_records_for_space(
            task_id=task_id,
            only_wrong=only_wrong,
            limit=limit,
            offset=offset,
            include_context=include_context,
            include_preview=include_preview,
        )

    def get_eval_context_for_space(
        self,
        *,
        task_id: str,
        sample_index: int,
        repeat_index: int,
        pass_index: int = 0,
    ) -> Any | None:
        return self.service.get_eval_context_for_space(
            task_id=task_id,
            sample_index=sample_index,
            repeat_index=repeat_index,
            pass_index=pass_index,
        )

    def get_task_bundle(self, *, task_id: str) -> dict[str, Any]:
        return self.service.get_task_bundle(task_id=task_id)

    def list_score_history(self, *, model: str, dataset: str) -> list[dict[str, Any]]:
        return self.service.list_score_history(model=model, dataset=dataset)

    def list_score_history_pairs(self) -> list[dict[str, Any]]:
        return self.service.list_score_history_pairs()

    def list_eval_answers_for_tasks(self, *, task_ids: list[int]) -> list[dict[str, Any]]:
        return self.service.list_eval_answers_for_tasks(task_ids=task_ids)

    def get_score_history_detail(self, *, task_id: str) -> dict[str, Any] | None:
        return self.service.get_score_history_detail(task_id=task_id)


__all__ = ["DashboardStore"]
