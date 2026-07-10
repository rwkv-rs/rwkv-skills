"""FastAPI JSON API for the RWKV Skills evaluation dashboard.

Data sources (no Rust backend, no raw SQL in this layer):
  * leaderboard scores  ->  ``results/space/score_index.jsonl`` via ``load_scores``
  * eval records/context ->  the project's ``EvalDbService`` (Postgres)

The Next.js frontend runs from ``client/`` and reaches these routes through
Next rewrites. This process serves the JSON API only.
"""

from __future__ import annotations

from dataclasses import replace
import threading
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .admin_api import register_admin_routes
from .charts_json import serialize_charts
from ..core.constants import (
    AUTO_MODEL_LABEL,
    DEFAULT_TABLE_VIEW,
    DOMAIN_GROUPS,
    EVAL_PAGE_SIZE,
    TABLE_VIEW_LABELS,
    _normalize_table_view,
)
from ..core.boards import BOARD_NORMAL, filter_entries_by_board
from ..core.data import ScoreEntry, list_models, load_scores
from .eval_service import (
    load_eval_context,
    load_eval_records,
    rebuild_score_index_from_db,
)
from .score_history import score_history, score_history_detail, score_history_options
from .screenshot import capture_page
from .store import DashboardStore
from ..core.selection import _compute_choices, _prepare_selection
from .serialize import serialize_leaderboard, serialize_selection

class _ScoreCache:
    """Process-wide cache of parsed score entries, refreshable on demand.

    Before parsing, the score index is rebuilt from the latest DB scores so the
    leaderboard always reflects newly-completed evals. The rebuild is best-effort:
    if the DB is unreachable we fall back to the existing on-disk index and surface
    the error. The downstream read/selection logic is unchanged.
    """

    def __init__(self, store: DashboardStore) -> None:
        self._store = store
        self._lock = threading.Lock()
        self._entries: list[ScoreEntry] | None = None
        self._errors: list[str] = []

    def _rebuild_from_db(self, errors: list[str]) -> None:
        try:
            rebuild_score_index_from_db(store=self._store)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"从数据库重建排行榜索引失败，回退到现有索引：{exc}")

    def _load_locked(self) -> tuple[list[ScoreEntry], list[str]]:
        errors: list[str] = []
        self._rebuild_from_db(errors)
        self._entries = load_scores(errors)
        self._errors = errors
        return self._entries, self._errors

    def get(self) -> tuple[list[ScoreEntry], list[str]]:
        with self._lock:
            if self._entries is None:
                return self._load_locked()
            return self._entries, self._errors

    def refresh(self) -> tuple[list[ScoreEntry], list[str]]:
        with self._lock:
            return self._load_locked()

def _domain_groups_payload() -> list[dict[str, Any]]:
    groups = [
        {"key": group["key"], "label": group["label"], "title": group["title"]}
        for group in DOMAIN_GROUPS
    ]
    # 朴素榜 (naive) is a flat tab alongside the domain tabs, not domain-split.
    groups.append({"key": "naive", "label": "朴素榜", "title": "朴素榜"})
    return groups


def create_app(store: DashboardStore | None = None) -> FastAPI:
    resolved_store = store or DashboardStore()
    cache = _ScoreCache(resolved_store)
    app = FastAPI(title="RWKV Skills Dashboard", version="2.0")

    # Permissive CORS so a separate Next.js dev server can call the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        entries, errors = cache.get()
        return {
            "auto_label": AUTO_MODEL_LABEL,
            "default_view": DEFAULT_TABLE_VIEW,
            "table_views": [{"key": k, "label": v} for k, v in TABLE_VIEW_LABELS.items()],
            "domain_groups": _domain_groups_payload(),
            "models": list_models(entries),
            "model_choices": _compute_choices(entries),
            "entry_count": len(entries),
            "errors": errors,
        }

    @app.post("/api/refresh")
    def refresh() -> dict[str, Any]:
        entries, errors = cache.refresh()
        return {"entry_count": len(entries), "errors": errors}

    @app.post("/api/capture-page")
    def capture_page_endpoint(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        try:
            payload = payload or {}
            return capture_page(
                url=payload.get("url"),
                width=payload.get("width"),
                height=payload.get("height"),
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"保存前端长截图失败：{exc}") from exc

    @app.get("/api/leaderboard")
    def leaderboard(
        model: str | None = Query(default=None),
        view: str = Query(default=DEFAULT_TABLE_VIEW),
    ) -> dict[str, Any]:
        entries, errors = cache.get()
        selection = _prepare_selection(entries, model or AUTO_MODEL_LABEL)
        payload = serialize_leaderboard(selection, all_entries=entries, view_mode=_normalize_table_view(view))
        payload["selection"] = serialize_selection(selection)
        normal_selection = replace(
            selection,
            entries=filter_entries_by_board(selection.entries, BOARD_NORMAL),
        )
        payload["charts"] = serialize_charts(normal_selection)
        payload["errors"] = errors
        return payload

    # ----- Score history (DB-backed; all official scores, no dedup) -----
    @app.get("/api/score-history/options")
    def score_history_opts() -> dict[str, Any]:
        try:
            return score_history_options(store=resolved_store)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"读取分数历史选项失败：{exc}") from exc

    @app.get("/api/score-history")
    def score_history_endpoint(
        model: str = Query(...),
        benchmark: str = Query(...),
    ) -> dict[str, Any]:
        try:
            return score_history(model=model, benchmark=benchmark, store=resolved_store)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"读取分数历史失败：{exc}") from exc

    @app.get("/api/score-history/detail")
    def score_history_detail_endpoint(task_id: int = Query(...)) -> dict[str, Any]:
        try:
            return score_history_detail(task_id=task_id, store=resolved_store)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"读取分数详情失败：{exc}") from exc

    @app.get("/api/eval-records")
    def eval_records(
        task_id: int = Query(...),
        only_wrong: bool = Query(default=False),
        limit: int = Query(default=EVAL_PAGE_SIZE, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        try:
            return load_eval_records(
                task_id=task_id,
                only_wrong=only_wrong,
                limit=limit,
                offset=offset,
                store=resolved_store,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"读取 eval 记录失败：{exc}") from exc

    @app.get("/api/eval-context")
    def eval_context(
        task_id: int = Query(...),
        sample_index: int = Query(...),
        repeat_index: int = Query(default=0),
        pass_index: int = Query(default=0),
    ) -> dict[str, Any]:
        return load_eval_context(
            task_id=task_id,
            sample_index=sample_index,
            repeat_index=repeat_index,
            pass_index=pass_index,
            store=resolved_store,
        )

    # --- Scheduler admin routes ------------------------------------------
    register_admin_routes(app)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
