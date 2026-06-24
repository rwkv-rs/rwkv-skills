"""FastAPI application serving the RWKV Skills evaluation dashboard.

Data sources (no Rust backend, no raw SQL in this layer):
  * leaderboard scores  ->  ``results/space/score_index.jsonl`` via ``load_scores``
  * eval records/context ->  the project's ``EvalDbService`` (Postgres)

The built React SPA (``frontend/dist``) is served as static files when present.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .admin_api import register_admin_routes
from .charts_json import serialize_charts
from .constants import (
    AUTO_MODEL_LABEL,
    DEFAULT_TABLE_VIEW,
    DOMAIN_GROUPS,
    EVAL_PAGE_SIZE,
    TABLE_VIEW_LABELS,
    _normalize_table_view,
)
from .data import ScoreEntry, list_models, load_scores
from .eval_service import load_eval_context, load_eval_records
from .selection import _compute_choices, _prepare_selection
from .serialize import serialize_leaderboard, serialize_selection

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPA_DIST = _REPO_ROOT / "frontend" / "dist"


class _ScoreCache:
    """Process-wide cache of parsed score entries, refreshable on demand."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[ScoreEntry] | None = None
        self._errors: list[str] = []

    def get(self) -> tuple[list[ScoreEntry], list[str]]:
        with self._lock:
            if self._entries is None:
                errors: list[str] = []
                self._entries = load_scores(errors)
                self._errors = errors
            return self._entries, self._errors

    def refresh(self) -> tuple[list[ScoreEntry], list[str]]:
        with self._lock:
            errors: list[str] = []
            self._entries = load_scores(errors)
            self._errors = errors
            return self._entries, self._errors


_cache = _ScoreCache()


def _domain_groups_payload() -> list[dict[str, Any]]:
    return [
        {"key": group["key"], "label": group["label"], "title": group["title"]}
        for group in DOMAIN_GROUPS
    ]


def create_app() -> FastAPI:
    app = FastAPI(title="RWKV Skills Dashboard", version="2.0")

    # Permissive CORS so the Vite dev server (localhost:5173) can call the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        entries, errors = _cache.get()
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
        entries, errors = _cache.refresh()
        return {"entry_count": len(entries), "errors": errors}

    @app.get("/api/leaderboard")
    def leaderboard(
        model: str | None = Query(default=None),
        view: str = Query(default=DEFAULT_TABLE_VIEW),
    ) -> dict[str, Any]:
        entries, errors = _cache.get()
        selection = _prepare_selection(entries, model or AUTO_MODEL_LABEL)
        payload = serialize_leaderboard(selection, all_entries=entries, view_mode=_normalize_table_view(view))
        payload["selection"] = serialize_selection(selection)
        payload["charts"] = serialize_charts(selection)
        payload["errors"] = errors
        return payload

    @app.get("/api/eval-records")
    def eval_records(
        task_id: int = Query(...),
        only_wrong: bool = Query(default=False),
        limit: int = Query(default=EVAL_PAGE_SIZE, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        try:
            return load_eval_records(
                task_id=task_id, only_wrong=only_wrong, limit=limit, offset=offset
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
        )

    # --- Scheduler admin routes (must precede the SPA catch-all) ----------
    register_admin_routes(app)

    # --- Static SPA (production build) ------------------------------------
    if _SPA_DIST.is_dir():
        assets = _SPA_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str) -> FileResponse:
            candidate = _SPA_DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(_SPA_DIST / "index.html"))

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
