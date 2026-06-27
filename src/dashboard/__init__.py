"""Evaluation dashboard backend for RWKV Skills (FastAPI + React/Vite SPA).

Split into two layers so the framework-agnostic logic stays testable and reusable:

``web`` – FastAPI app + JSON serialisation (api, admin_api, serialize,
          charts_json, eval_service). Entry point: ``src.dashboard.web.api:app``.
``core`` – pure leaderboard logic (data, metrics, selection, tables, charts,
           domains, constants, vocab, score_index, eval_records).

The React/Vite SPA lives separately under ``frontend/``.
"""

__all__: list[str] = []
