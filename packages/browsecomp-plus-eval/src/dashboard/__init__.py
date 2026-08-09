"""Evaluation dashboard backend for RWKV Skills (FastAPI API for a Next frontend).

Split into two layers so the framework-agnostic logic stays testable and reusable:

``web`` – FastAPI app + JSON serialisation (api, admin_api, serialize,
          charts_json, eval_service). Entry point: ``src.dashboard.web.api:app``.
``core`` – pure leaderboard logic (data, metrics, selection, tables, charts,
           domains, constants, vocab, score_index, eval_records).

The Next.js frontend lives separately under ``client/``.
"""

__all__: list[str] = []
