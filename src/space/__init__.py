"""Evaluation dashboard data layer for RWKV Skills.

Serves the FastAPI backend (``api``) behind the React/Vite SPA in ``frontend/``.
The pure-logic modules below are shared, framework-agnostic, and emit structured
data (JSON) rather than HTML.

Submodules
----------
api          – FastAPI app: leaderboard / eval-records / eval-context endpoints
serialize    – JSON serialisation of the pivot tables & selection state
charts_json  – JSON serialisation of the per-domain chart series
eval_service – EvalDbService wrapper for eval records & completion context
eval_records – eval record loading & context helpers
constants    – shared constants, dataclasses, tiny helpers
metrics      – metric extraction, score formatting, cell styling
selection    – model selection, lineage resolution, sorting
tables       – pivot table assembly & cell metadata
charts       – chart series computation
data         – score loading & normalisation
vocab        – tokeniser display helpers
"""

__all__: list[str] = []
