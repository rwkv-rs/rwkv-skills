"""FastAPI web layer for the RWKV Skills dashboard.

HTTP endpoints + JSON serialisation built on top of :mod:`src.dashboard.core`:
``api`` (leaderboard / eval-records / eval-context), ``admin_api`` (scheduler
control), ``serialize`` / ``charts_json`` (pivot & chart JSON), and
``eval_service`` (eval records, context, and DB→score-index rebuild).
"""
