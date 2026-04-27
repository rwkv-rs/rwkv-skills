from __future__ import annotations

"""Compatibility wrapper for the renamed function-call evaluator."""

from src.bin.eval_function_call import *  # noqa: F401,F403
from src.bin.eval_function_call import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
