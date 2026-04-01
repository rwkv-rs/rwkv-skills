from __future__ import annotations

"""CLI wrapper for the field-oriented function-calling runner."""

from src.eval.function_calling.runner import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
