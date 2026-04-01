from __future__ import annotations

"""CLI wrapper for the field-oriented maths runner."""

from src.eval.maths.runner import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
