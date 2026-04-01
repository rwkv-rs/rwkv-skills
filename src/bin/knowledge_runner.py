from __future__ import annotations

"""CLI wrapper for the field-oriented knowledge runner."""

from src.eval.knowledge.runner import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
