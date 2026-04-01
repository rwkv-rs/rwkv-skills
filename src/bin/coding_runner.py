from __future__ import annotations

"""CLI wrapper for the field-oriented coding runner."""

from src.eval.coding.runner import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
