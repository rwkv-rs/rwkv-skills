from __future__ import annotations

"""Compatibility wrapper for the unified coding runner."""

import sys
from typing import Sequence

from src.bin.coding_runner import main as _coding_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    args.extend(["--benchmark-kind", "human_eval", "--cot-mode", "no_cot"])
    return _coding_main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
