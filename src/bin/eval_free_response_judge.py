from __future__ import annotations

"""Compatibility wrapper for the unified maths runner."""

import sys
from typing import Sequence

from src.bin.maths_runner import main as _maths_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    args.extend(["--judge-mode", "llm"])
    return _maths_main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
