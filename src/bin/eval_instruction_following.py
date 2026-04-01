from __future__ import annotations

"""Compatibility wrapper for the unified instruction-following runner."""

import sys
from typing import Sequence

from src.bin.instruction_following_runner import main as _instruction_following_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    args.append("--no-param-search")
    return _instruction_following_main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
