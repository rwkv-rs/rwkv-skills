from __future__ import annotations

"""Compatibility wrapper for the unified function-calling runner."""

import sys
from typing import Sequence

from src.bin.function_calling_runner import main as _function_calling_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    args.extend(["--benchmark-kind", "browsecomp"])
    return _function_calling_main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
