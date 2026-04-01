from __future__ import annotations

"""Compatibility wrapper for the unified knowledge runner."""

import sys
from typing import Sequence

from src.bin.knowledge_runner import main as _knowledge_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    args.extend(["--cot-mode", "fake_cot"])
    return _knowledge_main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
