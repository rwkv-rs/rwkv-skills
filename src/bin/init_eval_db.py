from __future__ import annotations

"""Initialize evaluation database schema."""

import argparse
import sys

from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.infra.database import DatabaseManager


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize evaluation DB schema")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force initialization even if RWKV_DB_ENABLED is off.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = DEFAULT_DB_CONFIG
    if not config.enabled and not args.force:
        print("RWKV_DB_ENABLED is off. Use --force or set RWKV_DB_ENABLED=1.")
        return 2
    config.enabled = True
    db = DatabaseManager.instance()
    db.initialize(config)
    print("✅ eval DB schema initialized.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
