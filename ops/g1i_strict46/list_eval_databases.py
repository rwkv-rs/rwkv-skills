#!/usr/bin/env python3
"""List PostgreSQL databases reachable through the evaluation DB config."""

from __future__ import annotations

import psycopg

from src.db.pool import _build_conninfo
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


def main() -> None:
    with psycopg.connect(_build_conninfo(DEFAULT_DB_CONFIG)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT datname FROM pg_database "
                "WHERE NOT datistemplate ORDER BY datname"
            )
            for (name,) in cursor.fetchall():
                print(name)


if __name__ == "__main__":
    main()
