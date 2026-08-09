"""Disabled legacy merger for free-response global audit artifacts."""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "legacy free-response audit merge is disabled because it cannot prove "
        "PostgreSQL snapshot, metadata/dataset digests, module SHAs, exact "
        "group×partition coverage, or per-task row counts; use "
        "scripts/oneoff/merge_free_response_global_audit.py"
    )


if __name__ == "__main__":
    main()
