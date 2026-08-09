from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable


def iter_jsonl(
    path: str | Path,
    *,
    loads: Callable[[str], Any] | None = None,
) -> Iterable[Any]:
    """Yield parsed JSON objects from a UTF-8 JSONL file."""
    parser = loads or json.loads
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield parser(line)


__all__ = ["iter_jsonl"]
