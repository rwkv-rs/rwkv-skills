"""Leaderboard board classification: 正式榜 (normal) vs 朴素榜 (naive).

A score belongs to the 朴素榜 (naive board) when it was produced by a ``_naive``
runner. The ``scores`` table itself only stores ``cot_mode``/``metrics`` — the
naive flag lives on the ``task`` row, so this must be derived from
``task.evaluator`` and ``task.sampling_config->>'prompt_profile'`` (both carried
into each :class:`ScoreEntry` by the DB→score-index rebuild). The robust rule,
per the eval pipeline:

    prompt_profile == 'naive'  OR  evaluator LIKE '%_naive'   →  naive
    otherwise                                                  →  normal
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from .data import ScoreEntry

BOARD_NORMAL = "normal"
BOARD_NAIVE = "naive"

BOARDS: list[dict[str, str]] = [
    {"key": BOARD_NORMAL, "label": "正式榜"},
    {"key": BOARD_NAIVE, "label": "朴素榜"},
]


def _prompt_profile(entry: ScoreEntry) -> str | None:
    sampling = (entry.extra or {}).get("sampling_config")
    if isinstance(sampling, str):
        try:
            sampling = json.loads(sampling)
        except (ValueError, TypeError):
            return None
    if isinstance(sampling, dict):
        value = sampling.get("prompt_profile")
        return str(value) if value is not None else None
    return None


def is_naive_meta(evaluator: Any, sampling_config: Any) -> bool:
    """Raw-row variant of :func:`is_naive_entry` (for DB rows, not ScoreEntry).

    ``prompt_profile == 'naive'`` OR ``evaluator LIKE '%_naive'`` → 朴素榜.
    """
    if str(evaluator or "").endswith("_naive"):
        return True
    config = sampling_config
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except (ValueError, TypeError):
            config = None
    if isinstance(config, dict):
        return config.get("prompt_profile") == "naive"
    return False


def is_naive_entry(entry: ScoreEntry) -> bool:
    """True when the score comes from a ``_naive`` runner (朴素榜)."""
    evaluator = str(entry.task or "")
    if evaluator.endswith("_naive"):
        return True
    return _prompt_profile(entry) == "naive"


def board_of(entry: ScoreEntry) -> str:
    return BOARD_NAIVE if is_naive_entry(entry) else BOARD_NORMAL


def filter_entries_by_board(entries: Iterable[ScoreEntry], board: str) -> list[ScoreEntry]:
    """Return only the entries belonging to ``board`` (normal/naive).

    An unknown/empty board falls back to the normal board so the default view is
    always the 正式刷分线.
    """
    target = BOARD_NAIVE if str(board) == BOARD_NAIVE else BOARD_NORMAL
    return [entry for entry in entries if board_of(entry) == target]


def normalize_board(board: Any) -> str:
    return BOARD_NAIVE if str(board) == BOARD_NAIVE else BOARD_NORMAL


__all__ = [
    "BOARDS",
    "BOARD_NAIVE",
    "BOARD_NORMAL",
    "board_of",
    "filter_entries_by_board",
    "is_naive_entry",
    "is_naive_meta",
    "normalize_board",
]
