from __future__ import annotations

"""Compatibility export for the maths pipeline."""

from src.eval.maths.pipeline import (
    DEFAULT_COT_PROMPT,
    DEFAULT_DIRECT_PROMPT,
    FREE_RESPONSE_STOP_TOKENS,
    USER_SENTINEL,
    FreeResponsePipeline,
    FreeResponsePipelineResult,
)

__all__ = [
    "DEFAULT_COT_PROMPT",
    "DEFAULT_DIRECT_PROMPT",
    "FREE_RESPONSE_STOP_TOKENS",
    "USER_SENTINEL",
    "FreeResponsePipeline",
    "FreeResponsePipelineResult",
]
