from __future__ import annotations

"""Compatibility export for the instruction-following pipeline."""

from src.eval.instruction_following.pipeline import (
    InstructionFollowingPipeline,
    InstructionFollowingPipelineResult,
)

__all__ = ["InstructionFollowingPipeline", "InstructionFollowingPipelineResult"]
