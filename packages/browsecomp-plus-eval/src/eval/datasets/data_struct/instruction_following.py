from __future__ import annotations

"""记录 IFEval / ifbench 问题：包含 key、prompt、instruction id 列表、kwargs。"""

from dataclasses import dataclass

from .base import JsonlDataset, RecordBase


@dataclass(slots=True)
class InstructionFollowingRecord(RecordBase):
    """每条样本包含 prompt、要验证的指令 id 列表及参数（kwargs）。"""

    key: int
    prompt: str
    instruction_ids: list[str]
    kwargs_list: list[dict]


class InstructionFollowingDataset(JsonlDataset[InstructionFollowingRecord]):
    """Wrapper so evaluators can treat IFEval prompts像普通 Sequence。"""


__all__ = [
    "InstructionFollowingRecord",
    "InstructionFollowingDataset",
]
