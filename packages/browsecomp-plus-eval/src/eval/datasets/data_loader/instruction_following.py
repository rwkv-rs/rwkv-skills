from __future__ import annotations

import json
from .base import JsonlDatasetLoader

from src.eval.datasets.data_struct.instruction_following import (
    InstructionFollowingDataset,
    InstructionFollowingRecord,
)


class JsonlInstructionFollowingLoader(
    JsonlDatasetLoader[InstructionFollowingRecord, InstructionFollowingDataset]
):
    """解析 IFEval/ifbench 的 prompt + instruction_id 列表。"""

    dataset_cls = InstructionFollowingDataset

    def _parse_record(self, payload: dict) -> InstructionFollowingRecord:
        """Validate raw JSON and convert it into InstructionFollowingRecord."""
        key = payload.get("key")
        prompt = payload.get("prompt")
        instruction_id_list = payload.get("instruction_id_list")
        kwargs = payload.get("kwargs")
        if not isinstance(key, int):
            raise ValueError("key 字段必须为整数")
        if not isinstance(prompt, str):
            raise ValueError("prompt 字段必须为字符串")
        if not isinstance(instruction_id_list, list) or not all(
            isinstance(item, str) for item in instruction_id_list
        ):
            raise ValueError("instruction_id_list 必须为字符串列表")
        if not isinstance(kwargs, list) or not all(
            isinstance(item, dict) for item in kwargs
        ):
            raise ValueError("kwargs 必须为字典列表")
        return InstructionFollowingRecord(
            key=key,
            prompt=prompt,
            instruction_ids=list(instruction_id_list),
            kwargs_list=list(kwargs),
        )
