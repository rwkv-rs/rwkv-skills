from __future__ import annotations

import json
from typing import Any
from collections.abc import Sequence

from src.dataset.data_struct.free_answer import FreeAnswerDataset, FreeAnswerRecord
from .base import JsonlDatasetLoader


CANONICAL_QUESTION_KEYS: Sequence[str] = (
    "question",
    "problem",
    "prompt",
    "input",
    "instruction",
    "query",
    "task",
    "text",
)
CANONICAL_ANSWER_KEYS: Sequence[str] = (
    "answer",
    "expected_answer",
    "reference_answer",
    "reference_solution",
    "solution",
    "output",
    "target",
    "final_answer",
    "translation",
)
CANONICAL_SUBJECT_KEYS: Sequence[str] = (
    "subject",
    "category",
    "domain",
    "topic",
    "subset",
    "dataset",
    "tag",
    "group",
)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        raise TypeError("无法将 None 转换为字符串")
    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return json.dumps(value, ensure_ascii=False)


class JsonlFreeAnswerLoader(
    JsonlDatasetLoader[FreeAnswerRecord, FreeAnswerDataset]
):
    """兼容数学/翻译/生成任务的 JSONL，自适应常见字段别名。"""

    dataset_cls = FreeAnswerDataset

    def _extract_field(
        self,
        payload: dict,
        keys: Sequence[str],
        *,
        field_name: str,
        optional: bool = False,
    ) -> tuple[str | None, str | None]:
        """返回 (value, key)，并记录哪个别名被命中，方便生成 metadata。"""
        for key in keys:
            if key in payload:
                value = payload[key]
                if value is None:
                    continue
                try:
                    return _stringify(value), key
                except TypeError:
                    continue
        if optional:
            return None, None
        raise ValueError(f"{self.path}: 无法解析 {field_name} 字段，payload={payload}")

    def _parse_record(self, payload: dict) -> FreeAnswerRecord:
        """从任意 free-form JSON 里提取 question/answer/subject。"""
        question, question_key = self._extract_field(
            payload,
            CANONICAL_QUESTION_KEYS,
            field_name="question",
        )
        if question is None:
            raise AssertionError("mandatory question field missing")
        answer, answer_key = self._extract_field(
            payload,
            CANONICAL_ANSWER_KEYS,
            field_name="answer",
        )
        if answer is None:
            raise AssertionError("mandatory answer field missing")
        subject, subject_key = self._extract_field(
            payload,
            CANONICAL_SUBJECT_KEYS,
            field_name="subject",
            optional=True,
        )
        used_keys = {key for key in (question_key, answer_key, subject_key) if key}
        metadata = {
            k: v
            for k, v in payload.items()
            if k not in used_keys
        }
        return FreeAnswerRecord(
            question=question,
            answer=answer,
            subject=subject,
            metadata=metadata,
        )
