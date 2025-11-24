from __future__ import annotations

from collections.abc import Sequence

from src.dataset.data_struct.code_generation import (
    CodeGenerationDataset,
    CodeGenerationRecord,
)
from .base import JsonlDatasetLoader

PROMPT_KEYS: tuple[str, ...] = (
    "prompt",
    "question",
    "instruction",
    "text",
    "description",
)
TASK_ID_KEYS: tuple[str, ...] = (
    "task_id",
    "problem_id",
    "question_id",
    "id",
)


class JsonlCodeGenerationLoader(
    JsonlDatasetLoader[CodeGenerationRecord, CodeGenerationDataset]
):
    """读取 EvalPlus 样式 JSONL（MBPP/HS 等），结构化成代码生成记录。"""

    dataset_cls = CodeGenerationDataset

    def _extract_str(self, payload: dict, keys: Sequence[str], field_name: str) -> str:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError(f"{self.path}: {field_name} 字段缺失或类型错误, payload={payload}")

    def _parse_record(self, payload: dict) -> CodeGenerationRecord:
        """Normalise EvalPlus 字段（task_id/prompt/stubs/tests）为 dataclass。"""
        task_id = self._extract_str(payload, TASK_ID_KEYS, "task_id")
        prompt = self._extract_str(payload, PROMPT_KEYS, "prompt/question")
        starter_code = payload.get("starter_code") or payload.get("solution_stub")
        if starter_code is not None and not isinstance(starter_code, str):
            starter_code = str(starter_code)
        entry_point = payload.get("entry_point")
        if entry_point is not None and not isinstance(entry_point, str):
            entry_point = str(entry_point)
        canonical_solution = payload.get("canonical_solution")
        if canonical_solution is not None and not isinstance(canonical_solution, str):
            canonical_solution = str(canonical_solution)

        test_cases = (
            payload.get("test_list")
            or payload.get("test_cases")
            or payload.get("tests")
            or payload.get("unit_tests")
        )

        used_keys = set()
        for key in (*TASK_ID_KEYS, *PROMPT_KEYS, "starter_code", "solution_stub", "entry_point", "canonical_solution"):
            if key in payload:
                used_keys.add(key)
        for key in ("test_list", "test_cases", "tests", "unit_tests"):
            if key in payload:
                used_keys.add(key)

        metadata = {
            key: value
            for key, value in payload.items()
            if key not in used_keys
        }

        return CodeGenerationRecord(
            task_id=str(task_id),
            prompt=prompt,
            starter_code=starter_code,
            entry_point=entry_point,
            canonical_solution=canonical_solution,
            test_cases=test_cases,
            metadata=metadata,
        )
