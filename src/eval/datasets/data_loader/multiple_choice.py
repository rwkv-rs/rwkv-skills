from __future__ import annotations

from src.dataset.data_struct.multiple_choice import (
    MultipleChoiceDataset,
    MultipleChoiceRecord,
)
from .base import JsonlDatasetLoader


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class JsonlMultipleChoiceLoader(
    JsonlDatasetLoader[MultipleChoiceRecord, MultipleChoiceDataset]
):
    """Parse CEVAL/MMLU 等多选 JSONL，产出 `MultipleChoiceRecord` 序列。

    每行 JSON 至少包含字段：
    - question: str
    - answer: str (对应选项字母，如 "A")
    - subject: str (可选)
    - 选项字段："A"、"B"、……，直到缺失为止。
    """

    dataset_cls = MultipleChoiceDataset

    def _parse_record(self, payload: dict) -> MultipleChoiceRecord:
        """Handle CEVAL/MMLU 等字段差异，把答案/选项归一化。"""
        question = payload.get("question")
        if not isinstance(question, str):
            raise ValueError("question 字段缺失或类型错误")

        raw_answer = payload.get("answer")
        answer_letter: str | None = None
        answer_index: int | None = None

        if isinstance(raw_answer, str):
            stripped = raw_answer.strip()
            if not stripped:
                raise ValueError("answer 字段缺失或类型错误")
            upper = stripped.upper()
            if upper in ALPHABET:
                answer_letter = upper
            elif stripped.isdigit():
                answer_index = int(stripped)
            else:
                raise ValueError(f"未知的 answer: {raw_answer}")
        elif isinstance(raw_answer, int):
            answer_index = raw_answer
        else:
            raise ValueError("answer 字段缺失或类型错误")

        choices: list[str] = []
        for letter in ALPHABET:
            if letter not in payload:
                break
            choice_text = payload[letter]
            if not isinstance(choice_text, str):
                raise ValueError(f"选项 {letter} 类型错误")
            choices.append(choice_text)

        if not choices:
            alt_choices = payload.get("choices")
            if isinstance(alt_choices, list):
                for idx, choice_text in enumerate(alt_choices):
                    if not isinstance(choice_text, str):
                        raise ValueError(f"choices[{idx}] 类型错误")
                    choices.append(choice_text)

        if not choices:
            raise ValueError("未检测到任何选项字段")

        if answer_letter is not None:
            answer_index = ALPHABET.index(answer_letter)
        else:
            assert answer_index is not None
            if answer_index >= len(choices) and 0 <= answer_index - 1 < len(choices):
                answer_index -= 1

        if answer_index is None or not (0 <= answer_index < len(choices)):
            raise ValueError(
                f"answer 索引 {answer_index} 超出选项数量 (len={len(choices)})"
            )

        subject = payload.get("subject")
        if subject is not None and not isinstance(subject, str):
            raise ValueError("subject 字段类型必须为 str 或 None")

        metadata = {
            k: v
            for k, v in payload.items()
            if k not in {"question", "answer", "subject", "choices"} and k not in ALPHABET
        }

        return MultipleChoiceRecord(
            question=question,
            choices=choices,
            answer_index=answer_index,
            subject=subject,
            metadata=metadata,
        )
