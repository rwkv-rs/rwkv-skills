from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List
from collections.abc import Iterable

try:
    from datasets import load_dataset  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

from ..data_utils import configure_hf_home, download_file, write_jsonl
from src.dataset.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY

LANG_LIBS_URL = (
    "https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/"
    "refs/heads/main/lm_eval/tasks/mmlu_prox/lang_libs.py"
)


def _load_lang_libs(cache_dir: Path) -> tuple[dict, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "lang_libs.py"
    download_file(LANG_LIBS_URL, target)

    spec = importlib.util.spec_from_file_location("lang_libs", target)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法导入 lang_libs 模块")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "LANG_LIBS") or not hasattr(module, "LANG_SUBJECTS"):
        raise RuntimeError("lang_libs 缺少期望字段")
    return module.LANG_LIBS, module.LANG_SUBJECTS


def _iter_records(split: str, languages: list[str], cache_dir: Path) -> Iterable[dict]:
    configure_hf_home()
    if load_dataset is None:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 mmlu-prox 数据集，请运行 `pip install datasets`"
        )
    lang_libs, lang_subjects = _load_lang_libs(cache_dir)
    for language in languages:
        dataset = load_dataset("li-lab/MMLU-ProX", language, split=split)
        lib = lang_libs[language]
        subject_map = lang_subjects[language]

        for row in dataset:
            category = row.get("category", "").replace(" ", "_")
            options: list[str] = []
            for idx in range(10):
                value = row.get(f"option_{idx}")
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    options.append(text)
            answer_letter = str(row.get("answer", "")).strip().upper()
            answer_idx = ord(answer_letter) - ord("A")
            if not (0 <= answer_idx < len(options)):
                raise ValueError(f"答案索引越界: {answer_letter}")

            subject = subject_map.get(category, category)
            description = lib[3].format(subject=subject, ans_suffix=lib[5].format("X")) + "\n"
            choices = [choice for choice in options if choice]
            question_text = row.get("question", "").strip()
            payload: dict[str, object] = {
                "question": f"{description}{lib[0]}\n{row.get('question', '').strip()}\n{lib[1]}",
                "answer": answer_letter,
                "subject": category,
                "language": language,
                "subset": language,
                "category": category,
            }
            for idx, choice in enumerate(choices):
                payload[chr(ord("A") + idx)] = choice
            payload["description"] = description.strip()
            yield payload


@MULTIPLE_CHOICE_REGISTRY.register("mmlu-prox")
def prepare_mmlu_prox(output_root: Path, split: str = "test") -> list[Path]:
    if split not in {"validation", "test"}:
        raise ValueError("mmlu-prox 仅支持 validation 与 test split")
    dataset_dir = output_root / "mmlu-prox"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    cache_dir = dataset_dir / "_cache"
    languages = ["en", "de", "es", "fr", "it", "ja"]
    write_jsonl(target, _iter_records(split, languages, cache_dir))
    return [target]
