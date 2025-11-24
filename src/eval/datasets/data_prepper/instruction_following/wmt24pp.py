from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from collections.abc import Iterable, Sequence

from ..data_utils import configure_hf_home, write_jsonl
from src.dataset.data_prepper.prepper_registry import INSTRUCTION_FOLLOWING_REGISTRY

DATASET_ID = "google/wmt24pp"
DEFAULT_TARGET_LANGUAGES: Sequence[str] = ("de_DE", "es_MX", "fr_FR", "it_IT", "ja_JP")

LANGUAGE_NAMES: dict[str, str] = {
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "en": "English",
}


def _lang_display(lang: str) -> str:
    base = lang.split("_", 1)[0]
    return LANGUAGE_NAMES.get(base, lang)


def _load_language_pair(target_language: str) -> list[tuple[str, str]]:
    configure_hf_home()
    from datasets import load_dataset

    config = f"en-{target_language}"
    dataset = load_dataset(DATASET_ID, config, split="train")
    return [(example["source"], example["target"]) for example in dataset]


def _generate_rows(target_languages: Sequence[str]) -> Iterable[dict]:
    for tgt_lang in target_languages:
        pairs = _load_language_pair(tgt_lang)
        for source_text, target_text in pairs:
            yield {
                "text": source_text,
                "translation": target_text,
                "source_language": "en",
                "target_language": tgt_lang,
                "source_lang_name": _lang_display("en"),
                "target_lang_name": _lang_display(tgt_lang),
            }


@INSTRUCTION_FOLLOWING_REGISTRY.register("wmt24pp")
def prepare_wmt24pp(
    output_root: Path,
    split: str = "test",
    *,
    target_languages: Sequence[str] = DEFAULT_TARGET_LANGUAGES,
) -> list[Path]:
    if split != "test":
        raise ValueError("wmt24pp 仅提供 test split")

    dataset_dir = output_root / "wmt24pp"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"

    write_jsonl(target, _generate_rows(target_languages))
    return [target]


__all__ = ["prepare_wmt24pp"]
