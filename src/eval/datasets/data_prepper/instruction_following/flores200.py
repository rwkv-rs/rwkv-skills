from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from collections.abc import Iterable, Sequence

from ..data_utils import configure_hf_home, write_jsonl
from src.dataset.data_prepper.prepper_registry import INSTRUCTION_FOLLOWING_REGISTRY

DATASET_ID = "openlanguagedata/flores_plus"
DEFAULT_SOURCE_LANGUAGES: Sequence[str] = ("en", "de", "es", "fr", "it", "ja")
DEFAULT_TARGET_LANGUAGES: Sequence[str] = ("en", "de", "es", "fr", "it", "ja")

FLORES_LANGUAGE_CONFIGS: dict[str, str] = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
}

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
}


def _normalize_languages(src_langs: Sequence[str], tgt_langs: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    for lang in (*src_langs, *tgt_langs):
        if lang not in ordered:
            ordered.append(lang)
    return ordered


def _load_language_data(lang: str, split: str) -> list[str]:
    config = FLORES_LANGUAGE_CONFIGS.get(lang)
    if config is None:
        raise ValueError(f"flores200 暂不支持语言 {lang}")
    configure_hf_home()
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, config, split=split)
    return [example["text"] for example in dataset]


def _lang_display(lang: str) -> str:
    return LANGUAGE_NAMES.get(lang, lang)


def _generate_rows(
    datasets: dict[str, list[str]],
    src_langs: Sequence[str],
    tgt_langs: Sequence[str],
) -> Iterable[dict]:
    for src_lang in src_langs:
        for tgt_lang in tgt_langs:
            if src_lang == tgt_lang:
                continue
            src_texts = datasets[src_lang]
            tgt_texts = datasets[tgt_lang]
            for src_text, tgt_text in zip(src_texts, tgt_texts, strict=True):
                yield {
                    "text": src_text,
                    "translation": tgt_text,
                    "source_language": src_lang,
                    "target_language": tgt_lang,
                    "source_lang_name": _lang_display(src_lang),
                    "target_lang_name": _lang_display(tgt_lang),
                }


@INSTRUCTION_FOLLOWING_REGISTRY.register("flores200")
def prepare_flores200(
    output_root: Path,
    split: str = "devtest",
    *,
    source_languages: Sequence[str] = DEFAULT_SOURCE_LANGUAGES,
    target_languages: Sequence[str] = DEFAULT_TARGET_LANGUAGES,
) -> list[Path]:
    if split not in {"dev", "devtest"}:
        raise ValueError("flores200 仅支持 dev 或 devtest split")

    all_languages = _normalize_languages(source_languages, target_languages)
    datasets = {lang: _load_language_data(lang, split) for lang in all_languages}

    dataset_dir = output_root / "flores200"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"

    write_jsonl(target, _generate_rows(datasets, source_languages, target_languages))
    return [target]


__all__ = ["prepare_flores200"]
