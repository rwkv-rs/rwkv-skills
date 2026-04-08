from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable, Sequence

from ..data_utils import configure_hf_home
from src.eval.datasets.data_prepper.prepper_registry import INSTRUCTION_FOLLOWING_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec

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


def _records(
    split: str,
    *,
    source_languages: Sequence[str] = DEFAULT_SOURCE_LANGUAGES,
    target_languages: Sequence[str] = DEFAULT_TARGET_LANGUAGES,
) -> Iterable[dict]:
    if split not in {"dev", "devtest"}:
        raise ValueError("flores200 仅支持 dev 或 devtest split")

    all_languages = _normalize_languages(source_languages, target_languages)
    datasets = {lang: _load_language_data(lang, split) for lang in all_languages}
    return _generate_rows(datasets, source_languages, target_languages)


@INSTRUCTION_FOLLOWING_REGISTRY.register_spec("flores200")
def prepare_flores200_spec(
    output_root: Path,
    split: str = "devtest",
    *,
    source_languages: Sequence[str] = DEFAULT_SOURCE_LANGUAGES,
    target_languages: Sequence[str] = DEFAULT_TARGET_LANGUAGES,
) -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec(
        "flores200",
        output_root,
        split,
        load_rows=lambda resolved_split, _src=tuple(source_languages), _tgt=tuple(target_languages): _records(
            resolved_split,
            source_languages=_src,
            target_languages=_tgt,
        ),
        source_kind="hf_load_dataset",
        manifest_extra_factory=lambda resolved_split, _src=tuple(source_languages), _tgt=tuple(target_languages): {
            "dataset_id": DATASET_ID,
            "source_split": resolved_split,
            "source_languages": list(_src),
            "target_languages": list(_tgt),
        },
    )


__all__ = ["prepare_flores200_spec"]
