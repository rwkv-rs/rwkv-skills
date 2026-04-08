from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable, Sequence

from ..data_utils import configure_hf_home
from src.eval.datasets.data_prepper.prepper_registry import INSTRUCTION_FOLLOWING_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec

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


def _records(split: str, *, target_languages: Sequence[str] = DEFAULT_TARGET_LANGUAGES) -> Iterable[dict]:
    if split != "test":
        raise ValueError("wmt24pp 仅提供 test split")
    return _generate_rows(target_languages)


@INSTRUCTION_FOLLOWING_REGISTRY.register_spec("wmt24pp")
def prepare_wmt24pp_spec(
    output_root: Path,
    split: str = "test",
    *,
    target_languages: Sequence[str] = DEFAULT_TARGET_LANGUAGES,
) -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec(
        "wmt24pp",
        output_root,
        split,
        load_rows=lambda resolved_split, _targets=tuple(target_languages): _records(
            resolved_split,
            target_languages=_targets,
        ),
        source_kind="hf_load_dataset",
        manifest_extra_factory=lambda resolved_split, _targets=tuple(target_languages): {
            "dataset_id": DATASET_ID,
            "source_split": resolved_split,
            "target_languages": list(_targets),
        },
    )


__all__ = ["prepare_wmt24pp_spec"]
