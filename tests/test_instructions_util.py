from __future__ import annotations

import importlib
import sys

import nltk

MODULE_NAME = "src.eval.metrics.instruction_following.instructions_util"


def _reload_module():
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_instructions_util_import_has_no_nltk_download_side_effect(monkeypatch) -> None:
    downloads: list[str] = []
    monkeypatch.setattr(nltk, "download", lambda name, quiet=True: downloads.append(name))

    _reload_module()

    assert downloads == []


def test_count_sentences_initializes_nltk_resources_lazily(monkeypatch) -> None:
    module = _reload_module()
    module._ensure_nltk_resource.cache_clear()
    module._get_sentence_tokenizer.cache_clear()

    downloads: list[str] = []
    monkeypatch.setattr(module.nltk.data, "find", lambda _path: (_ for _ in ()).throw(LookupError()))
    monkeypatch.setattr(module.nltk, "download", lambda name, quiet=True: downloads.append(name))

    class _Tokenizer:
        def tokenize(self, text: str) -> list[str]:
            return [part for part in text.split(".") if part.strip()]

    monkeypatch.setattr(module.nltk.data, "load", lambda _path: _Tokenizer())

    assert downloads == []
    assert module.count_sentences("One sentence. Two sentence.") == 2
    assert downloads == ["punkt_tab", "punkt"]

