from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Protocol, Sequence


class BenchmarkTokenizer(Protocol):
    def encode(self, text: str) -> list[int]:  # pragma: no cover - protocol
        ...

    def decode(self, token_ids: Sequence[int]) -> str:  # pragma: no cover - protocol
        ...

    @property
    def label(self) -> str:  # pragma: no cover - protocol
        ...


@dataclass(slots=True)
class RwkvTokenizerAdapter:
    tokenizer_path: str
    tokenizer: object

    @classmethod
    def load(cls, tokenizer_path: str | None = None) -> "RwkvTokenizerAdapter":
        vocab_path = Path(tokenizer_path).expanduser().resolve() if tokenizer_path else None
        if vocab_path is not None and not vocab_path.exists():
            raise FileNotFoundError(f"RWKV tokenizer vocab 不存在: {vocab_path}")
        trie_tokenizer_cls = _load_rwkv_trie_tokenizer_class()
        tokenizer = trie_tokenizer_cls(str(vocab_path)) if vocab_path is not None else trie_tokenizer_cls()
        return cls(tokenizer_path=str(vocab_path or "pyrwkv-tokenizer-default"), tokenizer=tokenizer)

    @property
    def label(self) -> str:
        return f"rwkv:{self.tokenizer_path}"

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(token_ids)))


@dataclass(slots=True)
class HfTokenizerAdapter:
    reference: str
    tokenizer: object

    @classmethod
    def load(cls, reference: str) -> "HfTokenizerAdapter":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(reference, trust_remote_code=True)
        return cls(reference=reference, tokenizer=tokenizer)

    @property
    def label(self) -> str:
        return f"hf:{self.reference}"

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False))


def load_benchmark_tokenizer(
    *,
    tokenizer_type: str,
    tokenizer_ref: str | None,
) -> BenchmarkTokenizer:
    normalized = str(tokenizer_type).strip().lower()
    if normalized == "rwkv":
        return RwkvTokenizerAdapter.load(tokenizer_ref)
    if normalized == "hf":
        if not tokenizer_ref:
            raise ValueError("HF tokenizer 模式需要提供 --tokenizer-ref")
        return HfTokenizerAdapter.load(tokenizer_ref)
    raise ValueError(f"未知 tokenizer 类型: {tokenizer_type!r}")


def _load_rwkv_trie_tokenizer_class():
    errors: list[str] = []
    for module_name in ("pyrwkv_tokenizer", "rwkv_tokenizer", "rwkv.rwkv_tokenizer"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - report all attempted tokenizer modules.
            errors.append(f"{module_name}: {exc}")
            continue
        for attr_name in ("TRIE_TOKENIZER", "RWKV_TOKENIZER", "Tokenizer"):
            tokenizer_cls = getattr(module, attr_name, None)
            if tokenizer_cls is not None:
                return tokenizer_cls
        errors.append(f"{module_name}: missing tokenizer class")
    raise RuntimeError("无法加载外部 RWKV tokenizer；请安装 pyrwkv-tokenizer 或改用 --tokenizer-type hf。 " + "; ".join(errors))


__all__ = [
    "BenchmarkTokenizer",
    "HfTokenizerAdapter",
    "RwkvTokenizerAdapter",
    "load_benchmark_tokenizer",
]
