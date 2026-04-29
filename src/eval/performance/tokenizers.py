from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Protocol, Sequence

from transformers import AutoTokenizer


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
        vocab_path = (
            Path(tokenizer_path).expanduser().resolve()
            if tokenizer_path
            else (Path(__file__).resolve().parents[2] / "infer" / "rwkv7" / "rwkv_vocab_v20230424.txt")
        )
        if not vocab_path.exists():
            raise FileNotFoundError(f"RWKV tokenizer vocab 不存在: {vocab_path}")
        trie_tokenizer_cls = _load_rwkv_trie_tokenizer_class()
        return cls(tokenizer_path=str(vocab_path), tokenizer=trie_tokenizer_cls(str(vocab_path)))

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
    utils_path = Path(__file__).resolve().parents[2] / "infer" / "rwkv7" / "utils.py"
    spec = importlib.util.spec_from_file_location("rwkv_perf_utils", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 RWKV tokenizer utils: {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trie_tokenizer_cls = getattr(module, "TRIE_TOKENIZER", None)
    if trie_tokenizer_cls is None:
        raise RuntimeError(f"RWKV tokenizer utils 缺少 TRIE_TOKENIZER: {utils_path}")
    return trie_tokenizer_cls


__all__ = [
    "BenchmarkTokenizer",
    "HfTokenizerAdapter",
    "RwkvTokenizerAdapter",
    "load_benchmark_tokenizer",
]
