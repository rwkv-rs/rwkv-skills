from __future__ import annotations

"""RWKV 模型加载与命名配置。"""

from dataclasses import dataclass
from pathlib import Path
import types

from .rwkv7.rwkv7 import RWKV_x070
from .rwkv7.utils import TRIE_TOKENIZER


@dataclass(slots=True)
class ModelLoadConfig:
    weights_path: str
    device: str = "cuda"
    tokenizer_path: str | None = None


def load_rwkv_model(config: ModelLoadConfig):
    """Load RWKV 模型 + tokenizer。"""

    weights_path = Path(config.weights_path).expanduser().resolve()
    if weights_path.suffix == ".pth":
        base_path = weights_path.with_suffix("")
    else:
        base_path = weights_path
        weights_path = base_path.with_suffix(".pth")
    if not weights_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {weights_path}")

    default_tokenizer_path = Path(__file__).resolve().parent / "rwkv7" / "rwkv_vocab_v20230424.txt"
    tokenizer_path = (
        Path(config.tokenizer_path).expanduser().resolve()
        if config.tokenizer_path
        else default_tokenizer_path
    )
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer 词表不存在: {tokenizer_path}")

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    args.MODEL_NAME = str(base_path)
    model = RWKV_x070(args, device=config.device)
    tokenizer = TRIE_TOKENIZER(str(tokenizer_path))
    return model, tokenizer


__all__ = [
    "ModelLoadConfig",
    "load_rwkv_model",
]
