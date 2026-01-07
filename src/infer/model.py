from __future__ import annotations

"""RWKV 模型加载与命名配置。"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import types

from .rwkv7.rwkv7 import RWKV_x070
from .rwkv7.utils import TRIE_TOKENIZER


class ArchVersion(str, Enum):
    RWKV7 = "rwkv7"


class DataVersion(str, Enum):
    G0 = "g0"
    G0A = "g0a"
    G0A2 = "g0a2"
    G0A3 = "g0a3"
    G0A4 = "g0a4"
    G0B = "g0b"
    G0C = "g0c"
    G1 = "g1"
    G1A = "g1a"
    G1A2 = "g1a2"
    G1A3 = "g1a3"
    G1A4 = "g1a4"
    G1B = "g1b"
    G1C = "g1c"


class ParamSize(str, Enum):
    P0_1B = "0_1b"
    P0_4B = "0_4b"
    P1_5B = "1_5b"
    P2_9B = "2_9b"
    P7_2B = "7_2b"
    P13_3B = "13_3b"


@dataclass(slots=True)
class ModelLoadConfig:
    weights_path: str
    device: str = "cuda"
    tokenizer_path: str | None = None
    arch_version: ArchVersion = ArchVersion.RWKV7
    data_version: DataVersion | None = None
    num_params: ParamSize | None = None


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
    "ArchVersion",
    "DataVersion",
    "ParamSize",
    "ModelLoadConfig",
    "load_rwkv_model",
]
