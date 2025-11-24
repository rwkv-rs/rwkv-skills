from __future__ import annotations

import json
from pathlib import Path

from ..data_utils import download_file, write_jsonl
from src.dataset.data_prepper.prepper_registry import INSTRUCTION_FOLLOWING_REGISTRY

IFEVAL_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/instruction_following_eval/data/input_data.jsonl"
)


@INSTRUCTION_FOLLOWING_REGISTRY("ifeval")
def prepare_ifeval(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("IFEval 数据集仅提供 test split")

    dataset_dir = (output_root / "ifeval").expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_path = dataset_dir / "input_data.jsonl"

    download_file(IFEVAL_URL, raw_path)

    records = []
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["question"] = payload.get("prompt", "")
            records.append(payload)

    target = dataset_dir / "test.jsonl"
    write_jsonl(target, records)
    return [target]


__all__ = ["prepare_ifeval"]
