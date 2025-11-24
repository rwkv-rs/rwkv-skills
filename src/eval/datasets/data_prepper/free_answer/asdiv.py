from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
from collections.abc import Iterator

from ..data_utils import dataset_cache_dir, download_file, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY

DATA_URL = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml"


def _records(split: str) -> Iterator[dict]:
    if split != "test":
        raise ValueError("asdiv 仅提供 test split")
    cache_dir = dataset_cache_dir(Path("data"), "asdiv")
    source_path = cache_dir / "ASDiv.xml"
    download_file(DATA_URL, source_path)

    tree = ET.parse(source_path)
    root = tree.getroot()
    for problem in root.iter("Problem"):
        answer_text = problem.find("Answer").text or ""
        answer = answer_text.split("(")[0].strip()
        try:
            numeric = float(answer)
            if int(numeric) == numeric:
                answer = str(int(numeric))
            else:
                answer = str(numeric)
        except ValueError:
            pass
        yield {
            "problem": (problem.find("Body").text or "").strip() + " " + (problem.find("Question").text or "").strip(),
            "expected_answer": answer,
            "type": (problem.find("Solution-Type").text or "").strip(),
        }


@FREE_ANSWER_REGISTRY.register("asdiv")
def prepare_asdiv(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "asdiv"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _records(split))
    return [target]
