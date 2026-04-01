from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .download import UrlDownloadFile, download_url_files

PARQUET_ENDPOINT = "https://datasets-server.huggingface.co/parquet"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


@dataclass(frozen=True, slots=True)
class ParquetFile:
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int

    def relative_path(self) -> Path:
        return Path(self.config) / self.split / self.filename


def _read_json_response(url: str, params: dict[str, str]) -> dict:
    query = urllib.parse.urlencode(params)
    target = f"{url}?{query}"
    with urllib.request.urlopen(target, timeout=60.0) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def get_parquet_files(dataset: str) -> list[ParquetFile]:
    payload = _read_json_response(PARQUET_ENDPOINT, {"dataset": dataset})
    return [ParquetFile(**row) for row in payload.get("parquet_files", [])]


def get_split_row_count(dataset: str, config: str, split: str) -> int:
    payload = _read_json_response(
        ROWS_ENDPOINT,
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": "0",
            "length": "1",
        },
    )
    return int(payload["num_rows_total"])


def download_hf_parquet_splits(
    path: str | Path,
    root_name: str,
    dataset: str,
    config: str,
    splits: list[str] | tuple[str, ...],
    tasks: int,
) -> Path:
    files = [
        UrlDownloadFile(relative_path=file.relative_path(), url=file.url)
        for file in get_parquet_files(dataset)
        if file.config == config and file.split in splits
    ]
    if not files:
        raise FileNotFoundError(
            f"no parquet files found for dataset={dataset} config={config} splits={tuple(splits)}"
        )
    return download_url_files(path, root_name, files, tasks)


__all__ = [
    "ParquetFile",
    "download_hf_parquet_splits",
    "get_parquet_files",
    "get_split_row_count",
]
