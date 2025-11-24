from __future__ import annotations

"""Utility helpers for dataset preparation scripts (download, IO, HF cache, etc.)."""

import contextlib
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional
from collections.abc import Iterable, Iterator, Mapping

LOGGER = logging.getLogger(__name__)

_CHUNK_SIZE = 16 * 1024 * 1024
_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)}")


class DownloadError(RuntimeError):
    """Raised when a file download fails after all retries."""


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_cache_dir(root: Path, dataset_name: str) -> Path:
    cache_root = ensure_directory(root / "cache")
    return ensure_directory(cache_root / dataset_name)


def dataset_output_dir(root: Path, dataset_name: str, split: str) -> Path:
    return ensure_directory(root / dataset_name / split)


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_checksum(expected: str) -> tuple[str, str]:
    if ":" in expected:
        algorithm, digest = expected.split(":", 1)
        return algorithm.strip().lower(), digest.strip()
    digest = expected.strip()
    algorithm = "sha256" if len(digest) in (64, 128) else "md5"
    return algorithm, digest


def verify_checksum(path: Path, expected: str) -> bool:
    algorithm, digest = _parse_checksum(expected)
    actual = _hash_file(path, algorithm)
    if actual.lower() != digest.lower():
        LOGGER.warning(
            "Checksum mismatch for %s: expected %s=%s, got %s",
            path,
            algorithm,
            digest,
            actual,
        )
        return False
    return True


def download_file(
    url: str,
    target_path: Path,
    *,
    expected_checksum: str | None = None,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> Path:
    target_path = target_path.expanduser().resolve()
    ensure_directory(target_path.parent)

    if target_path.exists() and target_path.stat().st_size > 0:
        if expected_checksum is None or verify_checksum(target_path, expected_checksum):
            LOGGER.debug("Using cached file %s", target_path)
            return target_path
        LOGGER.info("Cached file %s failed checksum verification, redownloading", target_path)
        target_path.unlink()

    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        tmp_path: Path | None = None
        try:
            LOGGER.info("Downloading %s -> %s (attempt %d/%d)", url, target_path, attempt, attempts)
            with contextlib.ExitStack() as stack:
                response = stack.enter_context(urllib.request.urlopen(url))
                tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
                with stack.enter_context(tmp_path.open("wb")) as handle:
                    while True:
                        chunk = response.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
            os.replace(tmp_path, target_path)
        except (urllib.error.URLError, OSError) as exc:
            LOGGER.warning("Download failed (%s): %s", url, exc)
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            if attempt == attempts:
                raise DownloadError(f"Failed to download {url}") from exc
            time.sleep(retry_delay * attempt)
            continue

        if expected_checksum and not verify_checksum(target_path, expected_checksum):
            target_path.unlink(missing_ok=True)
            if attempt == attempts:
                raise DownloadError(f"Checksum mismatch for {url}")
            time.sleep(retry_delay * attempt)
            continue
        return target_path

    raise DownloadError(f"Failed to download {url}")


def unpack_archive(archive_path: Path, destination: Path, *, keep_archive: bool = True) -> Path:
    archive_path = archive_path.expanduser().resolve()
    destination = ensure_directory(destination.expanduser().resolve())

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(destination)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zipper:
            zipper.extractall(destination)
    elif archive_path.suffix == ".gz" and not archive_path.name.endswith(".tar.gz"):
        output_path = destination / archive_path.stem
        with gzip.open(archive_path, "rb") as fin, output_path.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    if not keep_archive:
        archive_path.unlink(missing_ok=True)

    return destination


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    ensure_directory(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    return path


def list_files(root: Path, suffix: str | None = None) -> list[Path]:
    root = root.expanduser().resolve()
    if suffix:
        return sorted(p for p in root.iterdir() if p.is_file() and p.suffix == suffix)
    return sorted(p for p in root.iterdir() if p.is_file())


def cleanup_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def configure_hf_home(root: Path | None = None) -> Path:
    default_root = Path(os.environ.get("RWKV_SKILLS_HF_HOME", "data/hf_cache"))
    cache_root = ensure_directory((root or default_root).expanduser().resolve())
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    return cache_root


def iter_hf_dataset(
    dataset_id: str,
    *,
    config: str | None = None,
    split: str = "test",
    streaming: bool = False,
    **kwargs: Any,
) -> Iterable[Mapping[str, Any]]:
    configure_hf_home()
    from datasets import load_dataset  # local import to speed up module import

    dataset = load_dataset(dataset_id, config, split=split, streaming=streaming, **kwargs)
    if streaming:
        return dataset
    return (example for example in dataset)


def extract_answer_from_solution(text: str, *, regex: str | None = r"The final answer is (.+)$") -> str | None:
    if not text:
        return None
    matches = _BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    if regex:
        search = re.search(regex, text, flags=re.MULTILINE)
        if search:
            return search.group(1).strip()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line
    return None


def load_qwen_dataset(dataset: str, split: str = "test") -> list[dict]:
    url = (
        "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/refs/heads/main/evaluation/data/{dataset}/{split}.jsonl"
    )
    cache_root = dataset_cache_dir(Path("data"), "qwen_math")
    target_dir = ensure_directory(cache_root / dataset)
    raw_path = target_dir / f"original_{split}.jsonl"

    if not raw_path.exists():
        download_file(url.format(dataset=dataset, split=split), raw_path)

    rows: list[dict] = []
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if "answer" in payload:
                payload["expected_answer"] = payload.pop("answer")
            if "problem" not in payload and "question" in payload:
                payload["problem"] = payload.pop("question")
            if dataset == "olympiadbench" and "final_answer" in payload:
                answers = payload.pop("final_answer")
                if isinstance(answers, (list, tuple)) and answers:
                    payload["expected_answer"] = str(answers[0]).strip("$")
            if dataset == "minerva_math" and "solution" in payload:
                extracted = extract_answer_from_solution(payload["solution"])
                if extracted is not None:
                    payload["expected_answer"] = extracted
            rows.append(payload)

    raw_path.unlink(missing_ok=True)
    return rows


__all__ = [
    "DownloadError",
    "cleanup_directory",
    "configure_hf_home",
    "dataset_cache_dir",
    "dataset_output_dir",
    "download_file",
    "ensure_directory",
    "extract_answer_from_solution",
    "iter_hf_dataset",
    "list_files",
    "load_qwen_dataset",
    "read_jsonl",
    "unpack_archive",
    "verify_checksum",
    "write_jsonl",
]
