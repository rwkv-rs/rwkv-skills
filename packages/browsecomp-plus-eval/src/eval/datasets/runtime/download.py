from __future__ import annotations

import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


@dataclass(frozen=True, slots=True)
class UrlDownloadFile:
    relative_path: Path
    url: str


def _repo_dir_name(repo: str) -> str:
    return repo.rstrip("/").split("/")[-1].replace(".git", "")


def _download_single_url_file(
    root_dir: Path,
    file: UrlDownloadFile,
    *,
    retries: int,
    timeout: float,
    retry_delay: float,
) -> Path:
    destination = root_dir / file.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(file.url, timeout=timeout) as response:
                with tmp_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            tmp_path.replace(destination)
            return destination
        except (urllib.error.URLError, OSError) as exc:
            tmp_path.unlink(missing_ok=True)
            if attempt >= attempts:
                raise RuntimeError(f"failed to download {file.url}: {exc}") from exc
            time.sleep(retry_delay * attempt)
    raise RuntimeError(f"failed to download {file.url}")


def download_url_files(
    path: str | Path,
    root_name: str,
    files: Iterable[UrlDownloadFile],
    tasks: int,
    *,
    retries: int = 5,
    timeout: float = 60.0,
    retry_delay: float = 2.0,
) -> Path:
    if tasks <= 0:
        raise ValueError("tasks must be positive")

    target_root = (Path(path).expanduser().resolve() / root_name).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    materialized = list(files)
    if not materialized:
        raise ValueError("files must not be empty")

    with ThreadPoolExecutor(max_workers=tasks) as executor:
        futures = [
            executor.submit(
                _download_single_url_file,
                target_root,
                file,
                retries=retries,
                timeout=timeout,
                retry_delay=retry_delay,
            )
            for file in materialized
        ]
        for future in as_completed(futures):
            future.result()
    return target_root


def download_hf_repo(
    path: str | Path,
    repo: str,
    *,
    revision: str = "main",
    repo_type: str = "dataset",
    root_name: str | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    target_root = (Path(path).expanduser().resolve() / (root_name or _repo_dir_name(repo))).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(target_root),
        allow_patterns=allow_patterns,
    )
    return target_root


def download_git_repo(
    path: str | Path,
    repo_url: str,
    *,
    revision: str = "HEAD",
    root_name: str | None = None,
    update_submodules: bool = False,
) -> Path:
    target_root = (Path(path).expanduser().resolve() / (root_name or _repo_dir_name(repo_url))).resolve()
    if not target_root.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_root)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    subprocess.run(
        ["git", "-C", str(target_root), "fetch", "--depth", "1", "origin", revision],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(target_root), "checkout", revision],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if update_submodules:
        subprocess.run(
            ["git", "-C", str(target_root), "submodule", "update", "--init", "--recursive"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    return target_root


__all__ = [
    "UrlDownloadFile",
    "download_git_repo",
    "download_hf_repo",
    "download_url_files",
]
