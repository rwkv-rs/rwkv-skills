from __future__ import annotations

"""Utility to fetch RWKV weights from Hugging Face mirrors with retries."""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
DEFAULT_TIMEOUT = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "900")

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_ENDPOINT", DEFAULT_ENDPOINT)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", DEFAULT_TIMEOUT)
os.environ.setdefault("DATASETS_HTTP_TIMEOUT", DEFAULT_TIMEOUT)

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError


DownloadSpec = tuple[str, Sequence[str] | str]


@dataclass(frozen=True)
class DownloadTarget:
    repo_id: str
    files: tuple[str, ...]

    @classmethod
    def from_item(cls, item: DownloadSpec) -> "DownloadTarget":
        repo_id, filenames = item
        if isinstance(filenames, str):
            files = (filenames,)
        else:
            files = tuple(filenames)
        return cls(repo_id=repo_id, files=files)

    def iter_tasks(self) -> Iterable[tuple[str, str]]:
        for fname in self.files:
            yield self.repo_id, fname


STATIC_FILES: list[DownloadSpec] = []
PTH_REPOS: tuple[str, ...] = ("BlinkDL/rwkv7-g1",)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path(os.environ.get("RWKV_WEIGHTS_DIR", REPO_ROOT / "weights"))
DEFAULT_REVISION = os.environ.get("RWKV_WEIGHTS_REVISION", "main")
MAX_AUTO_WORKERS = 8
INITIAL_BACKOFF_SECONDS = 5
MAX_BACKOFF_SECONDS = 300
# PTH_FILENAME_KEYWORD = "g1d"


def discover_pth_files(api: HfApi, repo_id: str, revision: str = DEFAULT_REVISION) -> tuple[str, ...]:
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 无法获取 {repo_id} 的 .pth 列表：{exc}")
        return ()

    pth_files = tuple(
        sorted(
            fname
            for fname in repo_files
            # if fname.endswith(".pth") and PTH_FILENAME_KEYWORD in Path(fname).name.lower()
        )
    )
    if not pth_files:
        print(f"⚠️  未在 {repo_id} 找到 .pth 文件")
        return ()

    print(f"🔍  {repo_id} 发现 {len(pth_files)} 个 .pth 文件")
    return pth_files


def build_download_targets(api: HfApi) -> list[DownloadTarget]:
    targets = [DownloadTarget.from_item(item) for item in STATIC_FILES]
    for repo_id in PTH_REPOS:
        files = discover_pth_files(api, repo_id)
        if files:
            targets.append(DownloadTarget(repo_id=repo_id, files=files))
    return targets


def download_one(repo_id: str, filename: str, out_dir: Path) -> Path:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        endpoint=DEFAULT_ENDPOINT,
        revision=DEFAULT_REVISION,
        local_dir=str(out_dir / repo_id.replace("/", "__")),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return Path(local_path)


def download_with_retry(repo_id: str, filename: str, out_dir: Path) -> Path:
    attempt = 1
    delay = INITIAL_BACKOFF_SECONDS

    while True:
        try:
            return download_one(repo_id, filename, out_dir)
        except KeyboardInterrupt:  # pragma: no cover - propagate interrupt
            raise
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
            wait = min(delay, MAX_BACKOFF_SECONDS)
            print(f"⚠️  重试：{repo_id}/{filename} (第 {attempt} 次失败，{wait}s 后重试) | {exc}")
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001
            wait = min(delay, MAX_BACKOFF_SECONDS)
            print(
                f"⚠️  重试：{repo_id}/{filename} (第 {attempt} 次失败，{wait}s 后重试) | {type(exc).__name__}: {exc}"
            )
            time.sleep(wait)
        attempt += 1
        delay = min(delay * 2, MAX_BACKOFF_SECONDS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 Hugging Face 权重到本地。")
    parser.add_argument("out_dir", nargs="?", default=str(DEFAULT_OUT_DIR), help="保存目录")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        help="并发线程数（默认自动按任务数和 CPU 核心估算）",
    )
    parser.add_argument(
        "--repo",
        action="append",
        help="额外需要抓取的仓库（可重复，多用于 PTH 仓库）",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_repos = tuple(args.repo or ())
    api = HfApi(endpoint=DEFAULT_ENDPOINT)
    targets = build_download_targets(api)
    for repo in extra_repos:
        files = discover_pth_files(api, repo)
        if files:
            targets.append(DownloadTarget(repo_id=repo, files=files))

    if not targets:
        print("⚠️  没有任何下载任务，退出。")
        return 0

    total_tasks = sum(len(target.files) for target in targets)
    default_workers = min(max(1, (os.cpu_count() or 4) * 2), MAX_AUTO_WORKERS, total_tasks or 1)
    max_workers = args.workers or default_workers

    print(f"保存目录：{out_dir}")
    print(f"总任务数：{total_tasks} | 并发线程数：{max_workers}")

    results: dict[tuple[str, str], Path] = {}
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(download_with_retry, repo, fname, out_dir): (repo, fname)
                for target in targets
                for repo, fname in target.iter_tasks()
            }
            for future in as_completed(future_map):
                repo, fname = future_map[future]
                path = future.result()
                size_gb = path.stat().st_size / (1024**3)
                print(f"✅ 已下载：{repo}/{fname} -> {path}  ({size_gb:.2f} GB)")
                results[(repo, fname)] = path
    except KeyboardInterrupt:
        print("\n用户中断，正在退出…")
        return 130

    print("\n=== 下载完成 ===")
    for target in targets:
        print(f"{target.repo_id}:")
        for fname in target.files:
            path = results.get((target.repo_id, fname))
            if path is None:
                print(f"  - {fname}: 未知状态（可能被中断）")
            else:
                print(f"  - {fname}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
