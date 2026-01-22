from __future__ import annotations

"""Process orchestration utilities for scheduler."""

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .perf import perf_logger
from .state import stop_all_jobs


def log_job_event(event: str, job_id: str, **payload: object) -> None:
    if perf_logger.enabled:
        perf_logger.log(event, job_id=job_id, **payload)


@dataclass
class JobFailure:
    job_id: str
    returncode: int
    log_path: Path
    command: tuple[str, ...]


class FailureMonitor:
    def __init__(self) -> None:
        self._queue: queue.Queue[JobFailure] = queue.Queue()
        self._aborting: threading.Event = threading.Event()

    def reset(self) -> None:
        self._aborting.clear()
        self._drain()

    def mark_aborting(self) -> None:
        self._aborting.set()
        self._drain()

    def watch(
        self,
        job_id: str,
        process: subprocess.Popen[bytes],
        log_path: Path,
        command: Sequence[str],
    ) -> None:
        thread = threading.Thread(
            target=self._wait_for_process,
            args=(job_id, process, log_path, tuple(command)),
            daemon=True,
        )
        thread.start()

    def _wait_for_process(
        self,
        job_id: str,
        process: subprocess.Popen[bytes],
        log_path: Path,
        command: tuple[str, ...],
    ) -> None:
        returncode = process.wait()
        if returncode == 0 or self._aborting.is_set():
            return
        self._queue.put(JobFailure(job_id=job_id, returncode=returncode, log_path=log_path, command=command))

    def wait_failure(self, timeout: float | None = None) -> JobFailure | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _drain(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:  # pragma: no cover - race guard
                break


FAILURE_MONITOR = FailureMonitor()


def launch_job(
    job_id: str,
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb", buffering=0) as stream:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=stream,
            stderr=stream,
            env=dict(env) if env is not None else None,
        )
    FAILURE_MONITOR.watch(job_id, process, log_path, command)
    return process


def handle_job_failure(
    failure: JobFailure,
    pid_dir: Path,
    job_metadata: dict[str, dict[str, object]] | None = None,
    launch_times: dict[str, float] | None = None,
) -> None:
    FAILURE_MONITOR.mark_aborting()
    print(f"❌ 任务 {failure.job_id} 异常退出 (returncode={failure.returncode})", flush=True)
    print(f"   命令: {' '.join(failure.command)}", flush=True)
    print("   停止所有正在运行的任务…", flush=True)
    stop_all_jobs(pid_dir)
    meta = job_metadata.pop(failure.job_id, {}) if job_metadata is not None else {}
    if "log_path" in meta:
        meta = dict(meta)
        meta.pop("log_path", None)
    runtime = None
    if launch_times is not None:
        start = launch_times.pop(failure.job_id, None)
        if start is not None:
            runtime = time.time() - start
    log_job_event(
        "job_fail",
        failure.job_id,
        returncode=failure.returncode,
        log_path=str(failure.log_path),
        runtime_s=runtime,
        **meta,
    )
    if failure.log_path.exists():
        try:
            log_text = failure.log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"⚠️  读取日志失败: {exc}", flush=True)
        else:
            separator = "=" * 80
            print(separator)
            print(f"[错误日志] {failure.log_path}")
            print(separator)
            print(log_text.rstrip("\n"))
            print(separator)
    else:
        print(f"⚠️  日志文件 {failure.log_path} 不存在", flush=True)


def list_idle_gpus(max_mem_mb: int) -> list[str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        print("⚠️  nvidia-smi 未找到，视为无空闲 GPU")
        return []

    if proc.returncode != 0:
        print("⚠️  nvidia-smi 调用失败，视为无空闲 GPU")
        return []

    idle: list[str] = []
    for line in proc.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        idx, util, mem = parts[:3]
        try:
            util_val = float(util)
            mem_val = float(mem)
        except ValueError:
            continue
        if util_val == 0.0 and mem_val < max_mem_mb:
            idle.append(idx)
    return idle


__all__ = [
    "FAILURE_MONITOR",
    "FailureMonitor",
    "JobFailure",
    "handle_job_failure",
    "launch_job",
    "list_idle_gpus",
    "log_job_event",
]
