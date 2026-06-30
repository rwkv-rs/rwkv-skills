from __future__ import annotations

"""Status, stop, and logs actions backed by the scheduler library."""

import sys
from pathlib import Path

from . import actions_base as base
from .actions_base import LogsOptions, RunningEntry, StatusOptions, StopOptions


def action_status(opts: StatusOptions) -> dict[str, RunningEntry]:
    running = base.load_running(opts.pid_dir)
    if not running:
        print("🟡 无运行任务")
        return running
    header = f"{'Job ID':<32} {'GPU':<6} PID"
    print(header)
    print("-" * len(header))
    for job_id, entry in sorted(running.items()):
        gpu = entry.gpu or "?"
        print(f"{job_id:<32} {gpu:<6} {entry.pid}")
    return running


def action_stop(opts: StopOptions) -> None:
    pid_dir = opts.pid_dir
    if opts.stop_all:
        running = base.load_running(pid_dir)
        if not running:
            print("ℹ️  无运行任务")
            return
        for job_id in sorted(running.keys()):
            base.stop_job(job_id, pid_dir)
        return

    if not opts.job_ids:
        print("请指定 job id，或使用 --all")
        return
    for job_id in opts.job_ids:
        base.stop_job(job_id, pid_dir)


def action_logs(opts: LogsOptions) -> None:
    run_log_dir = opts.run_log_dir
    pid_dir = opts.pid_dir
    if not run_log_dir.exists():
        print(f"Log 目录 {run_log_dir} 不存在")
        return
    running = base.load_running(pid_dir)
    if not running:
        print("当前没有运行中的任务；logs 仅展示活跃任务。")
        return

    display_items: list[tuple[str, Path]] = []
    for job_id in sorted(running.keys()):
        entry = running[job_id]
        log_path = entry.log_path
        if log_path is None:
            log_path = run_log_dir / f"{job_id}.log"
        elif not log_path.is_absolute():
            log_path = run_log_dir / log_path
        display_items.append((job_id, log_path))

    if not display_items:
        print("当前没有运行中的任务；logs 仅展示活跃任务。")
        return

    rotate_seconds = max(1, opts.rotate_seconds)
    alt_screen = "\033[?1049h"
    restore_screen = "\033[?1049l"
    clear_screen = "\033[2J\033[H"
    hide_cursor = "\033[?25l"
    show_cursor = "\033[?25h"

    try:
        if not base._write_stdout(alt_screen + hide_cursor):
            raise RuntimeError("stdout write failed")
        sys.stdout.flush()
        while True:
            for job_id, log_path in display_items:
                if not base._write_stdout(clear_screen):
                    raise RuntimeError("stdout write failed")
                sys.stdout.flush()
                lines = base.tail_file(log_path, opts.tail_lines)
                print(f"===> {job_id} | {log_path}")
                print("\n".join(lines))
                base.time.sleep(rotate_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        base._write_stdout(restore_screen + show_cursor)


__all__ = [
    "action_status",
    "action_stop",
    "action_logs",
]
