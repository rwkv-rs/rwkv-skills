from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence


def _strip_remainder_prefix(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def _default_command() -> list[str]:
    return [sys.executable, "-m", "src.space.app"]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Gradio space under a supervisor and restart it periodically."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:7860/",
        help="Readiness URL for each launch (default: http://127.0.0.1:7860/).",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for a launch to become ready.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between readiness probes.",
    )
    parser.add_argument(
        "--grace-seconds",
        type=float,
        default=10.0,
        help="Seconds to wait after SIGTERM before SIGKILL.",
    )
    parser.add_argument(
        "--restart-pause",
        type=float,
        default=1.0,
        help="Seconds to wait between stop and relaunch.",
    )
    parser.add_argument(
        "--restart-interval",
        type=float,
        default=3600.0,
        help="Seconds between automatic restarts after a launch becomes ready (default: 3600). Set <= 0 to disable periodic restarts.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Optional custom command. Example: -- uv run rwkv-skills-space",
    )
    return parser.parse_args(argv)


def _start_process(command: list[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(command, start_new_session=True)


def _wait_until_ready(
    url: str,
    timeout: float,
    poll_interval: float,
    *,
    proc: subprocess.Popen[bytes] | None = None,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=5.0) as response:
                status = getattr(response, "status", None)
                if status is None or 200 <= int(status) < 500:
                    return True
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(max(poll_interval, 0.1))
    return False


def _terminate_process(proc: subprocess.Popen[bytes], grace_seconds: float) -> int:
    if proc.poll() is not None:
        return int(proc.returncode or 0)

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return int(proc.poll() or 0)

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return int(proc.returncode or 0)
        time.sleep(0.1)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return int(proc.wait())


def _sleep_with_interrupts(duration: float) -> None:
    deadline = time.monotonic() + max(duration, 0.0)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.25))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    command = _strip_remainder_prefix(list(args.command)) or _default_command()

    active_proc: subprocess.Popen[bytes] | None = None
    stop_signum: int | None = None

    def _forward(signum: int, _frame: object) -> None:
        nonlocal stop_signum, active_proc
        stop_signum = signum
        if active_proc is None or active_proc.poll() is not None:
            return
        try:
            os.killpg(active_proc.pid, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    launch_index = 0
    while True:
        if stop_signum is not None:
            return 128 + stop_signum
        launch_index += 1
        print(f"[space-restart] launch #{launch_index}: {' '.join(command)}", flush=True)
        active_proc = _start_process(command)

        ready = _wait_until_ready(
            args.url,
            args.ready_timeout,
            args.poll_interval,
            proc=active_proc,
        )
        if not ready:
            print(
                f"[space-restart] launch #{launch_index} did not become ready within {args.ready_timeout:.1f}s",
                flush=True,
            )
            code = _terminate_process(active_proc, args.grace_seconds)
            active_proc = None
            if stop_signum is not None:
                return 128 + stop_signum
            return code if code != 0 else 1

        print(
            f"[space-restart] launch #{launch_index} is ready"
            + (
                f"; next restart in {args.restart_interval:.0f}s"
                if args.restart_interval > 0
                else "; periodic restart disabled"
            ),
            flush=True,
        )

        if args.restart_interval <= 0:
            code = int(active_proc.wait())
            active_proc = None
            if stop_signum is not None:
                return 128 + stop_signum
            return code

        deadline = time.monotonic() + args.restart_interval
        while True:
            if stop_signum is not None:
                code = _terminate_process(active_proc, args.grace_seconds)
                active_proc = None
                return 128 + stop_signum if stop_signum is not None else code
            if active_proc.poll() is not None:
                code = int(active_proc.returncode or 0)
                active_proc = None
                return code
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(max(args.poll_interval, 0.1), remaining, 1.0))

        print(f"[space-restart] restarting launch #{launch_index}", flush=True)
        _terminate_process(active_proc, args.grace_seconds)
        active_proc = None
        if stop_signum is not None:
            return 128 + stop_signum
        if args.restart_pause > 0:
            _sleep_with_interrupts(args.restart_pause)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
