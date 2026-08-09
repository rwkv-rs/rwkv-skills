#!/usr/bin/env python3
"""Create and verify the independent strict-46 per-run state directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import stat


STATE_PARENT = Path("/var/lib/rwkv-strict46")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
STATE_SUBDIRECTORIES = (
    "logs",
    "logs/scheduler",
    "logs/pids",
    "logs/runs",
    "logs/audits",
    "locks",
    "handoff-requests",
)


class StateError(RuntimeError):
    pass


def _require_trusted_parent_ancestors(parent: Path) -> None:
    for ancestor in (parent.parent, *parent.parent.parents):
        if ancestor.is_symlink():
            raise StateError(f"state ancestor must not be a symlink: {ancestor}")
        status = os.lstat(ancestor)
        if not stat.S_ISDIR(status.st_mode) or status.st_uid != 0:
            raise StateError(f"state ancestor is not a root-owned directory: {ancestor}")
        if status.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise StateError(f"state ancestor is group/other writable: {ancestor}")


def _lstat_directory(path: Path, *, owner_uid: int, mode: int | None = None) -> os.stat_result:
    if path.is_symlink():
        raise StateError(f"state path must not be a symlink: {path}")
    try:
        status = os.lstat(path)
    except FileNotFoundError as exc:
        raise StateError(f"state directory is missing: {path}") from exc
    if not stat.S_ISDIR(status.st_mode):
        raise StateError(f"state path is not a directory: {path}")
    if status.st_uid != owner_uid:
        raise StateError(
            f"state directory owner mismatch: {path}: {status.st_uid} != {owner_uid}"
        )
    actual_mode = stat.S_IMODE(status.st_mode)
    if actual_mode & 0o077:
        raise StateError(f"state directory must be private: {path}: {actual_mode:o}")
    if mode is not None and actual_mode != mode:
        raise StateError(f"state directory mode mismatch: {path}: {actual_mode:o} != {mode:o}")
    return status


def prepare_run_state(run_id: str, *, create: bool) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise StateError(f"unsafe strict-46 run id: {run_id!r}")
    owner_uid = os.geteuid()
    if owner_uid == 0:
        raise StateError("strict-46 run state must be owned by the unprivileged runner")

    parent = STATE_PARENT
    _require_trusted_parent_ancestors(parent)
    parent_status = _lstat_directory(parent, owner_uid=owner_uid)
    parent_device = parent_status.st_dev
    run_root = parent / run_id
    if create and not run_root.exists() and not run_root.is_symlink():
        os.mkdir(run_root, 0o700)
    root_status = _lstat_directory(run_root, owner_uid=owner_uid, mode=0o700)
    if root_status.st_dev != parent_device:
        raise StateError("strict-46 run state must stay on the provisioned filesystem")

    for relative in STATE_SUBDIRECTORIES:
        path = run_root / relative
        if create and not path.exists() and not path.is_symlink():
            path.mkdir(mode=0o700)
        status = _lstat_directory(path, owner_uid=owner_uid, mode=0o700)
        if status.st_dev != parent_device:
            raise StateError(f"cross-filesystem state directory refused: {path}")
    return run_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    try:
        print(prepare_run_state(args.run_id, create=args.create))
    except (OSError, StateError) as exc:
        print(f"strict-46 state refused: {exc}", file=os.sys.stderr)
        return 42
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
