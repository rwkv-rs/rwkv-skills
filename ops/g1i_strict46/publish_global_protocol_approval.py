#!/usr/bin/env python3
"""Atomically publish a content-addressed strict-46 global approval.

This tool never runs an audit and cannot turn a failed merge into a PASS.  It
only consumes the immutable acceptance emitted by the final global merge plus
content-addressed Judge protocol evidence, validates both, and publishes a new
read-only artifact without overwriting anything.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

_BOOTSTRAP_REPO = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_REPO) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO))

from ops.g1i_strict46.require_global_protocol_gate import (  # noqa: E402
    APPROVAL_DIRECTORY,
    ProtocolGateError,
    _canonical_json_bytes,
    _path_descriptor,
    _read_stable_bytes,
    _verify_global_approval,
    build_global_approval_payload,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--merge-acceptance", type=Path, required=True)
    parser.add_argument("--judge-evidence", type=Path, required=True)
    parser.add_argument("--runtime-evidence", type=Path, required=True)
    return parser.parse_args()


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise ProtocolGateError(
            f"global approval directory must not be a symlink: {path.parent}"
        )
    if path.exists():
        raise FileExistsError(f"refusing to overwrite global approval: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = _parse_args()
    repo = args.repo.expanduser().resolve()
    try:
        evidence_path = args.judge_evidence.expanduser()
        evidence = json.loads(_read_stable_bytes(evidence_path).decode("utf-8"))
        if not isinstance(evidence, dict):
            raise ProtocolGateError("Judge evidence root must be an object")
        runtime_evidence = json.loads(
            _read_stable_bytes(args.runtime_evidence.expanduser()).decode("utf-8")
        )
        if not isinstance(runtime_evidence, dict):
            raise ProtocolGateError("runtime evidence root must be an object")
        approval = build_global_approval_payload(
            repo,
            merge_acceptance_path=args.merge_acceptance.expanduser(),
            judge_protocol_evidence=evidence,
            runtime_attestation_evidence=runtime_evidence,
        )
        approval_sha = str(approval["approval_sha256"])
        output = repo / APPROVAL_DIRECTORY / f"{approval_sha}.acceptance.json"
        _publish_once(output, _canonical_json_bytes(approval))
        _verify_global_approval(
            repo,
            output,
            locked_descriptor=_path_descriptor(repo, output),
        )
    except (OSError, ValueError, ProtocolGateError) as exc:
        print(f"global approval publication failed: {exc}", file=sys.stderr)
        return 42
    print(json.dumps(_path_descriptor(repo, output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
