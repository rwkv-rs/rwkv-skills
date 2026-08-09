#!/usr/bin/env python3
"""Emit an unprivileged, content-addressed request for a root handoff service.

This program never invokes systemd, SSH, an inference engine, or a scheduler.
The root orchestrator must independently verify the frozen manifest and the
fixed transition before consuming a request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from ops.g1i_strict46 import require_global_protocol_gate as gate
from ops.g1i_strict46.runtime_state import prepare_run_state


REQUEST_SCHEMA = "rwkv.g1i-strict46-handoff-request.v1"
TRANSITIONS = {
    "157-2p9-to-1p5": {
        "host": "157",
        "gpu": 3,
        "port": 19439,
        "current_model": "rwkv7-g1i-2.9b-20260805-ctx16384",
        "next_model": "rwkv7-g1i-1.5b-20260805-ctx16384",
        "current_inference_unit": "rwkv-g1i-2p9b-gpu3-16k-c640.service",
        "next_inference_unit": "rwkv-g1i-1p5b-gpu3-16k-c640.service",
        "next_scheduler_unit": "rwkv-g1i-strict46-15-raw-20260806.service",
    },
    "8222-7p2-to-13p3": {
        "host": "8222",
        "gpu": 2,
        "port": 18074,
        "forwarded_host": "157",
        "forwarded_port": 29574,
        "current_model": "rwkv7-g1i-7.2b-20260805-ctx16384",
        "next_model": "rwkv7-g1i-13.3b-20260805-ctx16384",
        "current_inference_unit": "rwkv-g1i-7p2b-gpu2-16k-c640.service",
        "next_inference_unit": "rwkv-g1i-13p3b-gpu2-16k-c640.service",
        "next_scheduler_unit": "rwkv-g1i-strict46-133-raw-20260806.service",
    },
}


def _stable_sha(path: Path) -> str:
    return hashlib.sha256(gate._read_stable_bytes(path)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition", choices=sorted(TRANSITIONS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--frozen-runtime", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()

    try:
        state_root = prepare_run_state(args.run_id, create=False)
        configured_state = os.environ.get("RWKV_STRICT_STATE_ROOT", "").strip()
        if not configured_state or Path(configured_state).resolve(strict=True) != state_root:
            raise gate.ProtocolGateError(
                "RWKV_STRICT_STATE_ROOT does not match the verified run state"
            )
        runtime = args.frozen_runtime.expanduser().resolve(strict=True)
        manifest_path = runtime / gate.FROZEN_RUNTIME_MANIFEST
        manifest = json.loads(gate._read_stable_bytes(manifest_path).decode("utf-8"))
        manifest_sha = str(manifest.get("manifest_sha256") or "")
        if not gate._is_sha256(manifest_sha) or runtime.name != manifest_sha:
            raise gate.ProtocolGateError("frozen runtime is not content-addressed")
        approval = args.approval.expanduser().resolve(strict=True)
        lock = runtime / "ops/g1i_strict46/protocol_gate.lock.json"
        transition = dict(TRANSITIONS[args.transition])
        payload: dict[str, object] = {
            "schema_version": REQUEST_SCHEMA,
            "transition_id": args.transition,
            "transition": transition,
            "run_id": args.run_id,
            "state_root": str(state_root),
            "frozen_runtime": str(runtime),
            "frozen_manifest_sha256": manifest_sha,
            "approval_sha256": _stable_sha(approval),
            "protocol_lock_sha256": _stable_sha(lock),
            "requested_by_uid": os.geteuid(),
            "requested_at_unix_ns": time.time_ns(),
        }
        unsigned = gate._canonical_json_bytes(payload)
        request_sha = hashlib.sha256(unsigned).hexdigest()
        payload["request_sha256"] = request_sha
        destination = state_root / "handoff-requests" / f"{request_sha}.json"
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(gate._canonical_json_bytes(payload))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    except (OSError, ValueError, json.JSONDecodeError, gate.ProtocolGateError) as exc:
        print(f"handoff request refused: {exc}", file=os.sys.stderr)
        return 42

    print(destination)
    # EX_TEMPFAIL: request is durable, but no privileged action occurred.
    return 75


if __name__ == "__main__":
    raise SystemExit(main())
