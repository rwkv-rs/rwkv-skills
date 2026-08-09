from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from ops.g1i_strict46 import require_global_protocol_gate as gate
from ops.g1i_strict46 import runtime_attestation as runtime
from src.eval.scheduler import action_dispatch as scheduler_dispatch
from src.eval.scheduler.actions_base import DispatchOptions, InferenceConfig
from tests.test_g1i_runtime_attestation import _forward_fixture, _runtime_fixture
from tests.test_global_protocol_gate import (
    _approval_fixture,
    _runtime_evidence_fixture,
)


ROOT = Path(__file__).resolve().parents[1]


def _reseal_evidence(evidence: dict[str, object]) -> None:
    evidence.pop("evidence_sha256", None)
    evidence["evidence_sha256"] = gate._canonical_json_sha256(evidence)


@pytest.mark.parametrize("phase", ["dispatch", "recovery"])
def test_dispatch_and_recovery_fail_closed_without_runtime_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    """A valid protocol tree and a plausible /models name are insufficient.

    This intentionally stubs every non-runtime gate so the assertion cannot be
    satisfied accidentally by an unrelated approval or frozen-tree failure.
    Both direct dispatch and recovery must independently require a live
    listener-to-weight runtime proof.
    """

    lock = {
        "lock_sha256": "1" * 64,
        "global_approval": {},
        "runtime_attestation_evidence_sha256": "2" * 64,
    }
    approval = {"runtime_attestation_evidence": {"models": {}}}
    runtime_calls: list[dict[str, object]] = []

    def reject_missing_runtime(**kwargs):
        runtime_calls.append(kwargs)
        raise gate.ProtocolGateError("runtime attestation proof is unavailable")

    monkeypatch.setattr(gate, "_verify_strict_scope", lambda _repo: None)
    monkeypatch.setattr(gate, "_verify_protocol_invariants", lambda _repo: None)
    monkeypatch.setattr(gate, "_verify_lock", lambda _repo, _lock: lock)
    monkeypatch.setattr(
        gate,
        "verify_frozen_runtime",
        lambda **_kwargs: {
            "root": tmp_path,
            "python_executable": Path("/usr/bin/python3"),
        },
    )
    monkeypatch.setattr(
        gate,
        "_verify_global_approval",
        lambda *_a, **_kw: approval,
    )
    monkeypatch.setattr(gate, "verify_inference_endpoint", lambda *_a, **_kw: None)

    with pytest.raises(gate.ProtocolGateError, match="runtime attestation"):
        gate.require_gate(
            repo=tmp_path,
            lock_path=tmp_path / "protocol_gate.lock.json",
            phase=phase,
            model="rwkv7-g1i-1.5b-20260805-ctx16384",
            approval_path=tmp_path / "approval.json",
            infer_base_url="http://127.0.0.1:19439/v1",
            infer_api_key="rwkv-skills",
            frozen_runtime=tmp_path,
            runtime_route_verifier=reject_missing_runtime,
        )
    assert len(runtime_calls) == 1
    assert runtime_calls[0]["model"] == "rwkv7-g1i-1.5b-20260805-ctx16384"
    assert runtime_calls[0]["infer_base_url"] == "http://127.0.0.1:19439/v1"


def _strict_dispatch_options(tmp_path: Path) -> DispatchOptions:
    model = "rwkv7-g1i-13.3b-20260805-ctx16384"
    return DispatchOptions(
        log_dir=tmp_path / "logs",
        pid_dir=tmp_path / "pids",
        run_log_dir=tmp_path / "runs",
        job_order=("multi_choice_plain_naive",),
        inference=InferenceConfig(
            base_url="http://127.0.0.1:29574/v1",
            models=tuple(f"slot-{index}={model}" for index in range(2)),
            api_key="rwkv-skills",
        ),
    )


def test_direct_scheduler_dispatch_runs_attestation_before_any_state_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling action_dispatch directly is not an operator bypass."""

    opts = _strict_dispatch_options(tmp_path)
    monkeypatch.delenv("RWKV_STRICT_FROZEN_RUNTIME", raising=False)
    monkeypatch.delenv("RWKV_GLOBAL_PROTOCOL_APPROVAL", raising=False)
    wrote_state = False

    def forbidden_write(*_args, **_kwargs):
        nonlocal wrote_state
        wrote_state = True

    monkeypatch.setattr(scheduler_dispatch.base, "ensure_dirs", forbidden_write)
    with pytest.raises(
        RuntimeError,
        match="requires frozen runtime and global approval",
    ):
        scheduler_dispatch.action_dispatch(opts)
    assert wrote_state is False
    assert not opts.log_dir.exists()
    assert not opts.pid_dir.exists()
    assert not opts.run_log_dir.exists()


def test_direct_scheduler_dispatch_invokes_exact_frozen_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opts = _strict_dispatch_options(ROOT)
    approval = ROOT / "approval.acceptance.json"
    monkeypatch.setenv("RWKV_STRICT_FROZEN_RUNTIME", str(ROOT))
    monkeypatch.setenv("RWKV_GLOBAL_PROTOCOL_APPROVAL", str(approval))
    captured: list[tuple[list[str], dict[str, object]]] = []
    expected_provenance = {"verified": True}

    def fake_run(command, **kwargs):
        captured.append((list(command), dict(kwargs)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scheduler_dispatch.subprocess, "run", fake_run)
    monkeypatch.setattr(
        scheduler_dispatch,
        "_runtime_provenance_from_approval",
        lambda **_kwargs: expected_provenance,
    )
    result = scheduler_dispatch.require_strict_g1i_runtime_attestation(opts)

    assert len(captured) == 1
    command, kwargs = captured[0]
    assert command[1] == "-I"
    assert command[command.index("--phase") + 1] == "dispatch"
    assert command[command.index("--model") + 1] == (
        "rwkv7-g1i-13.3b-20260805-ctx16384"
    )
    assert command[command.index("--infer-base-url") + 1] == (
        "http://127.0.0.1:29574/v1"
    )
    assert "--require-current-python" in command
    assert kwargs["check"] is True
    assert result is expected_provenance


@pytest.mark.parametrize(
    ("model", "expected_route"),
    [
        ("rwkv7-g1i-1.5b-20260805-ctx16384", "local"),
        ("rwkv7-g1i-13.3b-20260805-ctx16384", "ssh_forward"),
    ],
)
def test_task_provenance_binds_approval_lock_weight_engine_and_route(
    tmp_path: Path,
    model: str,
    expected_route: str,
) -> None:
    evidence = _runtime_evidence_fixture()
    approval: dict[str, object] = {
        "runtime_attestation_evidence": evidence,
    }
    approval["approval_sha256"] = gate._canonical_json_sha256(approval)
    approval_path = tmp_path / "approval.json"
    approval_path.write_bytes(gate._canonical_json_bytes(approval))
    lock: dict[str, object] = {
        "global_approval": {
            "path": str(approval_path),
            "sha256": gate._sha256_file(approval_path),
        },
        "runtime_attestation_evidence_sha256": evidence["evidence_sha256"],
    }
    lock["lock_sha256"] = gate._canonical_json_sha256(lock)
    lock_path = tmp_path / "lock.json"
    lock_path.write_bytes(gate._canonical_json_bytes(lock))
    endpoint = evidence["models"][model]["route"]["scheduler_endpoint"]
    base_url = (
        f"{endpoint['scheme']}://{endpoint['host']}:{endpoint['port']}"
        f"{endpoint['api_prefix']}"
    )

    provenance = scheduler_dispatch._runtime_provenance_from_approval(
        approval_path=approval_path,
        lock_path=lock_path,
        model=model,
        infer_base_url=base_url,
    )

    runtime_artifact = evidence["models"][model]["runtime_attestation"]
    assert provenance["model"] == model
    assert provenance["route_kind"] == expected_route
    assert provenance["runtime_attestation_evidence_sha256"] == (
        evidence["evidence_sha256"]
    )
    assert provenance["runtime_attestation_artifact_sha256"] == (
        runtime_artifact["artifact_sha256"]
    )
    assert provenance["weight"] == runtime_artifact["model"]["weight"]
    assert provenance["runtime_tree_sha256"] == (
        runtime_artifact["runtime_tree"]["tree_sha256"]
    )
    if expected_route == "local":
        assert provenance["forward_attestation_artifact_sha256"] is None
    else:
        assert provenance["forward_attestation_artifact_sha256"] == (
            evidence["models"][model]["route"]["forward_attestation"][
                "artifact_sha256"
            ]
        )


def test_official_dispatch_and_recovery_chains_carry_runtime_proof() -> None:
    """All official scheduler entry points must preserve the proof contract."""

    root = Path(__file__).resolve().parents[1]
    ops = root / "ops" / "g1i_strict46"
    run_model = (ops / "run_model.sh").read_text(encoding="utf-8")
    recovery = (ops / "ensure_model_complete.sh").read_text(encoding="utf-8")
    recovery_runner = (ops / "run_audit_missing.py").read_text(encoding="utf-8")
    waiter_157 = (ops / "wait_157_1p5.sh").read_text(encoding="utf-8")
    waiter_8222 = (ops / "wait_8222_13p3.sh").read_text(encoding="utf-8")

    # Main dispatch gates after the presentation-level /models probe and once
    # more immediately before scheduler exec.  The behavioral test above
    # proves that this phase cannot pass without live runtime attestation.
    final_gate = run_model.rindex("--phase dispatch")
    scheduler_exec = run_model.index("src.eval.scheduler.cli dispatch")
    assert final_gate < scheduler_exec
    assert run_model.count("--require-current-python") >= 1

    # Recovery cannot call the scheduler directly from ensure_model_complete;
    # its frozen runner invokes the mandatory recovery phase immediately before
    # constructing and executing the scheduler command.
    assert '"$runtime_repo/ops/g1i_strict46/run_audit_missing.py"' in recovery
    recovery_gate = recovery_runner.index('phase="recovery"')
    recovery_dispatch = recovery_runner.index('"src.eval.scheduler.cli"')
    assert recovery_gate < recovery_dispatch

    # Unprivileged handoff paths only emit a content-addressed request.  They
    # cannot launch a model or scheduler and cannot bypass the root consumer.
    for waiter in (waiter_157, waiter_8222):
        assert "ops.g1i_strict46.handoff_request" in waiter
        assert "src.eval.scheduler.cli" not in waiter
        assert "run_model.sh" not in waiter
        assert "systemctl" not in waiter
        assert "systemd-run" not in waiter


def test_handoff_and_recovery_attest_at_the_irreversible_boundaries() -> None:
    """A stale pre-launch proof must never authorize the replacement runtime."""

    ops = ROOT / "ops" / "g1i_strict46"
    recovery = (ops / "ensure_model_complete.sh").read_text(encoding="utf-8")
    recovery_runner = (ops / "run_audit_missing.py").read_text(encoding="utf-8")
    waiter_157 = (ops / "wait_157_1p5.sh").read_text(encoding="utf-8")
    waiter_8222 = (ops / "wait_8222_13p3.sh").read_text(encoding="utf-8")

    # Every recovery round attests before spawning its wrapper, and that
    # wrapper independently re-attests immediately before scheduler dispatch.
    recovery_loop = recovery.index("for recovery_attempt in")
    recovery_attest = recovery.index("require_runtime_attestation", recovery_loop)
    recovery_spawn = recovery.index(
        '"$python" "$runtime_repo/ops/g1i_strict46/run_audit_missing.py"',
        recovery_attest,
    )
    assert recovery_loop < recovery_attest < recovery_spawn
    recovery_gate = recovery_runner.index('phase="recovery"')
    recovery_scheduler = recovery_runner.index('"src.eval.scheduler.cli"')
    assert recovery_gate < recovery_scheduler

    # Local handoff: audit/recovery and old-runtime attestation precede the
    # next-model approval and the durable request.  No destructive boundary is
    # crossed in the unprivileged process.
    local_audit = waiter_157.index("--phase audit")
    local_recovery = waiter_157.index("ensure_model_complete.sh")
    local_idle = waiter_157.index("handoff_idle_guard.sh")
    local_attest = waiter_157.index("--phase attest", local_idle)
    local_launch = waiter_157.index("--phase launch", local_attest)
    local_request = waiter_157.index("ops.g1i_strict46.handoff_request")
    assert local_audit < local_recovery < local_idle < local_attest < local_launch < local_request

    # 8222 proves the local current runtime and idle state before requesting a
    # fixed transition.  The root consumer must independently prove the 157
    # forward and database audit; this script cannot claim those proofs.
    remote_attest = waiter_8222.index("--phase attest")
    remote_idle = waiter_8222.index("handoff_idle_guard.sh", remote_attest)
    remote_launch = waiter_8222.index("--phase launch", remote_idle)
    remote_request = waiter_8222.index("ops.g1i_strict46.handoff_request")
    assert remote_attest < remote_idle < remote_launch < remote_request


def test_8222_unprivileged_handoff_has_no_ssh_or_service_control() -> None:
    """Only the privileged consumer may cross hosts or control services."""

    waiter = (
        ROOT / "ops" / "g1i_strict46" / "wait_8222_13p3.sh"
    ).read_text(encoding="utf-8")

    assert "/usr/bin/ssh" not in waiter
    assert "remote()" not in waiter
    assert "systemctl" not in waiter
    assert "systemd-run" not in waiter
    assert "handoff_request" in waiter
    assert "exit 75" in waiter


def test_remote_attestation_quotes_the_complete_remote_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSH arguments after the target are parsed again by a remote shell."""

    captured: list[list[str]] = []

    def fake_run(command, **_kwargs):
        captured.append(list(command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    prefix = [
        "/usr/bin/ssh",
        "-F",
        "/opt/rwkv ssh/config",
        "-o",
        "BatchMode=yes",
        "rwkv-8222",
    ]
    api_key = "literal-key; touch /tmp/remote-command-injection"
    gate._run_remote_runtime_attestation(
        repo=tmp_path / "frozen runtime",
        approval_path=tmp_path / "approval.json",
        model="rwkv7-g1i-13.3b-20260805-ctx16384",
        runtime_artifact={
            "endpoint": {
                "scheme": "http",
                "host": "127.0.0.1",
                "port": 18074,
                "api_prefix": "/v1",
            }
        },
        forward_artifact={"verification_argv_prefix": prefix},
        infer_api_key=api_key,
    )

    assert len(captured) == 1
    command = captured[0]
    assert command[: len(prefix)] == prefix
    # A single shell-escaped remote command is the only safe OpenSSH calling
    # convention here.  Passing each token as a local argv item lets ssh join
    # them without quoting before the remote shell sees them.
    assert len(command) == len(prefix) + 1
    remote_argv = shlex.split(command[-1])
    api_key_index = remote_argv.index("--infer-api-key")
    assert remote_argv[api_key_index + 1] == api_key
    assert remote_argv[0:2] == ["/usr/bin/python3", "-I"]


def test_runtime_evidence_rejects_resealed_wrong_host_and_route() -> None:
    evidence = copy.deepcopy(_runtime_evidence_fixture())
    model = "rwkv7-g1i-7.2b-20260805-ctx16384"
    entry = evidence["models"][model]
    runtime_artifact = entry["runtime_attestation"]
    runtime_artifact["host_label"] = "157"
    runtime_artifact["artifact_sha256"] = runtime.artifact_sha256(runtime_artifact)
    _reseal_evidence(evidence)

    with pytest.raises(gate.ProtocolGateError, match="host mismatch"):
        gate._verify_runtime_attestation_evidence(ROOT, evidence)

    evidence = copy.deepcopy(_runtime_evidence_fixture())
    entry = evidence["models"][model]
    scheduler_endpoint = dict(entry["route"]["scheduler_endpoint"])
    scheduler_endpoint["port"] += 1
    entry["route"]["scheduler_endpoint"] = scheduler_endpoint
    _reseal_evidence(evidence)
    with pytest.raises(
        gate.ProtocolGateError,
        match="forward listener/scheduler endpoint mismatch",
    ):
        gate._verify_runtime_attestation_evidence(ROOT, evidence)


def test_protocol_lock_binds_the_exact_runtime_evidence_digest(
    tmp_path: Path,
) -> None:
    approval_path, descriptor = _approval_fixture(tmp_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    evidence_sha = approval["runtime_attestation_evidence"]["evidence_sha256"]

    gate._verify_global_approval(
        tmp_path,
        approval_path,
        locked_descriptor=descriptor,
        locked_runtime_evidence_sha256=evidence_sha,
    )
    with pytest.raises(
        gate.ProtocolGateError,
        match="protocol lock runtime evidence binding mismatch",
    ):
        gate._verify_global_approval(
            tmp_path,
            approval_path,
            locked_descriptor=descriptor,
            locked_runtime_evidence_sha256="f" * 64,
        )


def test_ssh_forward_route_proves_local_tunnel_then_remote_runtime(
    tmp_path: Path,
) -> None:
    forward = _forward_fixture(tmp_path)
    (tmp_path / "remote").mkdir()
    remote_runtime = _runtime_fixture(
        tmp_path / "remote",
        host_label="8222",
        gpu=2,
        port=18074,
    )
    model = "rwkv7-g1i-13.3b-20260805-ctx16384"
    approval = {
        "runtime_attestation_evidence": {
            "models": {
                model: {
                    "runtime_attestation": remote_runtime["payload"],
                    "route": {
                        "kind": "ssh_forward",
                        "scheduler_endpoint": forward["payload"]["endpoint"],
                        "forward_attestation": forward["payload"],
                    },
                }
            }
        }
    }
    remote_calls: list[dict[str, object]] = []

    result = gate._verify_inference_runtime_route(
        repo=ROOT,
        approval=approval,
        approval_path=tmp_path / "approval.json",
        model=model,
        infer_base_url="http://127.0.0.1:29574/v1",
        infer_api_key="rwkv-skills",
        proc_root=forward["proc_root"],
        trusted_uid=os.getuid(),
        security_anchor=forward["trusted"],
        remote_runner=lambda **kwargs: remote_calls.append(kwargs),
    )

    assert result["forward"]["schema"] == runtime.FORWARD_VERIFICATION_SCHEMA
    assert result["remote_runtime_verified"] is True
    assert len(remote_calls) == 1
    assert remote_calls[0]["model"] == model
    assert remote_calls[0]["runtime_artifact"]["endpoint"]["port"] == 18074


def test_ssh_forward_route_fails_closed_when_remote_proof_fails(
    tmp_path: Path,
) -> None:
    forward = _forward_fixture(tmp_path)
    (tmp_path / "remote").mkdir()
    remote_runtime = _runtime_fixture(
        tmp_path / "remote",
        host_label="8222",
        gpu=2,
        port=18074,
    )
    model = "rwkv7-g1i-13.3b-20260805-ctx16384"
    approval = {
        "runtime_attestation_evidence": {
            "models": {
                model: {
                    "runtime_attestation": remote_runtime["payload"],
                    "route": {
                        "kind": "ssh_forward",
                        "scheduler_endpoint": forward["payload"]["endpoint"],
                        "forward_attestation": forward["payload"],
                    },
                }
            }
        }
    }

    def reject_remote(**_kwargs):
        raise gate.ProtocolGateError("remote runtime attestation failed")

    with pytest.raises(
        gate.ProtocolGateError,
        match="remote runtime attestation failed",
    ):
        gate._verify_inference_runtime_route(
            repo=ROOT,
            approval=approval,
            approval_path=tmp_path / "approval.json",
            model=model,
            infer_base_url="http://127.0.0.1:29574/v1",
            infer_api_key="rwkv-skills",
            proc_root=forward["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=forward["trusted"],
            remote_runner=reject_remote,
        )
