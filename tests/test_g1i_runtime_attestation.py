from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ops.g1i_strict46 import runtime_attestation as attestation


MODEL_NAME = "rwkv7-g1i-13.3b-20260805-ctx16384"


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _argv(executable: Path, weight: Path, port: int = 18074) -> list[str]:
    return [
        str(executable),
        "serve",
        str(weight),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-key",
        "not-persisted-in-attestation",
        "--tokenizer-mode",
        "rwkv",
        "--trust-request-chat-template",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "rwkv",
        "--max-model-len",
        "16384",
        "--served-model-name",
        MODEL_NAME,
        "--gpu-memory-utilization",
        "0.98",
        "--max-num-batched-tokens",
        "98304",
        "--max-num-seqs",
        "640",
        "--override-generation-config",
        '{"temperature":1e-5}',
    ]


def _environment(runtime_tree: Path, gpu: int = 2) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": str(runtime_tree),
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_RWKV7_WKV_MODE": "fp32io16",
        "VLLM_USE_RAPID_SAMPLER": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # The broad word "TOKEN" must not make a semantic vLLM setting
        # disappear from the attestation contract.
        "VLLM_MAX_NUM_BATCHED_TOKENS": "98304",
        # This must be observed and bound even though it is not on a hand list.
        "VLLM_CUSTOM_KERNEL_SWITCH": "strict",
        # Secrets are deliberately neither persisted nor returned.
        "RWKV_INFER_API_KEY": "secret-value",
    }


def _proc_tcp_row(port: int, inode: str) -> str:
    return (
        "  0: 0100007F:"
        f"{port:04X} 00000000:0000 0A 00000000:00000000 "
        f"00:00000000 00000000 1000 0 {inode} 1 0000000000000000 100 0 0 10 0\n"
    )


def _seal(path: Path, payload: dict[str, object]) -> None:
    payload["artifact_sha256"] = attestation.artifact_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    path.chmod(0o444)


def _runtime_fixture(
    tmp_path: Path, *, host_label: str = "8222", gpu: int = 2, port: int = 18074
):
    trusted = tmp_path / "trusted"
    trusted.mkdir(mode=0o700)
    runtime_tree = trusted / "vllm-rwkv"
    _write(
        runtime_tree / "vllm" / "model_executor" / "rwkv.py", b"RUNTIME = 'strict'\n"
    )
    _write(runtime_tree / "pyproject.toml", b"[project]\nname='vllm-rwkv'\n")
    executable = trusted / "venv" / "bin" / "vllm"
    _write(executable, b"#!/usr/bin/python3\n")
    executable.chmod(0o755)
    weight = trusted / "weights" / f"{MODEL_NAME}.pth"
    _write(weight, b"correct-g1i-weight-bytes")
    bad_weight = trusted / "weights" / "wrong-but-served-as-g1i.pth"
    _write(bad_weight, b"wrong-weight-bytes")
    working_directory = trusted / "work"
    working_directory.mkdir()

    argv = _argv(executable, weight, port)
    environment = _environment(runtime_tree, gpu)
    cgroup = [
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
        "rwkv-g1i-13p3b-gpu2-16k-c640.service"
    ]
    endpoint = {
        "scheme": "http",
        "host": "127.0.0.1",
        "port": port,
        "api_prefix": "/v1",
    }
    payload: dict[str, object] = {
        "schema": attestation.ATTESTATION_SCHEMA,
        "artifact_sha256": "",
        "host_label": host_label,
        "endpoint": endpoint,
        "model": {"name": MODEL_NAME, "weight": attestation.describe_file(weight)},
        "process": {
            "uid": os.getuid(),
            "executable": attestation.describe_file(executable),
            "working_directory": str(working_directory.resolve()),
            "argv_redacted": attestation.redact_argv(argv),
            "environment": attestation._semantic_environment(environment),
            "cgroup": cgroup,
            "systemd_unit": "rwkv-g1i-13p3b-gpu2-16k-c640.service",
            "gpu_index": gpu,
            "launch_parameters": attestation.launch_parameters(argv),
        },
        "runtime_tree": attestation.describe_tree(runtime_tree),
    }
    artifact_path = trusted / "attestations" / "g1i-13p3.json"
    _seal(artifact_path, payload)

    proc_root = tmp_path / "proc"
    pid = 4242
    inode = "998877"
    _write(
        proc_root / "net" / "tcp",
        (
            "sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
            + _proc_tcp_row(port, inode)
        ).encode(),
    )
    _write(
        proc_root / "net" / "tcp6",
        b"sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n",
    )
    process_root = proc_root / str(pid)
    (process_root / "fd").mkdir(parents=True)
    os.symlink(f"socket:[{inode}]", process_root / "fd" / "7")
    _write(process_root / "cmdline", b"\0".join(item.encode() for item in argv) + b"\0")
    _write(
        process_root / "environ",
        b"\0".join(f"{key}={value}".encode() for key, value in environment.items())
        + b"\0",
    )
    _write(process_root / "cgroup", (cgroup[0] + "\n").encode())
    # After the closing comm parenthesis, start time (field 22) is item 19.
    stat_fields = ["S", *(["0"] * 18), "123456"]
    _write(
        process_root / "stat",
        f"{pid} (vllm api server) {' '.join(stat_fields)}\n".encode(),
    )
    os.symlink(executable, process_root / "exe")
    os.symlink(working_directory, process_root / "cwd")

    return {
        "trusted": trusted,
        "runtime_tree": runtime_tree,
        "weight": weight,
        "bad_weight": bad_weight,
        "artifact": artifact_path,
        "payload": payload,
        "proc_root": proc_root,
        "process_root": process_root,
        "argv": argv,
        "endpoint_url": f"http://127.0.0.1:{port}/v1",
    }


def _models(_endpoint, _api_key):
    # A correct presentation name is not accepted as proof of model identity.
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


def _forward_fixture(tmp_path: Path):
    trusted = tmp_path / "trusted-forward"
    trusted.mkdir(mode=0o700)
    ssh = trusted / "bin" / "ssh"
    _write(ssh, b"approved-openssh-client")
    ssh.chmod(0o755)
    config = trusted / "ssh" / "config"
    known_hosts = trusted / "ssh" / "known_hosts"
    identity = trusted / "ssh" / "id_ed25519"
    _write(
        config,
        (
            "Host rwkv-8222\n"
            "  HostName 192.168.0.222\n"
            f"  IdentityFile {identity}\n"
            f"  UserKnownHostsFile {known_hosts}\n"
        ).encode(),
    )
    _write(known_hosts, b"192.168.0.222 ssh-ed25519 fixed-host-key\n")
    _write(identity, b"fixed-test-private-key")
    working_directory = trusted / "work"
    working_directory.mkdir()
    endpoint = {
        "scheme": "http",
        "host": "127.0.0.1",
        "port": 29574,
        "api_prefix": "/v1",
    }
    destination_endpoint = {
        "scheme": "http",
        "host": "127.0.0.1",
        "port": 18074,
        "api_prefix": "/v1",
    }
    argv = [
        str(ssh.resolve()),
        "-N",
        "-F",
        str(config.resolve()),
        "-i",
        str(identity.resolve()),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts.resolve()}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-L",
        "127.0.0.1:29574:127.0.0.1:18074",
        "rwkv-8222",
    ]
    verification_argv_prefix = [
        str(ssh.resolve()),
        "-F",
        str(config.resolve()),
        "-i",
        str(identity.resolve()),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts.resolve()}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "rwkv-8222",
    ]
    cgroup = [
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
        "rwkv-g1i-forward-8222.service"
    ]
    transport_files = sorted(
        [
            attestation.describe_file(config),
            attestation.describe_file(known_hosts),
            attestation.describe_file(identity),
        ],
        key=lambda item: item["path"],
    )
    payload: dict[str, object] = {
        "schema": attestation.FORWARD_ATTESTATION_SCHEMA,
        "artifact_sha256": "",
        "host_label": "157",
        "endpoint": endpoint,
        "destination": {
            "host_label": "8222",
            "endpoint": destination_endpoint,
        },
        "process": {
            "uid": os.getuid(),
            "executable": attestation.describe_file(ssh),
            "working_directory": str(working_directory.resolve()),
            "argv_redacted": argv,
            "cgroup": cgroup,
            "systemd_unit": "rwkv-g1i-forward-8222.service",
        },
        "transport_files": transport_files,
        "verification_argv_prefix": verification_argv_prefix,
    }
    payload["artifact_sha256"] = attestation.artifact_sha256(payload)

    proc_root = tmp_path / "proc-forward"
    pid = 5252
    inode = "445566"
    _write(
        proc_root / "net" / "tcp",
        (
            "sl local_address rem_address st tx_queue rx_queue tr tm->when "
            "retrnsmt uid timeout inode\n"
            + _proc_tcp_row(29574, inode)
        ).encode(),
    )
    _write(
        proc_root / "net" / "tcp6",
        b"sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n",
    )
    process_root = proc_root / str(pid)
    (process_root / "fd").mkdir(parents=True)
    os.symlink(f"socket:[{inode}]", process_root / "fd" / "5")
    _write(process_root / "cmdline", b"\0".join(item.encode() for item in argv) + b"\0")
    _write(process_root / "cgroup", (cgroup[0] + "\n").encode())
    stat_fields = ["S", *(["0"] * 18), "246810"]
    _write(
        process_root / "stat",
        f"{pid} (ssh) {' '.join(stat_fields)}\n".encode(),
    )
    os.symlink(ssh, process_root / "exe")
    os.symlink(working_directory, process_root / "cwd")
    return {
        "trusted": trusted,
        "ssh": ssh,
        "config": config,
        "payload": payload,
        "proc_root": proc_root,
        "process_root": process_root,
        "argv": argv,
    }


def test_verifies_listener_process_weight_runtime_gpu_and_launch(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path)
    result = attestation.verify_runtime_attestation(
        fixture["artifact"],
        MODEL_NAME,
        fixture["endpoint_url"],
        proc_root=fixture["proc_root"],
        trusted_uid=os.getuid(),
        security_anchor=fixture["trusted"],
        models_probe=_models,
    )

    assert result["schema"] == attestation.VERIFICATION_SCHEMA
    assert result["pid"] == 4242
    assert result["listener_inode"] == "998877"
    assert result["actual_weight"]["path"] == str(fixture["weight"].resolve())
    assert result["display_model"] == MODEL_NAME
    assert result["gpu_index"] == 2
    assert result["systemd_unit"] == "rwkv-g1i-13p3b-gpu2-16k-c640.service"
    assert "secret-value" not in json.dumps(result)


def test_correct_served_name_cannot_spoof_wrong_actual_weight(tmp_path: Path) -> None:
    fixture = _runtime_fixture(tmp_path)
    wrong_argv = list(fixture["argv"])
    wrong_argv[2] = str(fixture["bad_weight"])
    fixture["process_root"].joinpath("cmdline").write_bytes(
        b"\0".join(item.encode() for item in wrong_argv) + b"\0"
    )

    with pytest.raises(
        attestation.RuntimeAttestationError, match="actual model weight path mismatch"
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )


def test_weight_bytes_are_verified_even_when_path_and_served_name_match(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path)
    fixture["weight"].write_bytes(b"silently-replaced-weight")

    with pytest.raises(
        attestation.RuntimeAttestationError, match="file identity mismatch"
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )


@pytest.mark.parametrize(("gpu", "port"), [(3, 18074), (2, 18073)])
def test_8222_reserved_gpu3_and_port18073_are_rejected(
    tmp_path: Path, gpu: int, port: int
) -> None:
    fixture = _runtime_fixture(tmp_path, gpu=gpu, port=port)

    with pytest.raises(
        attestation.RuntimeAttestationError, match="reserved and forbidden"
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )


def test_attestation_artifact_must_have_no_write_bits(tmp_path: Path) -> None:
    fixture = _runtime_fixture(tmp_path)
    fixture["artifact"].chmod(0o644)

    with pytest.raises(
        attestation.RuntimeAttestationError, match="must have no write bits"
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )


def test_runtime_tree_mutation_after_approval_is_rejected(tmp_path: Path) -> None:
    fixture = _runtime_fixture(tmp_path)
    fixture["runtime_tree"].joinpath("vllm", "model_executor", "rwkv.py").write_text(
        "RUNTIME = 'mutated'\n", encoding="utf-8"
    )

    with pytest.raises(
        attestation.RuntimeAttestationError, match="runtime tree digest/inventory"
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )


def test_attestation_rejects_pythonpath_shadow_tree(tmp_path: Path) -> None:
    fixture = _runtime_fixture(tmp_path)
    payload = fixture["payload"]
    payload["process"]["environment"]["PYTHONPATH"] += os.pathsep + "/tmp/shadow"
    artifact = fixture["trusted"] / "attestations" / "shadowed-runtime.json"
    _seal(artifact, payload)

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="PYTHONPATH must contain exactly",
    ):
        attestation.verify_runtime_attestation(
            artifact,
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"object": "list", "data": []},
        {
            "object": "list",
            "data": [
                {"id": MODEL_NAME, "object": "model"},
                {"id": "spoofed-second-model", "object": "model"},
            ],
        },
    ],
)
def test_models_probe_must_expose_exactly_one_model(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    fixture = _runtime_fixture(tmp_path)

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="must expose exactly one model id",
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=lambda _endpoint, _api_key: payload,
        )


def test_models_probe_display_name_is_only_secondary_evidence(tmp_path: Path) -> None:
    fixture = _runtime_fixture(tmp_path)

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="endpoint display model does not match attestation",
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=lambda _endpoint, _api_key: {
                "object": "list",
                "data": [{"id": "served-name-spoof", "object": "model"}],
            },
        )


def test_forwarded_157_port_cannot_stand_in_for_8222_host_proof(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, host_label="8222", port=18074)

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="requested endpoint does not match the trusted attestation",
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            "http://127.0.0.1:29574/v1",
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )

    # The proof succeeds only when executed against the inference-host-local
    # socket.  A caller on 157 must therefore run this verifier over the fixed
    # SSH trust boundary on 8222, then probe the forwarded endpoint separately.
    result = attestation.verify_runtime_attestation(
        fixture["artifact"],
        MODEL_NAME,
        "http://127.0.0.1:18074/v1",
        proc_root=fixture["proc_root"],
        trusted_uid=os.getuid(),
        security_anchor=fixture["trusted"],
        models_probe=_models,
    )
    assert result["host_label"] == "8222"
    assert result["endpoint"]["port"] == 18074


def test_wrong_live_gpu_is_rejected_even_if_models_name_is_correct(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, host_label="157", gpu=2)
    changed = _environment(fixture["runtime_tree"], gpu=1)
    fixture["process_root"].joinpath("environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in changed.items())
        + b"\0"
    )

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="semantic environment does not match attestation",
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )


def test_semantic_token_named_environment_setting_is_still_bound(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path)
    changed = _environment(fixture["runtime_tree"], gpu=2)
    changed["VLLM_MAX_NUM_BATCHED_TOKENS"] = "1"
    fixture["process_root"].joinpath("environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in changed.items())
        + b"\0"
    )

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="semantic environment does not match attestation",
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )


def test_wrong_live_port_is_rejected_even_if_models_name_is_correct(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path)
    changed = list(fixture["argv"])
    port_index = changed.index("--port") + 1
    changed[port_index] = "18075"
    fixture["process_root"].joinpath("cmdline").write_bytes(
        b"\0".join(item.encode() for item in changed) + b"\0"
    )

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="command line does not match attestation",
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )


def test_wrong_live_systemd_unit_is_rejected(tmp_path: Path) -> None:
    fixture = _runtime_fixture(tmp_path)
    fixture["process_root"].joinpath("cgroup").write_text(
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
        "unapproved-vllm.service\n",
        encoding="utf-8",
    )

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="cgroup/systemd unit does not match attestation",
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )


def test_ambiguous_listener_is_rejected_before_http_identity(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path)
    table = fixture["proc_root"] / "net" / "tcp"
    table.write_text(
        table.read_text(encoding="ascii") + _proc_tcp_row(18074, "112233"),
        encoding="ascii",
    )

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="expected exactly one listening socket",
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )


def test_listener_swap_during_attestation_is_rejected(tmp_path: Path) -> None:
    fixture = _runtime_fixture(tmp_path)

    def swap_listener(_endpoint, _api_key):
        table = fixture["proc_root"] / "net" / "tcp"
        header = table.read_text(encoding="ascii").splitlines()[0]
        table.write_text(
            header + "\n" + _proc_tcp_row(18074, "112233"),
            encoding="ascii",
        )
        descriptor = fixture["process_root"] / "fd" / "7"
        descriptor.unlink()
        os.symlink("socket:[112233]", descriptor)
        return _models(_endpoint, _api_key)

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="listener changed during runtime verification",
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=swap_listener,
        )


def test_handoff_requires_a_fresh_runtime_proof(tmp_path: Path) -> None:
    fixture = _runtime_fixture(tmp_path)
    first = attestation.verify_runtime_attestation(
        fixture["artifact"],
        MODEL_NAME,
        fixture["endpoint_url"],
        proc_root=fixture["proc_root"],
        trusted_uid=os.getuid(),
        security_anchor=fixture["trusted"],
        models_probe=_models,
    )
    assert first["actual_weight"]["sha256"] == fixture["payload"]["model"][
        "weight"
    ]["sha256"]

    # Simulate a handoff/restart that keeps the same port and presentation
    # name but launches a different weight.  Reusing ``first`` would be unsafe;
    # running the verifier again must reject the new listener identity.
    wrong_argv = list(fixture["argv"])
    wrong_argv[2] = str(fixture["bad_weight"])
    fixture["process_root"].joinpath("cmdline").write_bytes(
        b"\0".join(item.encode() for item in wrong_argv) + b"\0"
    )

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="actual model weight path mismatch",
    ):
        attestation.verify_runtime_attestation(
            fixture["artifact"],
            MODEL_NAME,
            fixture["endpoint_url"],
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
            models_probe=_models,
        )


def test_forward_attestation_binds_157_listener_to_8222_destination(
    tmp_path: Path,
) -> None:
    fixture = _forward_fixture(tmp_path)

    result = attestation.verify_forward_attestation_payload(
        fixture["payload"],
        "http://127.0.0.1:29574/v1",
        proc_root=fixture["proc_root"],
        trusted_uid=os.getuid(),
        security_anchor=fixture["trusted"],
    )

    assert result["schema"] == attestation.FORWARD_VERIFICATION_SCHEMA
    assert result["endpoint"]["port"] == 29574
    assert result["destination"]["host_label"] == "8222"
    assert result["destination"]["endpoint"]["port"] == 18074
    assert result["pid"] == 5252


def test_forward_attestation_rejects_live_mapping_drift(tmp_path: Path) -> None:
    fixture = _forward_fixture(tmp_path)
    changed = list(fixture["argv"])
    mapping_index = changed.index("-L") + 1
    changed[mapping_index] = "127.0.0.1:29574:127.0.0.1:18075"
    fixture["process_root"].joinpath("cmdline").write_bytes(
        b"\0".join(item.encode() for item in changed) + b"\0"
    )

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="forward listener command line mismatch",
    ):
        attestation.verify_forward_attestation_payload(
            fixture["payload"],
            "http://127.0.0.1:29574/v1",
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
        )


def test_forward_attestation_rejects_transport_config_mutation(
    tmp_path: Path,
) -> None:
    fixture = _forward_fixture(tmp_path)
    fixture["config"].write_text(
        "Host rwkv-8222\n  StrictHostKeyChecking no\n",
        encoding="utf-8",
    )

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="file identity mismatch",
    ):
        attestation.verify_forward_attestation_payload(
            fixture["payload"],
            "http://127.0.0.1:29574/v1",
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
        )


def test_forward_attestation_requires_fail_closed_ssh_options(
    tmp_path: Path,
) -> None:
    fixture = _forward_fixture(tmp_path)
    payload = fixture["payload"]
    verification_argv = list(payload["verification_argv_prefix"])
    strict_index = verification_argv.index("StrictHostKeyChecking=yes")
    verification_argv[strict_index] = "StrictHostKeyChecking=accept-new"
    payload["verification_argv_prefix"] = verification_argv
    payload["artifact_sha256"] = attestation.artifact_sha256(payload)

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="SSH process/verifier options are not fail-closed",
    ):
        attestation.verify_forward_attestation_payload(
            payload,
            "http://127.0.0.1:29574/v1",
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
        )


def test_forward_attestation_binds_identity_and_known_hosts_bytes(
    tmp_path: Path,
) -> None:
    fixture = _forward_fixture(tmp_path)
    payload = fixture["payload"]
    payload["transport_files"] = [
        descriptor
        for descriptor in payload["transport_files"]
        if descriptor["path"] == str(fixture["config"].resolve())
    ]
    payload["artifact_sha256"] = attestation.artifact_sha256(payload)

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="identity|known.hosts|transport",
    ):
        attestation.verify_forward_attestation_payload(
            payload,
            "http://127.0.0.1:29574/v1",
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
        )


def test_forward_attestation_rejects_implicit_ssh_credentials_and_host_keys(
    tmp_path: Path,
) -> None:
    fixture = _forward_fixture(tmp_path)
    payload = fixture["payload"]
    verification_argv = list(payload["verification_argv_prefix"])
    identity_index = verification_argv.index("-i")
    del verification_argv[identity_index : identity_index + 2]
    user_known_hosts = next(
        value
        for value in verification_argv
        if value.startswith("UserKnownHostsFile=")
    )
    option_index = verification_argv.index(user_known_hosts)
    del verification_argv[option_index - 1 : option_index + 1]
    payload["verification_argv_prefix"] = verification_argv
    payload["artifact_sha256"] = attestation.artifact_sha256(payload)

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match=r"identity|known.hosts|explicit|exactly one -i",
    ):
        attestation.verify_forward_attestation_payload(
            payload,
            "http://127.0.0.1:29574/v1",
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
        )


@pytest.mark.parametrize(
    ("scheduler_port", "destination_port"),
    [(18073, 18074), (29574, 18073)],
)
def test_forward_attestation_rejects_reserved_18073_on_either_side(
    tmp_path: Path,
    scheduler_port: int,
    destination_port: int,
) -> None:
    fixture = _forward_fixture(tmp_path)
    payload = fixture["payload"]
    payload["endpoint"]["port"] = scheduler_port
    payload["destination"]["endpoint"]["port"] = destination_port
    payload["artifact_sha256"] = attestation.artifact_sha256(payload)

    with pytest.raises(
        attestation.RuntimeAttestationError,
        match="18073 is reserved and forbidden",
    ):
        attestation.verify_forward_attestation_payload(
            payload,
            f"http://127.0.0.1:{scheduler_port}/v1",
            proc_root=fixture["proc_root"],
            trusted_uid=os.getuid(),
            security_anchor=fixture["trusted"],
        )
