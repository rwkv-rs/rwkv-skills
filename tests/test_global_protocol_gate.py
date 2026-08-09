from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

import pytest

from ops.g1i_strict46 import require_global_protocol_gate as gate
from ops.g1i_strict46 import runtime_attestation as runtime


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": gate._sha256_file(path)}


def _runtime_evidence_fixture() -> dict[str, object]:
    models: dict[str, object] = {}
    layouts = {
        "rwkv7-g1i-1.5b-20260805-ctx16384": ("157", 19415, 3, 19415),
        "rwkv7-g1i-2.9b-20260805-ctx16384": ("157", 19429, 3, 19429),
        "rwkv7-g1i-7.2b-20260805-ctx16384": ("8222", 18072, 2, 29572),
        "rwkv7-g1i-13.3b-20260805-ctx16384": ("8222", 18074, 2, 29574),
    }
    for index, (model, (host, runtime_port, gpu, scheduler_port)) in enumerate(
        layouts.items(),
        start=1,
    ):
        executable = "/opt/rwkv-infer/bin/vllm"
        weight = f"/opt/rwkv-weights/{model}.pth"
        runtime_root = f"/opt/rwkv-infer/runtime-{index}"
        argv = [
            executable,
            "serve",
            weight,
            "--host",
            "127.0.0.1",
            "--port",
            str(runtime_port),
            "--tokenizer-mode",
            "rwkv",
            "--trust-request-chat-template",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "rwkv",
            "--max-model-len",
            "16384",
            "--served-model-name",
            model,
            "--gpu-memory-utilization",
            "0.98",
            "--max-num-batched-tokens",
            "98304",
            "--max-num-seqs",
            "640",
            "--override-generation-config",
            '{"temperature":1e-5}',
        ]
        tree_files = [
            {"relative_path": "vllm.py", "bytes": 1, "sha256": f"{index:x}" * 64}
        ]
        runtime_artifact: dict[str, object] = {
            "schema": runtime.ATTESTATION_SCHEMA,
            "artifact_sha256": "",
            "host_label": host,
            "endpoint": {
                "scheme": "http",
                "host": "127.0.0.1",
                "port": runtime_port,
                "api_prefix": "/v1",
            },
            "model": {
                "name": model,
                "weight": {"path": weight, "bytes": index, "sha256": f"{index:x}" * 64},
            },
            "process": {
                "uid": 1001,
                "executable": {
                    "path": executable,
                    "bytes": index,
                    "sha256": f"{index + 4:x}" * 64,
                },
                "working_directory": runtime_root,
                "argv_redacted": argv,
                "environment": {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONPATH": runtime_root,
                    **runtime.REQUIRED_ENVIRONMENT,
                },
                "cgroup": [
                    f"0::/user.slice/app.slice/rwkv-g1i-{index}.service"
                ],
                "systemd_unit": f"rwkv-g1i-{index}.service",
                "gpu_index": gpu,
                "launch_parameters": runtime.launch_parameters(argv),
            },
            "runtime_tree": {
                "path": runtime_root,
                "tree_sha256": runtime.canonical_sha256(tree_files),
                "files": tree_files,
            },
        }
        runtime_artifact["artifact_sha256"] = runtime.artifact_sha256(
            runtime_artifact
        )
        scheduler_endpoint = {
            "scheme": "http",
            "host": "127.0.0.1",
            "port": scheduler_port,
            "api_prefix": "/v1",
        }
        if host == "157":
            route: dict[str, object] = {
                "kind": "local",
                "scheduler_endpoint": scheduler_endpoint,
            }
        else:
            ssh_config = "/etc/rwkv-strict46/ssh_config"
            ssh_binary = "/usr/bin/ssh"
            ssh_identity = f"/etc/rwkv-strict46/id_ed25519-{index}"
            ssh_known_hosts = f"/etc/rwkv-strict46/known_hosts-{index}"
            ssh_options = [
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                f"UserKnownHostsFile={ssh_known_hosts}",
                "-o",
                "GlobalKnownHostsFile=/dev/null",
                "-o",
                "PasswordAuthentication=no",
                "-o",
                "KbdInteractiveAuthentication=no",
            ]
            target = "strict46-8222"
            forward_argv = [
                ssh_binary,
                "-N",
                "-L",
                f"127.0.0.1:{scheduler_port}:127.0.0.1:{runtime_port}",
                "-F",
                ssh_config,
                "-i",
                ssh_identity,
                *ssh_options,
                target,
            ]
            forward_artifact: dict[str, object] = {
                "schema": runtime.FORWARD_ATTESTATION_SCHEMA,
                "artifact_sha256": "",
                "host_label": "157",
                "endpoint": scheduler_endpoint,
                "destination": {
                    "host_label": "8222",
                    "endpoint": runtime_artifact["endpoint"],
                },
                "process": {
                    "uid": 1001,
                    "executable": {
                        "path": ssh_binary,
                        "bytes": 1,
                        "sha256": "a" * 64,
                    },
                    "working_directory": "/opt/rwkv-strict46",
                    "argv_redacted": forward_argv,
                    "cgroup": [
                        f"0::/user.slice/app.slice/rwkv-forward-{index}.service"
                    ],
                    "systemd_unit": f"rwkv-forward-{index}.service",
                },
                "transport_files": sorted(
                    [
                        {"path": ssh_config, "bytes": 1, "sha256": "b" * 64},
                        {"path": ssh_identity, "bytes": 1, "sha256": "c" * 64},
                        {
                            "path": ssh_known_hosts,
                            "bytes": 1,
                            "sha256": "d" * 64,
                        },
                    ],
                    key=lambda descriptor: str(descriptor["path"]),
                ),
                "verification_argv_prefix": [
                    ssh_binary,
                    "-F",
                    ssh_config,
                    "-i",
                    ssh_identity,
                    *ssh_options,
                    target,
                ],
            }
            forward_artifact["artifact_sha256"] = runtime.artifact_sha256(
                forward_artifact
            )
            route = {
                "kind": "ssh_forward",
                "scheduler_endpoint": scheduler_endpoint,
                "forward_attestation": forward_artifact,
            }
        models[model] = {
            "runtime_attestation": runtime_artifact,
            "route": route,
        }
    evidence: dict[str, object] = {
        "schema_version": gate.RUNTIME_EVIDENCE_SCHEMA,
        "models": models,
    }
    evidence["evidence_sha256"] = gate._canonical_json_sha256(evidence)
    return evidence


def _approval_fixture(
    repo: Path,
    *,
    mutate: Callable[[dict[str, object]], None] | None = None,
    writable: bool = False,
) -> tuple[Path, dict[str, str]]:
    candidate = repo / "src/eval/metrics/free_response.py"
    audit = repo / "scripts/oneoff/audit_free_response_extractor_global.py"
    merge_script = repo / "scripts/oneoff/merge_free_response_global_audit.py"
    snapshot = repo / "src/eval/datasets/snapshot.py"
    persistence = repo / "src/eval/evaluating/task_persistence.py"
    knowledge_pipeline = repo / "src/eval/tasks/knowledge/pipeline.py"
    runtime_verifier = repo / "ops/g1i_strict46/runtime_attestation.py"
    for path, body in (
        (candidate, "CANDIDATE = 1\n"),
        (audit, "AUDIT = 1\n"),
        (merge_script, "MERGE = 1\n"),
        (snapshot, "SNAPSHOT = 1\n"),
        (persistence, "PERSISTENCE = 1\n"),
        (knowledge_pipeline, "PIPELINE = 1\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    runtime_verifier.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "ops/g1i_strict46/runtime_attestation.py", runtime_verifier)

    artifact_records: list[dict[str, object]] = []
    for index, group in enumerate(("strategy_a", "strategy_b", "strategy_c")):
        artifact_path = repo / "evidence" / f"{group}-part-0.json"
        artifact_payload = json.dumps({"group": group, "partition_index": 0}).encode()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(artifact_payload)
        artifact_path.chmod(0o444)
        artifact_records.append(
            {
                "group": group,
                "partition_index": 0,
                "path": str(artifact_path),
                "bytes": len(artifact_payload),
                "sha256": gate._sha256_file(artifact_path),
            }
        )
    input_artifacts_sha256 = gate._canonical_json_sha256(artifact_records)
    provenance = {
        "database_snapshot_id": "00000003-00000001-1",
        "metadata_snapshot_digest": "1" * 64,
        "dataset_digests": {
            "demo_test": {
                "file_sha256": "2" * 64,
                "records_sha256": "3" * 64,
                "record_count": 2,
            }
        },
        "module_shas": {
            "audit": gate._sha256_file(audit),
            "baseline": "4" * 64,
            "candidate": gate._sha256_file(candidate),
        },
        "dependency_environment_sha256": "5" * 64,
        "dependency_manifest_file_sha256": "6" * 64,
        "task_inventory_digest": "7" * 64,
        "task_count": 1,
        "input_artifacts": artifact_records,
        "input_artifacts_sha256": input_artifacts_sha256,
    }
    audit_mode = {
        "full_scan_a": True,
        "full_real_scorer": True,
        "max_structural_rows": None,
        "database_snapshot_imported": True,
        "metadata_snapshot_digest_verified": True,
        "question_source": "current_production_snapshot",
        "independent_order_probe": True,
        "stable_row_order": "task_id,completions_id,eval_id",
        "frozen_code_verified": True,
        "dependency_manifest_verified": True,
        "atomic_part_artifact": True,
    }
    merged_result = {
        "gate": {"passed": True, "full_scan": True},
        "groups": ["strategy_a", "strategy_b", "strategy_c"],
        "merge_script_sha256": gate._sha256_file(merge_script),
        "input_artifacts": artifact_records,
        "input_artifacts_sha256": input_artifacts_sha256,
        "production_provenance": provenance,
        "audit_mode": audit_mode,
        "totals": {
            "scoring_errors": 0,
            "indeterminate_rows": 0,
            "judge_input_affected_rows": 0,
            "replay_affected_rows": 0,
            "blocking_timeout_count": 0,
        },
    }
    merged_path = repo / "evidence/merged.json"
    _write_json(merged_path, merged_result)
    merge_acceptance = {
        "schema_version": gate.MERGE_ACCEPTANCE_SCHEMA,
        "accepted": True,
        "gate_passed": True,
        "merge_script_sha256": gate._sha256_file(merge_script),
        "json": _descriptor(merged_path),
        "production_provenance": provenance,
    }
    merge_acceptance_path = repo / "evidence/merged.json.acceptance.json"
    _write_json(merge_acceptance_path, merge_acceptance)

    protocol = {
        "protocol_version": gate.JUDGE_PROTOCOL_VERSION,
        "model": "judge-model",
        "temperature": 0.0,
        "prompt_template_sha256": "8" * 64,
        "max_completion_tokens": 64,
        "response_contract": gate.JUDGE_RESPONSE_CONTRACT,
        "stream": False,
        "qwen3_enable_thinking": None,
        "max_workers": 8,
        "max_retries": 3,
        "recovery_rounds": 2,
    }
    rows = [
        {
            "task_id": 11 + index,
            "score_id": 21 + index,
            "benchmark": benchmark,
            "model": "rwkv7-g1i-1.5b-20260805-ctx16384",
            "protocol": {
                **protocol,
                "protocol_fingerprint_sha256": gate._canonical_json_sha256(protocol),
            },
        }
        for index, benchmark in enumerate(sorted(gate.EXPECTED_JUDGE_BENCHMARKS))
    ]
    judge_evidence: dict[str, object] = {
        "locked": True,
        "source": "production_score_metrics",
        "task_ids": [row["task_id"] for row in rows],
        "score_ids": [row["score_id"] for row in rows],
        "benchmarks": [row["benchmark"] for row in rows],
        "rows": rows,
    }
    judge_evidence["evidence_sha256"] = gate._canonical_json_sha256(judge_evidence)
    approval: dict[str, object] = gate.build_global_approval_payload(
        repo,
        merge_acceptance_path=merge_acceptance_path,
        judge_protocol_evidence=judge_evidence,
        runtime_attestation_evidence=_runtime_evidence_fixture(),
    )
    if mutate is not None:
        mutate(approval)
    approval.pop("approval_sha256", None)
    approval["approval_sha256"] = gate._canonical_json_sha256(approval)
    approval_path = (
        repo
        / gate.APPROVAL_DIRECTORY
        / f"{approval['approval_sha256']}.acceptance.json"
    )
    _write_json(approval_path, approval)
    if not writable:
        approval_path.chmod(0o444)
    return approval_path, gate._path_descriptor(repo, approval_path)


def test_global_approval_accepts_exact_content_addressed_pass_fixture(
    tmp_path: Path,
) -> None:
    approval, descriptor = _approval_fixture(tmp_path)

    gate._verify_global_approval(
        tmp_path,
        approval,
        locked_descriptor=descriptor,
    )


def test_global_approval_rejects_fail_forgery_staleness_and_replacement(
    tmp_path: Path,
) -> None:
    failed_repo = tmp_path / "failed"
    failed, failed_descriptor = _approval_fixture(
        failed_repo,
        mutate=lambda value: value.update(accepted=False, decision="FAIL"),
    )
    with pytest.raises(gate.ProtocolGateError, match="explicit PASS"):
        gate._verify_global_approval(
            failed_repo,
            failed,
            locked_descriptor=failed_descriptor,
        )

    stale_repo = tmp_path / "stale"
    stale, stale_descriptor = _approval_fixture(stale_repo)
    (stale_repo / "src/eval/metrics/free_response.py").write_text("CANDIDATE = 2\n")
    with pytest.raises(gate.ProtocolGateError, match="candidate SHA is stale"):
        gate._verify_global_approval(
            stale_repo,
            stale,
            locked_descriptor=stale_descriptor,
        )

    replaced_repo = tmp_path / "replaced"
    replaced, replaced_descriptor = _approval_fixture(replaced_repo)
    (replaced_repo / "evidence/merged.json").write_text("{}")
    with pytest.raises(gate.ProtocolGateError, match="merged result SHA mismatch"):
        gate._verify_global_approval(
            replaced_repo,
            replaced,
            locked_descriptor=replaced_descriptor,
        )

    transitive_repo = tmp_path / "transitive"
    transitive, transitive_descriptor = _approval_fixture(transitive_repo)
    (transitive_repo / "src/eval/tasks/knowledge/pipeline.py").write_text(
        "PIPELINE = 2\n"
    )
    with pytest.raises(gate.ProtocolGateError, match="protocol contract is stale"):
        gate._verify_global_approval(
            transitive_repo,
            transitive,
            locked_descriptor=transitive_descriptor,
        )

    merge_repo = tmp_path / "merge-script"
    merge_approval, merge_descriptor = _approval_fixture(merge_repo)
    (merge_repo / "scripts/oneoff/merge_free_response_global_audit.py").write_text(
        "MERGE = 2\n"
    )
    with pytest.raises(gate.ProtocolGateError, match="merge script SHA is stale"):
        gate._verify_global_approval(
            merge_repo,
            merge_approval,
            locked_descriptor=merge_descriptor,
        )

    artifact_repo = tmp_path / "input-artifact"
    artifact_approval, artifact_descriptor = _approval_fixture(artifact_repo)
    artifact = artifact_repo / "evidence/strategy_a-part-0.json"
    artifact.chmod(0o644)
    artifact.write_text("{}")
    artifact.chmod(0o444)
    with pytest.raises(gate.ProtocolGateError, match="input artifact (byte count|SHA)"):
        gate._verify_global_approval(
            artifact_repo,
            artifact_approval,
            locked_descriptor=artifact_descriptor,
        )


def test_global_approval_rejects_unlocked_or_forged_judge_evidence(
    tmp_path: Path,
) -> None:
    writable_repo = tmp_path / "writable"
    writable, descriptor = _approval_fixture(writable_repo, writable=True)
    with pytest.raises(gate.ProtocolGateError, match="writable"):
        gate._verify_global_approval(
            writable_repo,
            writable,
            locked_descriptor=descriptor,
        )

    forged_repo = tmp_path / "judge"

    def forge_judge(value: dict[str, object]) -> None:
        evidence = value["judge_protocol_evidence"]
        evidence["rows"][0]["protocol"]["protocol_fingerprint_sha256"] = "9" * 64
        evidence.pop("evidence_sha256")
        evidence["evidence_sha256"] = gate._canonical_json_sha256(evidence)

    forged, forged_descriptor = _approval_fixture(forged_repo, mutate=forge_judge)
    with pytest.raises(gate.ProtocolGateError, match="fingerprint mismatch"):
        gate._verify_global_approval(
            forged_repo,
            forged,
            locked_descriptor=forged_descriptor,
        )

    incomplete_repo = tmp_path / "judge-incomplete"

    def remove_judge_benchmark(value: dict[str, object]) -> None:
        evidence = value["judge_protocol_evidence"]
        evidence["rows"].pop()
        evidence["task_ids"] = [row["task_id"] for row in evidence["rows"]]
        evidence["score_ids"] = [row["score_id"] for row in evidence["rows"]]
        evidence["benchmarks"] = [row["benchmark"] for row in evidence["rows"]]
        evidence.pop("evidence_sha256")
        evidence["evidence_sha256"] = gate._canonical_json_sha256(evidence)

    incomplete, incomplete_descriptor = _approval_fixture(
        incomplete_repo,
        mutate=remove_judge_benchmark,
    )
    with pytest.raises(gate.ProtocolGateError, match="exact strict-46 Judge"):
        gate._verify_global_approval(
            incomplete_repo,
            incomplete,
            locked_descriptor=incomplete_descriptor,
        )


def test_inference_endpoint_requires_exact_single_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class Response:
        def __init__(self, model: str) -> None:
            self.model = model

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"data": [{"id": self.model}]}).encode()

    expected = "rwkv7-g1i-7.2b-20260805-ctx16384"

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header("Authorization")
        seen["timeout"] = timeout
        return Response(expected)

    monkeypatch.setattr(gate, "urlopen", fake_urlopen)
    assert (
        gate.verify_inference_endpoint(
            "http://127.0.0.1:29574/v1/",
            expected,
            api_key="secret",
        )
        == expected
    )
    assert seen == {
        "url": "http://127.0.0.1:29574/v1/models",
        "authorization": "Bearer secret",
        "timeout": 10.0,
    }

    monkeypatch.setattr(gate, "urlopen", lambda *_args, **_kwargs: Response("wrong"))
    with pytest.raises(gate.ProtocolGateError, match="endpoint model mismatch"):
        gate.verify_inference_endpoint(
            "http://127.0.0.1:29574/v1",
            expected,
            api_key="secret",
        )


def test_ordinary_source_lock_cannot_self_approve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "protocol.py"
    source.write_text("PROTOCOL = 1\n")
    monkeypatch.setattr(gate, "required_protocol_paths", lambda _repo: ("protocol.py",))
    monkeypatch.setattr(gate, "_verify_strict_scope", lambda _repo: None)
    monkeypatch.setattr(gate, "_verify_protocol_invariants", lambda _repo: None)
    lock = tmp_path / "protocol.lock.json"
    _write_json(lock, gate.current_lock_payload(tmp_path))

    with pytest.raises(gate.ProtocolGateError, match="approval is mandatory"):
        gate.require_gate(
            repo=tmp_path,
            lock_path=lock,
            phase="audit",
            model="rwkv7-g1i-1.5b-20260805-ctx16384",
        )


def test_real_gate_is_fail_closed_until_global_audit_approval_exists() -> None:
    environment = dict(os.environ)
    environment.pop("RWKV_GLOBAL_PROTOCOL_APPROVAL", None)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ops/g1i_strict46/require_global_protocol_gate.py"),
            "--repo",
            str(ROOT),
            "--phase",
            "audit",
            "--model",
            "rwkv7-g1i-1.5b-20260805-ctx16384",
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert result.returncode == 42
    assert "global protocol gate failed" in result.stderr


def _frozen_descriptor(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(payload),
        "sha256": gate.hashlib.sha256(payload).hexdigest(),
    }


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
        elif path.stat().st_mode & 0o111:
            path.chmod(0o555)
        else:
            path.chmod(0o444)
    root.chmod(0o555)


def _frozen_runtime_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    source = tmp_path / "source"
    approval, _approval_descriptor = _approval_fixture(source)
    lock = source / "ops/g1i_strict46/protocol_gate.lock.json"
    _write_json(lock, {"fixture": True})
    lock.chmod(0o444)
    approval_payload = json.loads(approval.read_text())
    protocol_tree = approval_payload["protocol_contract"]["protocol_tree"]

    python_root = tmp_path / "sealed-python"
    executable = python_root / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o555)
    (python_root / "dependency.txt").write_text("dependency-v1\n")
    (python_root / "dependency.txt").chmod(0o444)
    (python_root / "bin").chmod(0o555)
    python_root.chmod(0o555)

    # Temporary-directory ancestors are outside this unit fixture.  Every
    # fixture-owned node is still checked for ownership and read-only mode.
    monkeypatch.setattr(gate, "_require_trusted_ancestor_chain", lambda *_args, **_kwargs: None)
    python_contract = gate.build_python_runtime_contract(
        python_root,
        executable,
        trusted_uid=os.getuid(),
    )

    frozen = tmp_path / "frozen-staging"
    for relative in protocol_tree["files"]:
        destination = frozen / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    frozen_approval = frozen / gate.APPROVAL_DIRECTORY / approval.name
    frozen_approval.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(approval, frozen_approval)
    frozen_lock = frozen / "ops/g1i_strict46/protocol_gate.lock.json"
    frozen_lock.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(lock, frozen_lock)
    dataset_file = frozen / "data/strict46.jsonl"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text('{"id":"fixture"}\n')
    dataset_descriptor = _frozen_descriptor(dataset_file, frozen)
    unsigned: dict[str, object] = {
        "schema_version": gate.FROZEN_RUNTIME_SCHEMA,
        "release_policy": gate._release_policy(),
        "protocol_tree_sha256": protocol_tree["tree_sha256"],
        "protocol_files": protocol_tree["files"],
        "approval": _frozen_descriptor(frozen_approval, frozen),
        "protocol_lock": _frozen_descriptor(frozen_lock, frozen),
        "datasets": {
            dataset: dataset_descriptor for dataset in sorted(gate.EXPECTED_DATASETS)
        },
        "support_files": [],
        "python_runtime": python_contract,
    }
    manifest = dict(unsigned)
    manifest["manifest_sha256"] = gate._canonical_json_sha256(unsigned)
    _write_json(frozen / gate.FROZEN_RUNTIME_MANIFEST, manifest)
    _seal_tree(frozen)
    published = tmp_path / str(manifest["manifest_sha256"])
    os.replace(frozen, published)
    frozen = published
    return source, approval, lock, frozen, python_contract


def test_frozen_runtime_survives_mutable_repo_replacement_and_pins_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, approval, lock, frozen, python_contract = _frozen_runtime_fixture(
        tmp_path,
        monkeypatch,
    )
    verified = gate.verify_frozen_runtime(
        source_repo=source,
        frozen_root=frozen,
        approval_path=approval,
        lock_path=lock,
        trusted_uid=os.getuid(),
    )
    assert str(verified["python_executable"]) == str(
        Path(str(python_contract["root"])) / str(python_contract["executable"])
    )

    approved_files = json.loads(approval.read_text())["protocol_contract"][
        "protocol_tree"
    ]["files"]
    relative = next(iter(approved_files))
    original_frozen_bytes = (frozen / relative).read_bytes()
    mutable = source / relative
    mutable.chmod(0o644)
    mutable.write_bytes(b"REPLACED AFTER GATE\n")
    # Published execution must not reopen the mutable, pre-publication audit
    # evidence.  Its descriptor is already covered by the self-digested
    # approval that the privileged publisher validated before sealing.
    (source / "evidence/merged.json.acceptance.json").unlink()
    assert (frozen / relative).read_bytes() == original_frozen_bytes
    # Production imports remain rooted at the independently-owned snapshot;
    # changing the development checkout cannot redirect them.
    gate.verify_frozen_runtime(
        source_repo=source,
        frozen_root=frozen,
        approval_path=approval,
        lock_path=lock,
        trusted_uid=os.getuid(),
    )
    frozen_approval = frozen / gate.APPROVAL_DIRECTORY / approval.name
    gate._verify_global_approval(
        frozen,
        frozen_approval,
        locked_descriptor=gate._path_descriptor(frozen, frozen_approval),
        trust_published_evidence=True,
    )


def test_frozen_runtime_rejects_frozen_code_or_dependency_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, approval, lock, frozen, python_contract = _frozen_runtime_fixture(
        tmp_path,
        monkeypatch,
    )
    protocol_files = json.loads(approval.read_text())["protocol_contract"]["protocol_tree"]["files"]
    frozen_source = frozen / next(iter(protocol_files))
    frozen_source.chmod(0o644)
    frozen_source.write_text("MUTATED\n")
    frozen_source.chmod(0o444)
    with pytest.raises(gate.ProtocolGateError, match="protocol file SHA mismatch"):
        gate.verify_frozen_runtime(
            source_repo=source,
            frozen_root=frozen,
            approval_path=approval,
            lock_path=lock,
            trusted_uid=os.getuid(),
        )

    # Rebuild a clean fixture so the dependency failure is independently
    # attributable to the sealed Python environment.
    other = tmp_path / "dependency-case"
    source, approval, lock, frozen, python_contract = _frozen_runtime_fixture(
        other,
        monkeypatch,
    )
    dependency = Path(str(python_contract["root"])) / "dependency.txt"
    dependency.chmod(0o644)
    dependency.write_text("dependency-v2\n")
    dependency.chmod(0o444)
    with pytest.raises(gate.ProtocolGateError, match="Python runtime changed"):
        gate.verify_frozen_runtime(
            source_repo=source,
            frozen_root=frozen,
            approval_path=approval,
            lock_path=lock,
            trusted_uid=os.getuid(),
        )


def test_frozen_runtime_rejects_secret_support_file_even_with_valid_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, approval, lock, frozen, _python_contract = _frozen_runtime_fixture(
        tmp_path,
        monkeypatch,
    )
    frozen.chmod(0o755)
    data_root = frozen / "data"
    data_root.chmod(0o755)
    secret = data_root / "credentials.json"
    sentinel = "DO_NOT_PUBLISH_SENTINEL_SECRET"
    secret.write_text(sentinel, encoding="utf-8")
    secret.chmod(0o444)
    manifest_path = frozen / gate.FROZEN_RUNTIME_MANIFEST
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["support_files"] = [_frozen_descriptor(secret, frozen)]
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = gate._canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)
    assert sentinel not in manifest_path.read_text(encoding="utf-8")
    _seal_tree(frozen)
    republished = frozen.parent / str(manifest["manifest_sha256"])
    os.replace(frozen, republished)

    with pytest.raises(gate.ProtocolGateError, match="secret-like path"):
        gate.verify_frozen_runtime(
            source_repo=source,
            frozen_root=republished,
            approval_path=approval,
            lock_path=lock,
            trusted_uid=os.getuid(),
        )
