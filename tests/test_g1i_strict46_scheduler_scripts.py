from __future__ import annotations

import os
import json
import shlex
import subprocess
from pathlib import Path

import pytest

from ops.g1i_strict46 import require_global_protocol_gate as protocol_gate
from ops.g1i_strict46 import runtime_state


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops" / "g1i_strict46"

KNOWLEDGE = {
    "arc_easy",
    "mmlu",
    "openbookqa",
    "cmmlu",
    "commonsense_qa",
    "ceval",
    "truthfulqa_mc1",
    "mmlu_pro",
    "hellaswag",
    "mmlu_redux",
    "winogrande",
    "agieval_mcq",
    "mmlu_sr_question_and_answer",
    "bbh_mcq",
    "kmmlu",
    "gpqa_main",
    "gpqa_extended",
    "medqa",
    "gpqa_diamond",
    "medmcqa",
    "arc_challenge",
}
MATH = {
    "aime24",
    "aime25",
    "amc23",
    "answer_judge",
    "beyond_aime",
    "brumo25",
    "comp_math_24_25",
    "gaokao2023en",
    "gsm8k",
    "hmmt_feb25",
    "math_500",
    "math_odyssey",
    "minerva_math",
    "olympiadbench",
    "simpleqa",
    "svamp",
}
CODING = {
    "human_eval",
    "human_eval_cn",
    "human_eval_fix",
    "human_eval_plus",
    "mbpp",
    "mbpp_plus",
    "livecodebench",
}
INSTRUCTION = {"ifeval", "ifbench"}

STRICT_JOBS = {
    "multi_choice_plain_naive",
    "free_response_naive",
    "free_response_judge_naive",
    "code_human_eval_naive",
    "code_mbpp_naive",
    "code_livecodebench_plain_naive",
    "instruction_following_naive",
}


def _section(path: Path, start: str, end: str) -> list[str]:
    tokens = [token for token in shlex.split(path.read_text()) if token.strip()]
    return tokens[tokens.index(start) + 1 : tokens.index(end)]


def test_full_model_queue_is_exactly_strict46() -> None:
    script = OPS / "run_model.sh"
    jobs = set(_section(script, "--only-jobs", "--only-datasets"))
    datasets = _section(script, "--only-datasets", "--infer-base-url")

    assert jobs == STRICT_JOBS
    assert len(datasets) == len(set(datasets)) == 46
    assert set(datasets) == KNOWLEDGE | MATH | CODING | INSTRUCTION
    assert "multi_choice_cot_naive" not in script.read_text()
    text = script.read_text()
    # One final gate is sufficient and safer: it performs endpoint and live
    # runtime attestation after the frozen re-exec, immediately before the
    # scheduler process replaces the launcher.
    assert text.count('"$python" -I "$gate"') == 1
    assert "--print-frozen-python" in text
    assert text.count("--require-current-python") == 1
    assert 'exec "$python" -m src.eval.scheduler.cli dispatch' in text
    assert ".venv/bin/python" not in text
    assert "/home/rwkv/chase/rwkv-skills" not in text
    assert "RWKV_STRICT_CONTROL_REPO" not in text
    assert 'state_root=$("$python" -I' in text
    assert '"$state_root/logs/scheduler"' in text
    assert '"$state_root/logs/pids"' in text
    assert '"$state_root/logs/runs"' in text


def test_runtime_state_rejects_traversal_symlinks_and_non_private_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.geteuid() == 0:
        pytest.skip("fixture ownership requires a non-root test runner")
    parent = tmp_path / "state"
    parent.mkdir(mode=0o700)
    monkeypatch.setattr(runtime_state, "STATE_PARENT", parent)
    monkeypatch.setattr(
        runtime_state,
        "_require_trusted_parent_ancestors",
        lambda _parent: None,
    )

    state = runtime_state.prepare_run_state("strict46-run-1", create=True)
    assert state == parent / "strict46-run-1"
    assert state.stat().st_mode & 0o777 == 0o700
    with pytest.raises(runtime_state.StateError, match="unsafe"):
        runtime_state.prepare_run_state("../../home/chase", create=True)

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = parent / "linked-run"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(runtime_state.StateError, match="symlink"):
        runtime_state.prepare_run_state("linked-run", create=False)

    parent.chmod(0o770)
    with pytest.raises(runtime_state.StateError, match="private"):
        runtime_state.prepare_run_state("strict46-run-1", create=False)


def test_29_72_followup_is_exactly_the_13_unlaunched_strict_cells() -> None:
    script = OPS / "run_followup_29_72.sh"
    jobs = set(_section(script, "--only-jobs", "--only-datasets"))
    datasets = _section(script, "--only-datasets", "--infer-base-url")
    expected = {
        "answer_judge",
        "comp_math_24_25",
        "gaokao2023en",
        "math_odyssey",
        "minerva_math",
        "olympiadbench",
        "simpleqa",
        "svamp",
        "human_eval_cn",
        "human_eval_fix",
        "human_eval_plus",
        "mbpp_plus",
        "ifbench",
    }

    assert jobs == STRICT_JOBS - {
        "multi_choice_plain_naive",
        "code_livecodebench_plain_naive",
    }
    assert len(datasets) == len(set(datasets)) == 13
    assert set(datasets) == expected
    assert "multi_choice_cot_naive" not in script.read_text()
    tokens = shlex.split(script.read_text())
    assert tokens[tokens.index("--run-mode") + 1] == "missing"


def test_model_handoffs_are_unprivileged_content_addressed_requests() -> None:
    for name in ("wait_157_1p5.sh", "wait_8222_13p3.sh"):
        text = (OPS / name).read_text()
        assert "RWKV_STRICT_WAITER_REEXEC" in text
        assert "RWKV_STRICT_FROZEN_RUNTIME" in text
        assert "RWKV_STRICT_RUN_ID" in text
        assert "RWKV_STRICT_STATE_ROOT" in text
        assert ".venv/bin/python" not in text
        assert "ops.g1i_strict46.handoff_request" in text
        assert "exit 75" in text
        assert "systemctl" not in text
        assert "systemd-run" not in text
        assert "/usr/bin/ssh" not in text
        assert "run_model.sh" not in text

    local_text = (OPS / "wait_157_1p5.sh").read_text()
    assert "ensure_model_complete.sh" in local_text
    assert "240 60 3" in local_text
    assert local_text.index("ensure_model_complete.sh") < local_text.index(
        "handoff_idle_guard.sh"
    )


def test_8222_handoff_request_binds_exact_route_model_gpu_and_units() -> None:
    waiter = (OPS / "wait_8222_13p3.sh").read_text()
    request = (OPS / "handoff_request.py").read_text()

    assert "--attest-runtime-host-local" in waiter
    assert "--transition 8222-7p2-to-13p3" in waiter
    for expected in (
        '"host": "8222"',
        '"gpu": 2',
        '"port": 18074',
        '"forwarded_host": "157"',
        '"forwarded_port": 29574',
        '"current_model": "rwkv7-g1i-7.2b-20260805-ctx16384"',
        '"next_model": "rwkv7-g1i-13.3b-20260805-ctx16384"',
        '"current_inference_unit": "rwkv-g1i-7p2b-gpu2-16k-c640.service"',
        '"next_inference_unit": "rwkv-g1i-13p3b-gpu2-16k-c640.service"',
        '"next_scheduler_unit": "rwkv-g1i-strict46-133-raw-20260806.service"',
    ):
        assert expected in request


def test_endpoint_model_identity_rejects_wrong_or_multiple_models() -> None:
    expected = "rwkv7-g1i-1.5b-20260805-ctx16384"
    assert protocol_gate.validate_single_model_response(
        {"data": [{"id": expected}]}, expected
    ) == expected

    with pytest.raises(protocol_gate.ProtocolGateError, match="model mismatch"):
        protocol_gate.validate_single_model_response(
            {"data": [{"id": "wrong-model"}]}, expected
        )
    with pytest.raises(protocol_gate.ProtocolGateError, match="exactly one"):
        protocol_gate.validate_single_model_response(
            {"data": [{"id": expected}, {"id": "another-model"}]}, expected
        )


def test_protocol_lock_recheck_detects_toctou_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "protocol.py"
    source.write_text("PROTOCOL = 1\n")
    monkeypatch.setattr(
        protocol_gate,
        "required_protocol_paths",
        lambda _repo: ("protocol.py",),
    )
    lock = tmp_path / "protocol.lock.json"
    lock.write_text(json.dumps(protocol_gate.current_lock_payload(tmp_path)))

    protocol_gate._verify_lock(tmp_path, lock)
    source.write_text("PROTOCOL = 2\n")
    with pytest.raises(protocol_gate.ProtocolGateError, match="hash drift"):
        protocol_gate._verify_lock(tmp_path, lock)


def test_handoff_gate_order_and_reserved_8222_gpu3() -> None:
    waiter_157 = (OPS / "wait_157_1p5.sh").read_text()
    assert waiter_157.index("--phase audit") < waiter_157.index("ensure_model_complete.sh")
    assert waiter_157.index("ensure_model_complete.sh") < waiter_157.index(
        "handoff_idle_guard.sh"
    )
    assert waiter_157.index("--phase attest") < waiter_157.index("--phase launch")
    assert waiter_157.index("--phase launch") < waiter_157.index("handoff_request")

    waiter_8222 = (OPS / "wait_8222_13p3.sh").read_text()
    assert waiter_8222.index("--phase attest") < waiter_8222.index(
        "handoff_idle_guard.sh"
    )
    assert waiter_8222.index("--phase launch") < waiter_8222.index("handoff_request")
    request = (OPS / "handoff_request.py").read_text()
    assert '"gpu": 2' in request
    for forbidden in ("CUDA_VISIBLE_DEVICES=3", "port=18073", "GPU3", "gpu3"):
        assert forbidden not in waiter_8222


def test_legacy_user_systemd_chains_are_not_in_the_frozen_inventory() -> None:
    inventory = set(protocol_gate.protocol_inventory_paths(ROOT))
    for name in (
        "chain_157_g1h_15_29.sh",
        "chain_8222_g1h_133_72.sh",
        "wait_and_recover_math.sh",
        "wait_math_wave_then_followup.sh",
    ):
        assert f"ops/g1i_strict46/{name}" not in inventory


def test_shared_model_gate_has_long_commit_window_and_targeted_recovery() -> None:
    wait_text = (OPS / "wait_for_model_audit.sh").read_text()
    recovery_text = (OPS / "ensure_model_complete.sh").read_text()

    assert "attempts=${2:-240}" in wait_text
    assert "interval_s=${3:-60}" in wait_text
    assert '--require-model-complete "$model"' in wait_text
    assert "g1i_strict46_gate_${safe_model}.json" in wait_text
    assert "audit_attempts=${5:-240}" in recovery_text
    assert "audit_interval_s=${6:-60}" in recovery_text
    assert "recovery_rounds=${7:-3}" in recovery_text
    assert "run_audit_missing.py" in recovery_text
    assert "--audit-output" in recovery_text
    assert "g1i_strict46_recovery_${safe_model}_${recovery_attempt}.json" in recovery_text
    assert "flock 9" in recovery_text
    assert "systemctl --user" not in recovery_text
    assert "systemd-run --user" not in recovery_text
    assert '"$python" "$runtime_repo/ops/g1i_strict46/run_audit_missing.py"' in recovery_text
    assert recovery_text.index("if wait_for_audit") < recovery_text.index(
        "for recovery_attempt"
    )


def _write_fake_command(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def _idle_guard_env(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    _write_fake_command(
        fake_bin / "systemctl",
        """
if [[ "$*" == *"list-units"* ]]; then
  printf '%s' "${FAKE_UNIT_OUTPUT:-}"
elif [[ "$*" == *"LoadState"* ]]; then
  printf '%s\\n' "${FAKE_LOAD_STATE:-loaded}"
elif [[ "$*" == *"ActiveState"* ]]; then
  printf '%s\\n' "${FAKE_ACTIVE_STATE:-inactive}"
elif [[ "$*" == *"Result"* ]]; then
  printf '%s\\n' "${FAKE_RESULT:-success}"
else
  exit 64
fi
""",
    )
    _write_fake_command(
        fake_bin / "ps",
        """
if [[ -n "${FAKE_PS_COUNTER:-}" ]]; then
  count=0
  [[ -f "$FAKE_PS_COUNTER" ]] && count=$(<"$FAKE_PS_COUNTER")
  count=$((count + 1))
  printf '%s\\n' "$count" >"$FAKE_PS_COUNTER"
  if (( count <= ${FAKE_PS_BUSY_UNTIL:-0} )); then
    printf '%s\\n' "${FAKE_PS_LINE:-}"
  fi
else
  printf '%s' "${FAKE_PS_OUTPUT:-}"
fi
""",
    )
    _write_fake_command(
        fake_bin / "ss",
        "printf '%s' \"${FAKE_SS_OUTPUT:-}\"\n",
    )
    _write_fake_command(fake_bin / "sleep", ":\n")
    return {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}


def _run_idle_guard(
    tmp_path: Path,
    *,
    extra_env: dict[str, str] | None = None,
    scheduler: str = "test-scheduler.service",
    recovery_prefix: str = "test-recovery",
    observations: str = "3",
) -> subprocess.CompletedProcess[str]:
    env = _idle_guard_env(tmp_path)
    env.update(extra_env or {})
    return subprocess.run(
        [
            "bash",
            str(OPS / "handoff_idle_guard.sh"),
            "test-model",
            "19439",
            scheduler,
            recovery_prefix,
            "2",
            "0",
            observations,
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_idle_guard_requires_two_consecutive_clean_observations(tmp_path: Path) -> None:
    counter = tmp_path / "ps-count"
    result = _run_idle_guard(
        tmp_path,
        extra_env={
            "FAKE_PS_COUNTER": str(counter),
            "FAKE_PS_BUSY_UNTIL": "1",
            "FAKE_PS_LINE": (
                "123 python -m src.eval.tasks.maths.runner "
                "--infer-model test-model --infer-base-url http://127.0.0.1:19439/v1"
            ),
        },
    )

    assert result.returncode == 0, result.stderr
    assert counter.read_text().strip() == "3"
    assert "observation 2/3 idle (1/2)" in result.stdout
    assert "observation 3/3 idle (2/2)" in result.stdout


def test_idle_guard_fails_closed_for_scheduler_or_recovery_activity(
    tmp_path: Path,
) -> None:
    scheduler_result = _run_idle_guard(
        tmp_path,
        extra_env={"FAKE_ACTIVE_STATE": "active"},
        observations="2",
    )
    assert scheduler_result.returncode == 25
    assert "scheduler not safely complete" in scheduler_result.stderr

    recovery_result = _run_idle_guard(
        tmp_path,
        extra_env={
            "FAKE_UNIT_OUTPUT": (
                "test-recovery-1.service loaded active running recovery lane"
            )
        },
        observations="2",
    )
    assert recovery_result.returncode == 25
    assert "active recovery unit" in recovery_result.stderr


def test_idle_guard_allows_collected_scheduler_only_after_explicit_opt_in(
    tmp_path: Path,
) -> None:
    fail_closed = _run_idle_guard(
        tmp_path,
        extra_env={"FAKE_LOAD_STATE": "not-found"},
        observations="2",
    )
    assert fail_closed.returncode == 25
    assert "scheduler not loaded" in fail_closed.stderr

    explicitly_verified = _run_idle_guard(
        tmp_path,
        extra_env={
            "FAKE_LOAD_STATE": "not-found",
            "HANDOFF_ALLOW_COLLECTED_SCHEDULER": "1",
        },
        observations="2",
    )
    assert explicitly_verified.returncode == 0, explicitly_verified.stderr
    assert "handoff idle guard passed" in explicitly_verified.stdout


def test_idle_guard_fails_closed_for_runner_or_established_connection(
    tmp_path: Path,
) -> None:
    runner_result = _run_idle_guard(
        tmp_path,
        extra_env={
            "FAKE_PS_OUTPUT": (
                "321 python -m src.eval.tasks.knowledge.runner "
                "--infer-model test-model --infer-base-url http://127.0.0.1:19439/v1\\n"
            )
        },
        observations="2",
    )
    assert runner_result.returncode == 25
    assert "matching evaluation process" in runner_result.stderr

    socket_result = _run_idle_guard(
        tmp_path,
        extra_env={
            "FAKE_SS_OUTPUT": (
                "0 0 127.0.0.1:19439 127.0.0.1:54321 users:((python,pid=99))\\n"
            )
        },
        observations="2",
    )
    assert socket_result.returncode == 25
    assert "established target-port connection" in socket_result.stderr
