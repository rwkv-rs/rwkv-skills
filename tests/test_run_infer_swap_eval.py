from __future__ import annotations

from src.bin import run_infer_swap_eval, validate_infer_swap_phases


def test_run_infer_swap_eval_builds_lightning_scheduler_args_by_default() -> None:
    args = run_infer_swap_eval.parse_args(
        [
            "--infer-max-workers",
            "8",
            "--remote-batch-size",
            "8",
            "--only-datasets",
            "mcp_bench_test",
        ]
    )

    scheduler_args = run_infer_swap_eval.build_scheduler_args(args)

    assert _flag_value(scheduler_args, "--infer-protocol") == "lightning"
    assert _flag_value(scheduler_args, "--infer-seed-policy") == "preserve"
    assert _flag_value(scheduler_args, "--infer-max-workers") == "8"
    assert _flag_value(scheduler_args, "--remote-batch-size") == "8"


def test_phase_gate_validation_accepts_lightning_probe_and_protocol_smoke() -> None:
    dispatch_args = run_infer_swap_eval.parse_args(
        [
            "--infer-max-workers",
            "4",
            "--remote-batch-size",
            "4",
            "--only-datasets",
            "mcp_bench_test",
        ]
    )
    source_manifest = validate_infer_swap_phases.build_source_manifest()
    payload = {
        "schema_version": run_infer_swap_eval.MIN_PHASE_GATE_SCHEMA_VERSION,
        "generated_at_utc": "2026-06-07T00:00:00+00:00",
        "ok": True,
        "required_phase_names": list(run_infer_swap_eval.REQUIRED_PHASE_GATE_PHASES_FOR_DISPATCH),
        "source_manifest": source_manifest,
        "source_digest": validate_infer_swap_phases.source_digest(source_manifest),
        "phases": [
            _phase(name)
            for name in run_infer_swap_eval.REQUIRED_PHASE_GATE_PHASES_FOR_DISPATCH
        ],
    }
    by_name = {phase["name"]: phase for phase in payload["phases"]}
    by_name["readiness_json"]["details"] = {
        "ready_to_dispatch": True,
        "queue_pending_count": 1,
        "expected_queue_count": 1,
        "expected_infer_max_workers": 4,
        "expected_remote_batch_size": 4,
        "probe_gpu_full_concurrency": 4,
        "probe_largest_successful_concurrency": 4,
        "probe_model": run_infer_swap_eval.DEFAULT_INFER_MODEL,
        "protocol_smoke_ok": True,
        "protocol_smoke_protocols": [
            {
                "protocol": "lightning",
                "ok": True,
                "request_count": run_infer_swap_eval.DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE,
                "nonempty_output_count": run_infer_swap_eval.DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE,
            }
        ],
    }
    by_name["probe_json"]["details"] = {
        "model": run_infer_swap_eval.DEFAULT_INFER_MODEL,
        "protocol": "lightning",
        "required_concurrency": 4,
        "gpu_full_concurrency": 4,
        "largest_successful_concurrency": 4,
    }
    by_name["summary_json"]["details"] = {
        "total_count": 1,
        "datasets": ["mcp_bench_test"],
    }

    assert run_infer_swap_eval.validate_phase_gate_payload(payload, dispatch_args=dispatch_args) == ()


def _phase(name: str) -> dict[str, object]:
    return {"name": name, "ok": True, "errors": [], "details": {}}


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]
