from __future__ import annotations

from src.bin import run_infer_swap_eval, validate_infer_swap_phases


def test_validate_readiness_payload_accepts_vllm_probe_and_smoke() -> None:
    payload = {
        "ready_to_dispatch": True,
        "errors": [],
        "queue_pending_count": 1,
        "expected_queue_count": 1,
        "probe_model": run_infer_swap_eval.DEFAULT_INFER_MODEL,
        "probe_protocol": "vllm",
        "protocol_smoke_ok": True,
        "protocol_smoke_protocols": [
            {
                "protocol": "vllm",
                "ok": True,
                "request_count": run_infer_swap_eval.DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE,
                "nonempty_output_count": run_infer_swap_eval.DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE,
            }
        ],
        "expected_infer_max_workers": 4,
        "expected_remote_batch_size": 4,
        "probe_gpu_full_concurrency": 4,
        "probe_largest_successful_concurrency": 4,
    }

    assert validate_infer_swap_phases.validate_readiness_payload(payload) == ()
