from __future__ import annotations

from src.bin import draft_infer_swap_speedup_doc, run_infer_swap_eval


def test_speedup_doc_readiness_validation_accepts_vllm_protocol() -> None:
    readiness = {
        "ready_to_dispatch": True,
        "errors": [],
        "queue_pending_count": 1,
        "expected_queue_count": 1,
        "probe_model": "demo",
        "probe_protocol": "vllm",
        "expected_infer_max_workers": 4,
        "expected_remote_batch_size": 4,
        "probe_gpu_full_concurrency": 4,
        "probe_largest_successful_concurrency": 4,
        "protocol_smoke_protocols": [
            {
                "protocol": "vllm",
                "ok": True,
                "request_count": run_infer_swap_eval.DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE,
                "nonempty_output_count": run_infer_swap_eval.DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE,
            }
        ],
    }
    summary = {"model": "demo", "total_count": 1}
    probe = {
        "model": "demo",
        "gpu_full_concurrency": 4,
        "largest_successful_concurrency": 4,
    }
    launch_bundle = {"launch_parameters": {"infer_max_workers": 4, "remote_batch_size": 4}}

    ok, reason = draft_infer_swap_speedup_doc.validate_readiness_evidence(
        summary=summary,
        probe=probe,
        launch_bundle=launch_bundle,
        readiness=readiness,
    )

    assert ok, reason
