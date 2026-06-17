from __future__ import annotations

from src.bin import audit_infer_swap_readiness, run_infer_swap_eval


def test_audit_defaults_to_lightning_protocol() -> None:
    args = audit_infer_swap_readiness.parse_args([])

    assert args.infer_protocol == "lightning"
    assert args.infer_seed_policy == "preserve"


def test_audit_protocol_smoke_requires_expected_lightning_only() -> None:
    protocols = [
        {
            "protocol": "lightning",
            "ok": True,
            "request_count": run_infer_swap_eval.DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE,
            "nonempty_output_count": run_infer_swap_eval.DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE,
        }
    ]

    assert audit_infer_swap_readiness.validate_protocol_smoke(protocols) == ()


def test_audit_probe_payload_accepts_lightning_protocol() -> None:
    args = audit_infer_swap_readiness.parse_args(
        [
            "--infer-model",
            "demo",
            "--infer-max-workers",
            "4",
            "--remote-batch-size",
            "4",
        ]
    )
    payload = {
        "model": "demo",
        "protocol": "lightning",
        "largest_successful_concurrency": 4,
        "gpu_full_concurrency": 4,
    }

    assert audit_infer_swap_readiness.validate_probe_payload(
        payload,
        args,
        expected_workers=4,
        expected_batch_size=4,
    ) == ()
