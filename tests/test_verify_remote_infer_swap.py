from __future__ import annotations

from src.bin import verify_remote_infer_swap


def test_verify_remote_infer_swap_defaults_to_vllm_protocol() -> None:
    args = verify_remote_infer_swap.parse_args(
        [
            "--infer-base-url",
            "http://127.0.0.1:8000",
            "--infer-model",
            "demo",
        ]
    )

    assert args.protocols == "vllm"
    assert verify_remote_infer_swap.DEFAULT_PROTOCOLS == ("vllm",)
    assert verify_remote_infer_swap._normalize_protocols([args.protocols]) == ("vllm",)
