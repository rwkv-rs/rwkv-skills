from __future__ import annotations

from src.bin import preflight_remote_eval
from src.bin.verify_remote_infer_swap import ProtocolVerification, RemoteInferSwapVerification


def test_preflight_defaults_to_vllm_protocol_smoke() -> None:
    args = preflight_remote_eval.parse_args(
        [
            "--infer-base-url",
            "http://127.0.0.1:8000",
            "--infer-model",
            "demo",
        ]
    )

    assert args.protocols == "vllm"


def test_protocol_smoke_uses_requested_vllm_protocol(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_verify_remote_infer_swap(**kwargs):
        captured.update(kwargs)
        return RemoteInferSwapVerification(
            base_url=str(kwargs["base_url"]),
            model=str(kwargs["model"]),
            prompt="prompt",
            max_tokens=8,
            batch_size=2,
            protocols=(
                ProtocolVerification(
                    protocol="vllm",
                    status="ok",
                    elapsed_s=0.1,
                    request_count=2,
                    output_count=2,
                    nonempty_output_count=2,
                    output_chars=4,
                ),
            ),
        )

    monkeypatch.setattr(preflight_remote_eval, "verify_remote_infer_swap", _fake_verify_remote_infer_swap)

    check = preflight_remote_eval._check_protocol_smoke(
        infer_base_url="http://127.0.0.1:8000",
        infer_model="demo",
        infer_api_key="",
        infer_timeout_s=1.0,
        protocols=("vllm",),
        batch_size=2,
        max_tokens=8,
    )

    assert check.ok
    assert captured["protocols"] == ("vllm",)
