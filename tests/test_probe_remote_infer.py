from __future__ import annotations

from src.bin import probe_remote_infer
from src.eval.scheduler import remote_profiler
from src.infer.sampling import GenerationOutput


def test_probe_remote_infer_args_default_to_vllm_performance_probe() -> None:
    args = probe_remote_infer.parse_args(
        [
            "--infer-base-url",
            "http://127.0.0.1:8000",
            "--infer-model",
            "demo",
            "--warmup-requests",
            "3",
            "--max-p95-latency-s",
            "2.5",
            "--min-throughput-gain",
            "0.05",
        ]
    )

    assert args.infer_protocol == "vllm"
    assert args.warmup_requests == 3
    assert args.max_p95_latency_s == 2.5
    assert args.min_throughput_gain == 0.05


def test_remote_profiler_records_cold_warmup_and_latency_curve(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _FakeRemoteInferenceBackend:
        def __init__(self, config) -> None:  # noqa: ANN001
            self.config = config
            assert config.protocol == "vllm"

        def generate(
            self,
            prompts,
            *,
            sampling,
            batch_size,
            prompt_stop_suffixes=None,
            show_progress=True,
            **_kwargs,
        ):
            del sampling, prompt_stop_suffixes, show_progress
            calls.append({"prompt_count": len(prompts), "batch_size": batch_size})
            for index, _prompt in enumerate(prompts):
                if self.config.request_latency_callback is not None:
                    self.config.request_latency_callback(
                        self.config.chat_completions_url(),
                        0.01 + index * 0.001,
                        True,
                    )
            return [
                GenerationOutput(
                    prompt_index=index,
                    prompt=str(prompt),
                    token_ids=[],
                    text="ok",
                    finish_reason="stop_token",
                )
                for index, prompt in enumerate(prompts)
            ]

    monkeypatch.setattr(remote_profiler, "RemoteInferenceBackend", _FakeRemoteInferenceBackend)

    result = remote_profiler.probe_remote_inference(
        base_url="http://127.0.0.1:8000",
        model="demo",
        protocol="vllm",
        candidates=(1, 2),
        warmup_requests=2,
        max_p95_latency_s=1.0,
    )

    assert [call["prompt_count"] for call in calls] == [1, 2, 1, 2]
    assert result.protocol == "vllm"
    assert result.cold_first_request_latency_s == 0.01
    assert result.cold_first_request_error is None
    assert result.warmup_request_count == 2
    assert result.warmup_error is None
    assert [point.concurrency for point in result.points] == [1, 2]
    assert all(point.p50_latency_s is not None for point in result.points)
    assert all(point.p95_latency_s is not None for point in result.points)
    assert result.suggested_infer_max_workers in {1, 2}
    assert result.suggested_remote_batch_size == result.suggested_infer_max_workers
