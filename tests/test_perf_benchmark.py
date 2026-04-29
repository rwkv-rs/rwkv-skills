from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from src.bin import run_perf_benchmark as perf_cli
from src.eval.performance import config as perf_config_module
from src.eval.performance.runner import ServiceBenchmarkConfig, run_service_benchmark
from src.eval.performance.schema import PerfBenchmarkResult
from src.eval.performance.service_client import ServiceRequestResult
from src.eval.performance.vllm_launcher import VllmLaunchConfig, _explain_launch_failure, _normalize_vllm_args


@dataclass(slots=True)
class _DummyTokenizer:
    label: str = "dummy"

    def encode(self, text: str) -> list[int]:
        return [1 for _ in text.split()] or [1]

    def decode(self, token_ids) -> str:
        return " ".join("tok" for _ in token_ids)


class _FakeClient:
    def benchmark_many(
        self,
        *,
        prompts,
        max_tokens,
        temperature,
        top_p,
        base_seed,
        max_workers,
    ):
        return [
            ServiceRequestResult(
                request_index=index,
                text="tok tok",
                ttft_s=0.1 + index * 0.01,
                e2el_s=0.3 + index * 0.01,
                finish_reason="stop",
            )
            for index, _ in enumerate(prompts)
        ]


def test_run_service_benchmark_emits_concurrency_and_batch_size_points(monkeypatch) -> None:
    monkeypatch.setattr("src.eval.performance.runner._build_client", lambda config: _FakeClient())
    monkeypatch.setattr(
        "src.eval.performance.runner.build_prompt_for_target_tokens",
        lambda tokenizer, ctx_len: ("tok " * ctx_len, ctx_len),
    )

    result = run_service_benchmark(
        ServiceBenchmarkConfig(
            base_url="http://127.0.0.1:8000",
            model="demo-model",
            api_key="",
            protocol="openai-chat",
            stack_name="demo",
            engine_name="vllm",
            precision="bf16",
            tokenizer_label="dummy",
            tokenizer=_DummyTokenizer(),
            ctx_lens=(128,),
            concurrency_levels=(1, 2),
            batch_sizes=(4,),
            output_tokens=32,
            warmup_runs=0,
            measure_runs=1,
            temperature=1.0,
            top_p=1.0,
            timeout_s=30.0,
            base_seed=None,
            gpu_index=None,
            service_metadata={},
            hardware_metadata={},
        )
    )

    assert isinstance(result, PerfBenchmarkResult)
    assert result.schema_version == 2
    assert result.workload["point_kinds"] == ["concurrency", "batch_size"]
    assert [(point.point_kind, point.load_value) for point in result.points] == [
        ("concurrency", 1),
        ("concurrency", 2),
        ("batch_size", 4),
    ]
    assert result.points[0].concurrency == 1
    assert result.points[0].batch_size is None
    assert result.points[-1].concurrency is None
    assert result.points[-1].batch_size == 4


def test_vllm_launch_command_includes_optional_flags() -> None:
    config = VllmLaunchConfig(
        model="Qwen/Qwen2.5-7B-Instruct",
        host="127.0.0.1",
        port=9000,
        dtype="bfloat16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        max_num_seqs=32,
        max_num_batched_tokens=16384,
        trust_remote_code=True,
        extra_args=("--disable-log-requests",),
    )

    command = config.build_command()

    assert command[:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--model" in command
    assert "--dtype" in command
    assert "--tensor-parallel-size" in command
    assert "--gpu-memory-utilization" in command
    assert "--max-model-len" in command
    assert "--max-num-seqs" in command
    assert "--max-num-batched-tokens" in command
    assert "--trust-remote-code" in command
    assert "--no-enable-log-requests" in command


def test_vllm_launch_command_prefers_explicit_python_executable() -> None:
    config = VllmLaunchConfig(
        model="Qwen/Qwen2-1.5B-Instruct",
        python_executable="/tmp/vllm-env/bin/python",
    )

    command = config.build_command()

    assert command[:3] == ["/tmp/vllm-env/bin/python", "-m", "vllm.entrypoints.openai.api_server"]


def test_build_vllm_launch_config_uses_current_interpreter_for_default_command() -> None:
    args = perf_cli.parse_args(["--launch-vllm", "--model", "Qwen/Qwen2-1.5B-Instruct"])

    config = perf_cli._build_vllm_launch_config(args)

    assert config.python_executable == sys.executable
    assert config.command == ()


def test_resolve_perf_config_path_accepts_named_config(monkeypatch, tmp_path: Path) -> None:
    perf_root = tmp_path / "configs" / "perf"
    perf_root.mkdir(parents=True)
    config_path = perf_root / "qwen2_1_5b_instruct.toml"
    config_path.write_text("[service]\nmodel='demo'\n", encoding="utf-8")
    monkeypatch.setattr(perf_config_module, "PERF_CONFIG_ROOT", perf_root)

    resolved = perf_config_module.resolve_perf_config_path("qwen2_1_5b_instruct")

    assert resolved == config_path.resolve()


def test_parse_args_loads_perf_config_defaults(monkeypatch, tmp_path: Path) -> None:
    perf_root = tmp_path / "configs" / "perf"
    perf_root.mkdir(parents=True)
    config_path = perf_root / "qwen2_5_1_5b_instruct.toml"
    config_path.write_text(
        """
[service]
model = "Qwen/Qwen2.5-1.5B-Instruct"
engine_name = "vllm"
precision = "fp16"

[tokenizer]
type = "hf"

[workload]
ctx_lens = [512, 1024]
concurrency_levels = [1, 2]
batch_sizes = [1, 4]

[vllm]
launch = true
python_executable = "/tmp/vllm-env/bin/python"
port = 8001
dtype = "half"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(perf_config_module, "PERF_CONFIG_ROOT", perf_root)

    args = perf_cli.parse_args(["--config", "qwen2_5_1_5b_instruct"])

    assert args.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert args.engine_name == "vllm"
    assert args.precision == "fp16"
    assert args.tokenizer_type == "hf"
    assert args.ctx_lens == "512,1024"
    assert args.concurrency_levels == "1,2"
    assert args.batch_sizes == "1,4"
    assert args.launch_vllm is True
    assert args.vllm_python == "/tmp/vllm-env/bin/python"
    assert args.vllm_port == 8001
    assert args.vllm_dtype == "half"


def test_explain_launch_failure_surfaces_torch_vllm_abi_mismatch() -> None:
    message = _explain_launch_failure(
        "ImportError: ... undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb"
    )

    assert "PyTorch/CUDA 二进制不兼容" in message


def test_normalize_vllm_args_maps_legacy_disable_log_requests() -> None:
    normalized = _normalize_vllm_args(("--disable-log-requests", "--foo"))

    assert normalized == ("--no-enable-log-requests", "--foo")
