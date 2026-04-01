from __future__ import annotations

from src.eval.benchmark_config import (
    config_path_for_benchmark,
    resolve_benchmark_model_config,
    resolve_sampling_config,
)
from src.infer.sampling import SamplingConfig


def test_config_path_for_benchmark_strips_split_suffix() -> None:
    path = config_path_for_benchmark("human_eval_plus_test")
    assert path.name == "human_eval_plus.toml"


def test_resolve_math_500_cot_config_merges_default_and_template() -> None:
    config = resolve_benchmark_model_config("math_500_test", "rwkv7-g1a-2.9b", stage="cot")

    assert config is not None
    assert config.pass_k == (1,)
    assert config.avg_k == (4,)
    assert config.report_pass_k == (1,)
    assert config.report_avg_k == (4,)
    assert config.sampling_overrides["max_generate_tokens"] == 4096
    assert config.sampling_overrides["top_k"] == 500
    assert config.sampling_overrides["temperature"] == 0.3
    assert config.sampling_overrides["stop_tokens"] == (0, 261, 24281)


def test_resolve_livecodebench_final_sampling_config_uses_code_template() -> None:
    config = resolve_sampling_config("livecodebench_test", "rwkv7-g1a-2.9b", stage="final")

    assert config is not None
    assert config.max_generate_tokens == 8192
    assert config.temperature == 0.6
    assert config.top_p == 0.6
    assert config.stop_tokens == (6884, 21214)
    assert config.pad_zero is True


def test_resolve_sampling_config_supports_fallback_templates() -> None:
    base = SamplingConfig(max_generate_tokens=128, temperature=1.0, top_p=1.0)

    config = resolve_sampling_config(
        "unknown_benchmark_test",
        "rwkv7-g1a-2.9b",
        stage="cot",
        base=base,
        fallback_templates="code_default",
    )

    assert config is not None
    assert config.max_generate_tokens == 1024
    assert config.temperature == 0.6
    assert config.top_p == 0.6
    assert config.stop_tokens == (0, 261, 6884, 21214, 24281)
