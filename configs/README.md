# Benchmark Configs

Place per-benchmark TOML files under this folder:

- Path: configs/<benchmark>.toml
- Benchmark name: dataset slug without split suffix (e.g. math_500_test -> math_500)
- Each file defines tables keyed by model name.
- Optional [default] table applies to all models; a model table overrides it.
- Templates live in configs/_templates.toml; each top-level table is a template.
- Stage-specific configs can use [cot] / [final] tables (optionally with nested model tables).

Supported keys per model table:
- SamplingConfig fields: max_generate_tokens, temperature, top_k, top_p,
  alpha_presence, alpha_frequency, alpha_decay, stop_tokens, ban_tokens,
  pad_zero, no_penalty_token_ids, sample_mode, noise
- Evaluation fields: pass_k, avg_k, report_pass_k, report_avg_k (free-response only)

Notes:
- CLI flags override config values.
- pass_k / avg_k (free-response) can be configured here; CLI flags override them.
- llm_judge stays in evaluator code or CLI flags; it is not read from TOML.
- free_response applies sampling overrides to CoT generation.
- livecodebench applies sampling overrides to both CoT and final stages.
- livecodebench defaults to full_code_* templates when configs/livecodebench.toml is missing.
- Model names match case-insensitively; safe_slug normalization is supported.
- Use template = "name" or templates = ["base", "override"] to merge templates before overrides.
- When both [default] and [cot]/[final] exist, values are merged in order: default -> stage -> model.
- When benchmark config is missing, callers may supply fallback_templates to use templates from configs/_templates.toml.
