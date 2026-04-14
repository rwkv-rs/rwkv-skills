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
  pad_zero, no_penalty_token_ids
- Legacy evaluation fields: pass_k, avg_k, report_pass_k, report_avg_k

Notes:
- CLI flags override config values.
- Scheduler-facing benchmark runs now use the auto avg@k execution plan:
  sample exactly 5000 attempts worth of work, by deterministic downsampling when dataset size > 5000,
  or by repeating the full dataset until reaching 5000 effective samples when dataset size <= 5000.
- Current benchmark jobs do not report pass@k; existing pass_k/report_pass_k TOML fields are retained only for
  legacy scripts and compatibility.
- For benchmark jobs, zeroshot / cot_mode selection is controlled by the evaluator entrypoint rather than TOML.
- llm_judge stays in evaluator code or CLI flags; it is not read from TOML.
- free_response applies sampling overrides to CoT generation.
- livecodebench applies sampling overrides to both CoT and final stages.
- livecodebench defaults to full_code_* templates when configs/livecodebench.toml is missing.
- Model names match case-insensitively; safe_slug normalization is supported.
- Use template = "name" or templates = ["base", "override"] to merge templates before overrides.
- When both [default] and [cot]/[final] exist, values are merged in order: default -> stage -> model.
- When benchmark config is missing, callers may supply fallback_templates to use templates from configs/_templates.toml.

Unified run configs for `python -m src.main` live under `configs/run/`:

- Path: `configs/run/<benchmark>.toml`
- Use `python -m src.main --benchmark <benchmark>` to resolve that file automatically
- Use `python -m src.main --config <name>` to resolve `configs/run/<name>.toml`
- These run configs are separate from the sampling configs in `configs/<benchmark>.toml`
