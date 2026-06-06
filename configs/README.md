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
- Evaluation fields: pass_k, avg_k, report_pass_k, report_avg_k, max_samples (free_response and multi_choice_cot; max_samples is also read by direct/code/instruction evaluators)
- Prompt fields: cot_prompt_template, final_prompt_template, judge_prompt_template
- RWKV multi-turn agent plugin field: agent_plugin_enabled
- Function-calling agent tool routing fields: tool_router_mode (`off` or `lexical`), tool_router_max_tools, tool_router_trigger_tool_count, tool_router_trigger_catalog_chars, tool_router_context_chars, tool_router_description_chars
- Function-calling long-context routing fields: long_context_router_mode (`off` or `lexical`), long_context_min_chars, long_context_chunk_chars, long_context_overlap_lines, long_context_max_evidence_chunks, long_context_max_evidence_chars, long_context_query_chars. Legacy sibling-runner aliases are accepted: long_doc_mode, long_doc_min_chars, long_doc_max_chars, long_doc_overlap_lines, long_doc_max_evidence_chunks, long_doc_max_evidence_chars, long_doc_query_chars.
- Function-calling agent runner limit fields: history_max_chars, prompt_max_chars, max_steps, max_tool_errors, decision_max_tokens, max_repeated_tool_calls
- TAU attempt scheduler fields: tau_sample_workers, tau_attempt_retries, tau_judge_concurrency
- TAU official user/judge runtime fields: user_model, user_api_key, user_base_url, judge_model, judge_api_key, judge_base_url

Notes:
- CLI flags override config values.
- pass_k / avg_k can be configured for CoT evaluators (free_response / free_response_judge / multi_choice_cot); CLI flags override them.
- avg_k / report_avg_k accept either integers (e.g. `16`) or ratios in `(0, 1)` (e.g. `0.2`).
- When `avg_k` is a ratio, the evaluator uses the first `ceil(ratio * repeats)` samples available for each problem.
- max_samples is used as the default sample limit when CLI `--max-samples` is omitted.
- Most llm_judge settings stay in evaluator code or CLI flags. BrowseComp-Plus is the exception: its OpenAI-compatible judge is read from `[default.browsecomp_plus_judge]`.
- `agent_plugin_enabled = true` is the preferred switch for long-context multi-turn agent benchmarks. It defaults both tool routing and long-context routing to lexical for the agent evaluator. Use the existing `tool_router_mode = "off"` or `long_context_router_mode = "off"` fields only when one side must be disabled.
- Official TAU2/TAU3 records use the dedicated TAU runner, not the generic function-calling agent loop. For non-lightweight TAU records, configure `user_model` and `user_api_key` in the benchmark TOML or set `USER_MODEL_NAME` / `USER_API_KEY` in the environment; `judge_model` defaults to the user model when needed.
- TAU `tau_sample_workers` controls single-sample attempt workers for one model task. It is not model prompt batch size; RWKV generation is still protected by the model-stage limiter and completion writes remain queue-based.
- TAU official runs automatically enable lexical long-context routing when `tool_router_mode = "lexical"` unless `long_context_router_mode = "off"` is set. The default TAU chunk size is 1000 characters and the default evidence budget is 6000 characters. `prompt_max_chars` is only the final rendered agent prompt hard cap.
- Set `RWKV_TAU_LLM_TIMEOUT_S` for formal TAU runs so external user-simulator and judge LLM latency is bounded. `RWKV_TAU_USER_TIMEOUT_S` and `RWKV_TAU_JUDGE_TIMEOUT_S` can override the user and judge sides separately.
- `cot_prompt_template` / `final_prompt_template` are currently used by free_response and free_response_judge.
- `judge_prompt_template` is currently used by free_response_judge.
- free_response applies sampling overrides to CoT generation.
- livecodebench applies sampling overrides to both CoT and final stages.
- livecodebench defaults to full_code_* templates when configs/livecodebench.toml is missing.
- Model names match case-insensitively; safe_slug normalization is supported.
- Use template = "name" or templates = ["base", "override"] to merge templates before overrides.
- When both [default] and [cot]/[final] exist, values are merged in order: default -> stage -> model.
- When benchmark config is missing, callers may supply fallback_templates to use templates from configs/_templates.toml.
