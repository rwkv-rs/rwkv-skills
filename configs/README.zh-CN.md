# Benchmark 配置

将每个 benchmark 的 TOML 文件放在此目录下：

- 路径：configs/<benchmark>.toml
- Benchmark 名称：数据集 slug 去掉 split 后缀（例如 math_500_test -> math_500）
- 每个文件包含以模型名为键的 table。
- 可选 [default] table 适用于所有模型；模型 table 会覆盖它。
- 模板定义在 configs/_templates.toml；每个顶层 table 都是一个模板。
- 支持阶段配置：可使用 [cot] / [final]（也可在其中再嵌套模型 table）。

每个模型 table 支持的字段：
- SamplingConfig 字段：max_generate_tokens、temperature、top_k、top_p、
  alpha_presence、alpha_frequency、alpha_decay、stop_tokens、ban_tokens、
  pad_zero、no_penalty_token_ids
- 评测字段：pass_k、avg_k、report_pass_k、report_avg_k、max_samples（free_response 与 multi_choice_cot；direct/code/instruction 入口也会读取 max_samples）
- Prompt 字段：cot_prompt_template、final_prompt_template、judge_prompt_template

备注：
- CLI 参数会覆盖配置值。
- pass_k / avg_k 可用于 CoT 评测入口（free_response / free_response_judge / multi_choice_cot），CLI 参数优先生效。
- avg_k / report_avg_k 既支持整数（例如 `16`），也支持 `(0, 1)` 之间的小数比例（例如 `0.2`）。
- 当 `avg_k` 配成比例时，评测会对每道题使用前 `ceil(比例 * repeats)` 个可用样本计算 avg。
- 未显式传入 CLI `--max-samples` 时，会默认读取配置里的 `max_samples`。
- llm_judge 仍由评测脚本或 CLI 控制，不从 TOML 读取。
- cot_prompt_template / final_prompt_template 当前用于 free_response 和 free_response_judge。
- judge_prompt_template 当前用于 free_response_judge。
- free_response 的采样配置只用于 CoT 生成阶段。
- livecodebench 的采样配置同时作用于 CoT 和 final 阶段。
- 缺少 configs/livecodebench.toml 时，livecodebench 默认使用 full_code_* 模板。
- 模型名大小写不敏感，支持 safe_slug 归一化。
- 可用 template = "name" 或 templates = ["base", "override"] 先合并模板再覆盖。
- 同时存在 [default] 与 [cot]/[final] 时，按 default -> stage -> model 的顺序合并。
- 如果缺少 benchmark 文件，调用方可通过 fallback_templates 使用 configs/_templates.toml 里的模板。
