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
- 兼容旧脚本的评测字段：pass_k、avg_k、report_pass_k、report_avg_k

备注：
- CLI 参数会覆盖配置值。
- 面向 scheduler 的正式 benchmark 现在统一走 auto avg@k 执行计划：
  数据量 > 5000 时做确定性随机抽样；数据量 <= 5000 时重复整套题，直到达到 5000 个有效样本。
- 当前 benchmark job 不再统计 pass@k；现有 pass_k / report_pass_k 字段只保留给旧脚本和兼容路径。
- benchmark job 的 zeroshot / cot_mode 由具体 evaluator 入口控制，不再通过 TOML 切换。
- llm_judge 仍由评测脚本或 CLI 控制，不从 TOML 读取。
- free_response 的采样配置只用于 CoT 生成阶段。
- livecodebench 的采样配置同时作用于 CoT 和 final 阶段。
- 缺少 configs/livecodebench.toml 时，livecodebench 默认使用 full_code_* 模板。
- 模型名大小写不敏感，支持 safe_slug 归一化。
- 可用 template = "name" 或 templates = ["base", "override"] 先合并模板再覆盖。
- 同时存在 [default] 与 [cot]/[final] 时，按 default -> stage -> model 的顺序合并。
- 如果缺少 benchmark 文件，调用方可通过 fallback_templates 使用 configs/_templates.toml 里的模板。
