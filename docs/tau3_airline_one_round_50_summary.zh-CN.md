# tau3_bench_airline_base 单轮 50 条实验总结

日期：2026-05-26  
模型：`rwkv7-g1f-7.2b-20260414-ctx8192`  
接入方式：远端 GPU0 服务经 `autossh` 转发到本地 `http://127.0.0.1:19081`  
数据集：`tau3_bench_airline_base`，50 条样本，`avg_k=1`  
运行形态：单轮表现测试，`max_steps=3`，`batch_size=1`，`infer_max_workers=1`

## 结果

| 变体 | task_id | avg@1 | success_rate | agent_error_rate | avg_agent_turns | avg_stage_prompt_chars | long_doc_prompt_rate | tool_route_routed_rate | tool_route_avg_selected_tools |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline：`long_doc=off`，`tool_router=off` | 1008 | 0.0 | 0.0 | 0.0 | 1.0 | 24442.0 | 0.0 | 0.0 | 14.0 |
| chunk：`long_doc=lexical`，`tool_router=off` | 1009 | 0.0 | 0.0 | 0.0 | 1.0 | 24442.0 | 0.0 | 0.0 | 14.0 |
| router：`long_doc=off`，`tool_router=lexical` | 1010 | 0.0 | 0.0 | 0.02 | 0.98 | 23050.47 | 0.0 | 1.0 | 11.47 |

## 关键观察

1. 三组 50 条完整实验的 `avg@1` 和 `success_rate` 都是 0.0。
2. chunk 机制在这组 airline 单轮测试中没有实际触发，`long_doc_prompt_rate=0.0`。原因是该测试首轮没有长工具输出或长历史需要压缩，所以它不会改变 prompt，也不会改变得分。
3. tool-router lexical 触发了实际工具裁剪，`tool_route_routed_rate=1.0`，平均工具数从 14 个降到 11.47 个，平均 prompt 字符数从 24442 降到 23050.47。
4. router 明显改变了模型行为，但没有带来成功率提升。50 条中工具分布大致为：`transfer_to_human_agents` 25 次、`get_user_details` 5 次、`search_direct_flight` 1 次、`search_onestop_flight` 3 次、`respond` 13 次，另有 1 条曾按旧逻辑被运行时错误记录为失败样本。
5. baseline/chunk 的主要失败形态是模型在首轮直接 `respond`，要求用户提供 user id、出发地、目的地等信息，而不是主动利用任务上下文调用工具。
6. router 的主要失败形态变为过早转人工、参数抽取错误或调用了不够关键的查询工具；这说明只减少工具列表还不足以完成 tau3 airline 任务。
7. `prompt_max_chars=8192` 目前不能完整约束 tau3 airline 的系统提示、政策和工具 schema 总体长度。实际记录的 prompt 仍在 23K 到 24K 字符左右，说明上下文压力主要来自环境说明和工具目录。

## 本轮代码层修正

1. 为 tau 官方 user simulator / judge 接入了超时参数，环境变量优先级为 `RWKV_TAU_LLM_TIMEOUT_S`、`RWKV_TAU_USER_TIMEOUT_S`、`RWKV_LLM_TIMEOUT_S`。
2. tau 官方运行时不再做单样本异常兜底；官方 orchestrator、user simulator 或 judge 抛出的异常会直接向上抛出，让任务失败/中止，避免把运行异常伪造成可对榜 0 分样本。
3. DB 结构和续跑路径保持本项目现有实现，没有按外部项目覆盖。

## 测试

已通过相关单元测试：

```text
55 passed, 1 warning
```

此前扩大覆盖的相关测试也通过：

```text
79 passed
77 passed
```

## 结论

这轮实验说明：对 tau3 airline 的单轮 50 条测试，chunk 机制没有作用面；tool-router 可以降低工具目录长度并改变行为，但当前 prompt/决策约束不足，仍无法拿到可对榜的成功样本。下一步应优先处理 tau3 airline 的任务策略提示和工具调用顺序约束，再做 chunk+router 的完整因子组合。

## 官方 user/judge 对齐后重测

日期：2026-05-26 14:07-14:35  
user simulator：`gpt-4.1-2025-04-14`  
NL assertion judge：`gpt-4.1-2025-04-14`  
并发设置：`batch_size=1`，`infer_max_workers=1`，`db_write_queue=1`，`RWKV_EVAL_RUN_MODE=rerun`

| 变体 | task_id | avg@1 | success_rate | agent_error_rate | avg_agent_turns | avg_stage_prompt_chars | long_doc_prompt_rate | tool_route_routed_rate | tool_route_avg_selected_tools |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline：`long_doc=off`，`tool_router=off` | 1012 | 0.0 | 0.0 | 0.0 | 1.0 | 24442.0 | 0.0 | 0.0 | 14.0 |
| chunk：`long_doc=lexical`，`tool_router=off` | 1013 | 0.0 | 0.0 | 0.0 | 1.0 | 24442.0 | 0.0 | 0.0 | 14.0 |
| router：`long_doc=off`，`tool_router=lexical` | 1014 | 0.0 | 0.0 | 0.0 | 1.0 | 23153.26 | 0.0 | 1.0 | 11.78 |

首轮输出分布：

| task_id | `respond` | `transfer_to_human_agents` | `get_user_details` | `search_onestop_flight` | `search_direct_flight` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1012 | 50 | 0 | 0 | 0 | 0 |
| 1013 | 50 | 0 | 0 | 0 | 0 |
| 1014 | 12 | 25 | 7 | 5 | 1 |

重测结论：

1. 使用官方推荐形态的 `gpt-4.1-2025-04-14` 作为 user/judge 后，三组仍然全部 0 分。
2. `chunk` 组仍未触发，`long_doc_prompt_rate=0.0`，说明单轮 airline 样本没有长文压缩作用面。
3. `router` 组没有运行时错误，且继续显著改变首轮行为，但仍未把工具调用转化为成功样本。
4. 本轮中转站在低并发下总体稳定，只有 router 组中间出现过单条慢调用，没有中断任务。

## 与旧有分数的差异

旧记录中 `rwkv7-g1f-7.2b-20260414-ctx8192` 在 `tau2_bench_airline_base` 的 task `1001` 得到 `avg@1=0.18`，对应 50 条中 9 条通过。该任务是 tau2，而本轮是 `tau3_bench_airline_base`；旧任务按完整多轮运行，日志命令包含 `--max-steps 200` 和 `--decision-max-tokens 1024`。DB 回看显示 task `1001` 的平均 agent 轮数为 42.56，首个通过样本用了 39 个 agent turn。

本轮三组 task `1012`、`1013`、`1014` 是单轮首决策消融，`--max-steps 3`、`--decision-max-tokens 256`，DB 记录的 `avg_agent_turns=1.0`，每条最多只有 1 次 agent 决策。因此它只能衡量首轮行为是否正确，不能直接复现旧的完整多轮 tau2 分数。

## 完整多轮复测状态

为复现旧 task `1001` 的设置，已启动 `tau2_bench_airline_base`、50 条、`--max-steps 200`、`--decision-max-tokens 1024`、低并发的完整多轮复测，生成 task `1015`。该任务最终写入 50 条 Completed，但不能作为有效分数使用。

原因是第 1 条之后，当前 `.env` 的 user/judge 中转 key 已返回 HTTP 403：`User has been banned`。DB 回看显示 task `1015` 中 49/50 条为旧异常兜底逻辑捕获后的 0 分样本，`agent_error_rate=0.98`，因此 `avg@1=0.0` 是无效 API 状态导致的假结果，而不是模型真实能力。

后续完整 tau2/tau3 多轮对榜测试需要先更换可用的 `USER_API_KEY`/`JUDGE_API_KEY`，或等待当前中转 key 恢复；在此之前不应继续生成可对榜分数。

## 2026-05-28 g1g 真实 tau3 airline 三组消融

模型：`rwkv7-g1g-7.2b-20260523-ctx8192`  
推理端：远端旧推理引擎，经本地 `http://127.0.0.1:19081` 转发  
官方数据：`references/tau2-bench/data`，`tau3_bench_airline/base`  
user simulator：`gpt-5.4-mini`  
NL assertion judge：`gpt-5.4`  
运行形态：官方 tau3 真实环境、真实工具执行、50 条、`avg_k=1`、`max_steps=80`、`batch_size=4`

| 变体 | task_id | avg@1 | success_rate | agent_error_rate | avg_agent_turns | avg_stage_prompt_chars | avg_sample_prompt_chars | long_doc_prompt_rate | tool_route_routed_rate | tool_route_avg_selected_tools |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline：`long_doc=off`，`tool_router=off` | 1040 | 0.04 | 0.04 | 0.90 | 4.56 | 6492.07 | 29603.84 | 0.0 | 0.0 | 14.0 |
| lexical chunk：`long_doc=lexical`，`tool_router=off` | 1041 | 0.02 | 0.02 | 0.96 | 5.14 | 6564.71 | 33742.60 | 1.0 | 0.0 | 14.0 |
| parallel chunk/router：`long_doc=model_parallel`，`tool_router=model_parallel` | 1042 | 0.08 | 0.08 | 0.88 | 3.10 | 6821.95 | 21579.65 | 1.0 | 1.0 | 10.0 |

关键结论：

1. 三组均是真实 tau3 airline 运行，不是 mock，也不是轨迹比对；官方 user simulator、环境工具、数据库状态变更和 judge 都实际参与。
2. 本轮最好的配置是第三组 `parallel chunk/router`，`success_rate=0.08`，高于 baseline 的 `0.04` 和 lexical chunk 的 `0.02`。
3. lexical chunk 单独启用后没有提升，反而下降；说明“只做本地词面切块再拼回去”会引入噪声或破坏局部上下文，不足以解决 tau3 airline 的工具决策问题。
4. parallel chunk/router 把平均样本 prompt 字符数从 baseline 的 `29603.84` 降到 `21579.65`，平均选中工具数从 `14.0` 降到 `10.0`，并把平均 agent turns 从 `4.56` 降到 `3.10`。这说明模型辅助切分和工具路由确实减少了无关上下文和工具表。
5. 主要失败仍来自 agent 本体的工具参数抽取和任务状态理解：常见错误包括把姓名当 user_id、生成 `user_0/user_1`、把 flight number 当 reservation_id、`book_reservation` 缺少 passenger `dob`、以及过早 `transfer_to_human_agents`。
6. 本轮没有出现数据库或推理端中断；`python -m src.main ...` 在本地出现过 psycopg pool 初始化超时，实际运行改用 `python -c "from src.main import main; ..."` 调同一入口完成。

注意：task `1041`/`1042` 是旧的约 8K step prompt 口径，`avg_sample_prompt_chars` 当时也是“每个样本所有 step 的累计 prompt 字符数”。它们不再作为 3K chunk 消融的有效对比，只保留为历史记录。

## 2026-05-29 g1g 3K step prompt 重跑

目标：第二/第三组每次 agent step 控制在约 3000 字符；第三组增加 RWKV 并行分片，尽量缩短每个分片上下文。  
统计口径修正：`avg_sample_prompt_chars` 现在表示单次模型调用平均 prompt 字符数；旧的“样本内所有 step 累计值”保留为 `avg_sample_total_prompt_chars`。  
checker：官方 tau3 reward/score 正常写库；LLM failure checker 只用于诊断，第三组运行时显式关闭，避免非官方 checker API 空响应影响 score 写入。

| 变体 | task_id | avg@1 | success_rate | agent_error_rate | decision_parse_error_rate | avg_agent_turns | avg_sample_prompt_chars | avg_sample_total_prompt_chars | max_stage_prompt_chars | tool_route_avg_selected_tools |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical chunk/router 3K：`long_doc=lexical`，`tool_router=lexical` | 1043 | 0.38 | 0.38 | 0.72 | 0.66 | 5.46 | 2781.70 | 15820.92 | 3065.0 | 6.0 |
| parallel chunk/router 3K：`long_doc=model_parallel`，`tool_router=model_parallel` | 1044 | 0.46 | 0.46 | 0.34 | 0.28 | 3.54 | 2790.64 | 10080.47 | 3065.0 | 6.0 |

第三组关键配置：

1. `prompt_max_chars=3072`，`history_max_chars=4200`，`decision_max_tokens=384`。
2. 文档 chunk：`long_doc_max_chars=650`，`long_doc_max_evidence_chunks=2`，`long_doc_max_evidence_chars=1800`。
3. RWKV 并行路由：`long_doc_model_parallel_batch_size=16`，`tool_router_parallel_chunk_tools=1`，`tool_router_parallel_batch_size=16`，`infer_max_workers=16`。
4. 工具目录路由后平均保留 `6.0/14` 个工具，避免整张工具表反复进入每个 step。

3K 重跑结论：

1. 两组都把单次 step prompt 稳定压到约 `2.8K`，最大值 `3065`，符合 3K 目标。
2. 第三组 `success_rate=0.46`，高于第二组 `0.38`；`agent_error_rate` 从 `0.72` 降到 `0.34`，`decision_parse_error_rate` 从 `0.66` 降到 `0.28`。
3. 第三组平均轮数从 `5.46` 降到 `3.54`，样本累计 prompt 从 `15820.92` 降到 `10080.47`。这说明 RWKV 并行分片不仅缩短单步上下文，也减少了多轮拖延。
4. 主要残留问题仍是 agent 决策质量：常见错误包括伪造 `user_0`、`USER-12345`、`R123456789`，把 flight number 当 reservation id，或向用户暴露内部推理/继续索要已给过的信息。
5. 当前最佳消融结果是 task `1044`。后续应在 1044 的 3K 并行路由基础上，继续优化 ID 抽取、工具调用参数 schema 约束和“已有信息优先查工具”的决策提示。

2026-05-29 分数修正：旧表曾记录 task `1043=0.10`、task `1044=0.24`，这是本地 adapter 额外把带 parse/loop-guard 诊断的样本强行置 0 造成的低估。官方 tau evaluator 已经给 `reward=1` 的样本应按官方 reward 计分；parse/loop-guard 只保留为 diagnostics。修正后本地 DB 中 task `1043` 为 `19/50=0.38`，task `1044` 为 `23/50=0.46`。
