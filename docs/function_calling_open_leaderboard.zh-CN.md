# Function-Calling 开源 Benchmark 综合榜

更新时间：2026-05-23  
数据来源：本地 PostgreSQL `scores` / `completions` latest run  
模型范围：`rwkv7-g1e/g1f` 的 13.3B 与 7.2B

## 口径

纳入当前本地已经完成的开源 function-calling benchmark：

- APIBank：`apibank_level1_test`、`apibank_level2_test`
- BFCL / Gorilla：`bfcl_simple_python_test`、`bfcl_multiple_test`、`bfcl_exec_*`
- ToolAlpaca：`toolalpaca_eval_real_test`、`toolalpaca_eval_simulated_test`

主榜使用“榜单族等权”：先分别计算 APIBank、BFCL、ToolAlpaca 的平均分，再对三个榜单族求平均。这样避免 BFCL 子项数量更多导致综合分被 BFCL 主导。

`toolalpaca_eval_simulated_test` 当前仍混有 simulator timeout 与本地 exact-match 口径偏严问题，因此同时给出一个去掉 simulated 的稳定榜。

外部公开榜单目前先纳入 CoALM 论文 Table 3 的 API-Bank Level-1 / Level-2 `ROUGE-L` 结果。注意：外部 `ROUGE-L` 与我们本地真实执行后的 `Accuracy / Correctness` 不是同一指标，下面的混合表用于横向定位，不作为严格同口径 SOTA 结论。

## API-Bank 公开混合榜

| 排名 | 模型 | 指标 | Level-1 | Level-2 | 平均 | 来源 |
|---:|---|---|---:|---:|---:|---|
| 1 | CoALM 70B | ROUGE-L | 92.70 | 83.20 | 87.95 | CoALM Table 3 |
| 2 | CoALM 8B | ROUGE-L | 92.80 | 81.90 | 87.35 | CoALM Table 3 |
| 3 | CoALM 405B | ROUGE-L | 93.40 | 77.80 | 85.60 | CoALM Table 3 |
| 4 | Hammer2.0-7B | ROUGE-L | 90.10 | 74.00 | 82.05 | CoALM Table 3 |
| 5 | g1f 13.3B | Accuracy / Correctness | 79.07 | 83.87 | 81.47 | local official run |
| 6 | g1e 13.3B | Accuracy / Correctness | 79.65 | 82.49 | 81.07 | local official run |
| 7 | Qwen2.5-7B-Instruct | ROUGE-L | 84.30 | 73.90 | 79.10 | CoALM Table 3 |
| 8 | g1f 7.2B | Accuracy / Correctness | 77.33 | 76.04 | 76.69 | local official run |
| 9 | Llama-3.1-8B-Instruct | ROUGE-L | 72.70 | 75.20 | 73.95 | CoALM Table 3 |
| 10 | ToolAce | ROUGE-L | 81.50 | 63.60 | 72.55 | CoALM Table 3 |
| 11 | g1e 7.2B | Accuracy / Correctness | 72.67 | 58.99 | 65.83 | local official run |
| 12 | Granite-20B-Code | ROUGE-L | 60.30 | 45.70 | 53.00 | CoALM Table 3 |
| 13 | LDST | ROUGE-L | 8.30 | 7.10 | 7.70 | CoALM Table 3 |
| 14 | tod-zero-bqag3oyb | ROUGE-L | 3.70 | 4.20 | 3.95 | CoALM Table 3 |
| 15 | Fnc-TOD 13B | ROUGE-L | 3.90 | 3.30 | 3.60 | CoALM Table 3 |
| 16 | nc-latent-tod-step-2 | ROUGE-L | 3.20 | 3.20 | 3.20 | CoALM Table 3 |

## 主榜：开源 Function-Calling 综合榜

| 排名 | 模型 | 综合分 | APIBank | BFCL | ToolAlpaca |
|---:|---|---:|---:|---:|---:|
| 1 | rwkv7-g1f-13.3B-20260415 | 75.66 | 81.47 | 90.00 | 55.50 |
| 2 | rwkv7-g1e-13.3B-20260309 | 74.55 | 81.07 | 88.71 | 53.88 |
| 3 | rwkv7-g1f-7.2B-20260414 | 66.78 | 76.69 | 85.29 | 38.38 |
| 4 | rwkv7-g1e-7.2B-20260301 | 62.71 | 65.83 | 83.79 | 38.50 |

## 稳定榜：去掉 ToolAlpaca simulated

| 排名 | 模型 | 综合分 | APIBank | BFCL | ToolAlpaca real |
|---:|---|---:|---:|---:|---:|
| 1 | rwkv7-g1f-13.3B-20260415 | 82.16 | 81.47 | 90.00 | 75.00 |
| 2 | rwkv7-g1e-13.3B-20260309 | 81.18 | 81.07 | 88.71 | 73.75 |
| 3 | rwkv7-g1f-7.2B-20260414 | 71.91 | 76.69 | 85.29 | 53.75 |
| 4 | rwkv7-g1e-7.2B-20260301 | 68.21 | 65.83 | 83.79 | 55.00 |

## 全量逐 Benchmark 平均

逐 benchmark 等权，使用四个模型都有分数的 11 个 benchmark，不纳入 `bfcl_v3_test` 的 7B 零分缺口。

| 排名 | 模型 | 平均分 | benchmark 数 |
|---:|---|---:|---:|
| 1 | rwkv7-g1f-13.3B-20260415 | 82.18 | 11 |
| 2 | rwkv7-g1e-13.3B-20260309 | 80.99 | 11 |
| 3 | rwkv7-g1f-7.2B-20260414 | 75.19 | 11 |
| 4 | rwkv7-g1e-7.2B-20260301 | 72.29 | 11 |

## 明细分数

| benchmark | g1f 13.3B | g1e 13.3B | g1f 7.2B | g1e 7.2B |
|---|---:|---:|---:|---:|
| apibank_level1_test | 79.07 | 79.65 | 77.33 | 72.67 |
| apibank_level2_test | 83.87 | 82.49 | 76.04 | 58.99 |
| bfcl_exec_multiple_ast_test | 86.00 | 82.00 | 84.00 | 80.00 |
| bfcl_exec_multiple_test | 94.00 | 88.00 | 94.00 | 90.00 |
| bfcl_exec_parallel_multiple_test | 87.50 | 87.50 | 70.00 | 72.50 |
| bfcl_exec_parallel_test | 90.00 | 88.00 | 88.00 | 84.00 |
| bfcl_exec_simple_test | 98.00 | 98.00 | 92.00 | 92.00 |
| bfcl_multiple_test | 86.00 | 87.00 | 83.00 | 81.50 |
| bfcl_simple_python_test | 88.50 | 90.50 | 86.00 | 86.50 |
| toolalpaca_eval_real_test | 75.00 | 73.75 | 53.75 | 55.00 |
| toolalpaca_eval_simulated_test | 36.00 | 34.00 | 23.00 | 22.00 |

## 结论

当前开源 function-calling 综合榜里，g1f 系列整体优于同参数量 g1e：

- 13.3B：g1f 主榜 75.66，高于 g1e 74.55；稳定榜 82.16，高于 g1e 81.18。
- 7.2B：g1f 主榜 66.78，高于 g1e 62.71；稳定榜 71.91，高于 g1e 68.21。

需要注意：

- APIBank 的日期缺省年份规则会影响部分样本，但没有单条答案泄漏。
- BFCL 当前结构性输出问题已经修正，分数主要反映模型真实工具选择和参数生成。
- ToolAlpaca simulated 分数不建议作为强结论，后续最好改成官方 LLM judge 口径，或至少给 simulator 加 retry/降并发后再重跑。
