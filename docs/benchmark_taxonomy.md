# Benchmark 四域分类与结构约定

请求集(`src/eval/benchmark_sources.py` 的 `REQUESTED_BENCHMARK_SOURCES`)是**单一事实源**:每个 benchmark 的
名字 / dataset slug / 集成方式(integration)/ 调度 job / **域(domain)** / 数据来源都只在这里声明,
注册表、调度矩阵、prepper 注册、dashboard 分组全部由此派生。

域取值:`knowledge | math | code | agent`(`BENCHMARK_DOMAINS`),按域分桶见 `BENCHMARKS_BY_DOMAIN`。

## agent 域(21)—— 多轮 agent 通道 `function_agent_loop`(或专用通道)

模型以 RWKV 训练格式多轮行动,官方沙盒/verifier 判分(细节见 `docs/agent_loop.md`):

| benchmark | 问题领域 | executor | verifier |
| --- | --- | --- | --- |
| terminal_bench_2_1 | 终端环境任务 | shell_sandbox(docker,官方任务镜像) | terminal_bench_official |
| nl2repo | 自然语言生成完整仓库 | shell_sandbox(subprocess 工作区) | nl2repo_official(官方 post_processor + Docker 基础镜像测试) |
| deepswe | 长程软件工程 | shell_sandbox(docker) | repo_tests_official(任务自带程序化测试) |
| browsecomp | 浏览器搜索/深度查找 | 专用通道 `function_browsecomp`(可启用 agentic web_search + 官方 judge) | LLM judge |
| widesearch | 宽搜索/大规模信息收集 | web_search(实时检索;可由行覆盖为回放) | widesearch_official |
| deepsearchqa | 深度研究/多步搜索 | web_search(实时检索;可由行覆盖为回放) | llm_rubric_judge |
| mcp_atlas | MCP 工具调用 | mcp_worker | mcp_atlas_official(官方 claim-coverage) |
| toolathlon | 通用工具使用 | mcp_worker | toolathlon_official(官方逐任务 evaluator) |
| apex_agents | 职业服务工作流 | shell_sandbox | unsupported_official |
| claweval | 通用真实 agent 工作流 | shell_sandbox | unsupported_official |
| wildclawbench | 真实运行环境 agent | shell_sandbox | unsupported_official |
| skillsbench | Agent skill 使用 | shell_sandbox | unsupported_official |
| hle_with_tools | 工具增强专家问答 | web_search(实时检索;HF 源需授权) | llm_rubric_judge |
| hy_backend_2_0 | 后端工程 agent(内部) | 按行格式自动分类 | expected_tool_calls / llm_rubric_judge |
| hy_swe_max | 高难软件工程 agent(内部) | 同上 | 同上 |
| hy_companybench | 企业工作流 agent(内部) | 同上 | 同上 |
| e_bench | enterprise/execution agent(内部) | 同上 | 同上 |
| hy_finmodelbench | 金融建模 agent(内部) | 同上 | 同上 |
| prodbench | 产品/运营工作流 agent(内部) | 同上 | 同上 |
| hy_skillsworld | skill-based workflow agent(内部) | 同上 | 同上 |
| hy_euler_pro | 工具辅助数学/编程求解 agent(内部) | 同上 | 同上 |

## code 域(3)—— coding 通道 `code_swe_bench`(官方 docker harness)

| benchmark | 问题领域 | 判分 |
| --- | --- | --- |
| swe_bench_multilingual | 多语言真实 GitHub issue 修复 | 官方 `swebench.harness.run_evaluation`(docker) |
| swe_bench_verified | 人工验证 Python issue 修复 | 同上 |
| swe_bench_pro | 更难工程级修复 | 通用通道;Scale 官方 Pro harness 待接 |

## math 域(8)—— `free_response`(math_verify)或 `free_response_judge`(LLM judge)

| benchmark | 判分通道 |
| --- | --- |
| matharena_apex / arxivmath / horizonmath / hy_math / imoanswerbench | free_response(对齐 MathArena 官方 final-answer 解析) |
| usamo_2026(证明)/ phybench / cmt_benchmark | free_response_judge |

## knowledge 域(8)

| benchmark | 判分通道 |
| --- | --- |
| gpqa_diamond | multi_choice_cot(选择题) |
| hle | free_response(+judge 回退) |
| frontierscience_research / frontierscience_olympiad(无公开数据,本地源) | free_response_judge |
| superchem / cl_bench / cl_bench_life(rubrics 全对制)/ aa_lcr(长上下文阅读) | free_response_judge |

## 通道职责(结构约定)

| 通道 | 职责 | 域 |
| --- | --- | --- |
| `multi_choice_*` | 选择题 | knowledge |
| `free_response` / `free_response_judge` | 自由作答(math_verify / LLM judge) | math、knowledge |
| `code_*` | 代码生成/修复,官方 harness 判分 | code |
| `function_agent_loop` | 多轮 agent(官方沙盒 verifier + 格式转换器) | agent |
| `function_browsecomp` | BrowseComp 专用(agent 域) | agent |
| `function_agent_tool_call` | 通用单轮 JSON 工具调用;请求集已无成员,保留为通用/回归通道 | - |
| `function_bfcl_* / tau_* / mcp_bench` 等 | 既有 FC 基准,不属请求集 | - |

## 数据与资产 env(与 `.env.example` 对应)

- 数据源:`RWKV_FREE_ANSWER_SOURCE_ROOT` / `RWKV_AGENT_LOOP_SOURCE_ROOT` / `RWKV_AGENT_TOOL_CALL_SOURCE_ROOT`(每数据集可用 `..._SOURCE_<NAME>` 覆盖)。
- 官方资产:`RWKV_TERMINAL_BENCH_ROOT`、`RWKV_WIDESEARCH_OFFICIAL_ROOT`(+`RWKV_WIDESEARCH_EVAL_COMMAND`)、`RWKV_MCP_ATLAS_ROOT`、`RWKV_TOOLATHLON_ROOT`。
- LLM judge:`JUDGE_MODEL` / `JUDGE_API_KEY` / `JUDGE_BASE_URL`。

## dashboard 分组

`src/dashboard/core/domains.py`:agent 域 benchmark 显示为 **"agent系列"**(仍计入 function-call 聚合视图),
其余 function-calling 数据集维持 "function_call系列";knowledge/math/code 沿用既有分组。
