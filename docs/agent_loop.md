# Agent-loop 多轮评测通道(`function_agent_loop`)

通用多轮 agentic benchmark 通道。模型以 RWKV 训练格式行动:

```
System: Tools:
[ ...JSON 工具目录... ]
Return only a JSON function call.
The JSON shape is {"name":"tool_name","arguments":{...}}.
Use only listed tool names.
Call {"name":"final_answer","arguments":{"answer":"..."}} when the task is complete.

User: <任务指令>

Assistant: ```json
{"name":"bash","arguments":{"command":"ls"}}
```

User: Function output:
{"success":true,"output":"..."}

Assistant: ```json
```

- **Executor**(`src/eval/tasks/function_calling/agent_loop_executors.py`)把 JSON call 映射到真实环境;
- **Verifier**(`agent_loop_verifiers.py`)用 **官方判别器** 给最终状态/答案打分;
- 官方资产 / docker / judge 端点缺失时,`preflight_agent_loop_runtime` 在生成前直接报错,**绝不伪造分数**。

## Manifest 行 schema(prepper 产物)

```json
{
  "task_id": "...",
  "instruction": "...",
  "system_extra": "",
  "tools": [{"name": "...", "description": "...", "parameters": {...}}],
  "executor": {"kind": "manifest_replay|shell_sandbox|mcp_worker", "config": {...}},
  "verifier": {"kind": "expected_tool_calls|llm_rubric_judge|terminal_bench_official|nl2repo_official|widesearch_official|mcp_atlas_official|toolathlon_official|unsupported_official", "config": {...}},
  "expected_tool_calls": [{"name": "...", "arguments": {...}, "argument_options": {...}}],
  "recorded_tool_outputs": [{"name": "...", "arguments": {...}, "output": ..., "error": ""}],
  "metadata": {"source_benchmark": "...", "official_task_id": "...", "docker_image": "...", "rubrics": [...]}
}
```

源数据行可以只有 QA / rubrics / 多轮 messages —— prepper(`src/eval/datasets/data_prepper/function_calling/agent_loop.py`)按格式自动分类:

| 行格式 | 归类 |
| --- | --- |
| 带 `executor`/`verifier` | 原样透传 |
| 带 `recorded_tool_outputs`/`tool_outputs` | manifest 回放执行 |
| 带 `rubrics`/`rubric` | `llm_rubric_judge`(judge 端点见下) |
| 只有 QA(question/answer) | 单步 `final_answer` + `expected_tool_calls` 判定(官方单轮即单轮) |

## 数据源

默认根:`data/agent_loop_sources/<benchmark>/test.jsonl`;env 覆盖:
`RWKV_AGENT_LOOP_SOURCE_ROOT` 或每数据集 `RWKV_AGENT_LOOP_SOURCE_<NAME>`。

## 各 benchmark 官方资产准备

| benchmark | executor | verifier | 需要的 env / 资产 |
| --- | --- | --- | --- |
| terminal_bench_2_1 | shell_sandbox(docker, 官方任务镜像) | terminal_bench_official(容器内跑官方任务测试) | `RWKV_TERMINAL_BENCH_ROOT`=terminal-bench-2-1 checkout;docker;行 metadata.official_task_id + docker_image |
| widesearch | web_search 或 manifest_replay | widesearch_official(官方数据/gold + 官方 evaluation 逻辑) | `RWKV_WIDESEARCH_OFFICIAL_ROOT`;默认读取 `data/widesearch_official` 或 `RWKV_WIDESEARCH_DATA_ROOT`;可选 `RWKV_WIDESEARCH_EVAL_COMMAND` |
| mcp_atlas | mcp_worker | mcp_atlas_official(官方 score_claims.py claim-coverage) | `RWKV_MCP_ATLAS_ROOT`;`JUDGE_MODEL/JUDGE_API_KEY[/JUDGE_BASE_URL]`;executor.config.runtime_root |
| toolathlon | mcp_worker | toolathlon_official(官方逐任务 evaluator) | `RWKV_TOOLATHLON_ROOT`;docker/podman;行 verifier.config.evaluator_command |
| deepsearchqa | web_search 或 manifest_replay | llm_rubric_judge | `RWKV_WEB_SEARCH_API_URL` / `RWKV_WEB_SEARCH_API_KEY`;`JUDGE_MODEL/JUDGE_API_KEY` |
| hle_with_tools | web_search 或 manifest_replay | llm_rubric_judge | HF `cais/hle` 授权;`RWKV_WEB_SEARCH_API_URL` / `RWKV_WEB_SEARCH_API_KEY`;`JUDGE_MODEL/JUDGE_API_KEY` |
| deepswe | shell_sandbox(docker) | repo_tests_official(行内 test_command,官方程序化 verifier) | docker;行 verifier.config.test_command |
| nl2repo | shell_sandbox(subprocess 工作区) | nl2repo_official(官方 post_processor + Docker 基础镜像测试) | `RWKV_NL2REPO_ROOT`=NL2RepoBench checkout;docker |
| claweval / wildclawbench / skillsbench / apex_agents | shell_sandbox | unsupported_official(v1 preflight 报错并给出 clone/配置指引) | 待接入官方 harness |
| 内部 hy_* / e_bench / prodbench / hy_euler_pro | 按行格式自动分类 | expected_tool_calls 或 llm_rubric_judge | 本地源 |

实时浏览模式:行内把 executor 换成 `{"kind": "web_search"}` 即注入 `web_search`/`fetch_url` 工具,
后端为 .env 配置的通用 JSON 搜索端点(`RWKV_WEB_SEARCH_API_URL` / `RWKV_WEB_SEARCH_API_KEY`,
Serper 风格 X-API-KEY);未配置时 preflight 报错。离线 manifest 回放不需要任何 API。

域分类总表见 `docs/benchmark_taxonomy.md`(frontierscience_* 属 knowledge 域,走 free_response_judge,不在本通道)。

judge 端点从 `.env` 读取(`JUDGE_MODEL` / `JUDGE_API_KEY` / `JUDGE_BASE_URL`),也可用 `--judge-model/--judge-api-key/--judge-base-url` 覆盖。

## 运行

```bash
python -m src.eval.tasks.function_calling.runner \
  --dataset widesearch_test.jsonl \
  --max-steps 20 --max-tool-errors 5 \
  --agent-loop-command-timeout-s 60 --agent-loop-max-output-chars 8000
```

调度侧 job 名为 `function_agent_loop`,数据集矩阵由 `benchmark_registry.py` 自动派生。

长上下文路由: `function_agent_loop` 默认使用 `--candidate-router-mode auto`。当工具表过大、原始对话/metadata 上下文超过
candidate-router budget,或 long-doc compaction 实际压缩了消息时,该轮会走 `parallel_candidate`;显式传
`--candidate-router-mode off` 可关闭。
