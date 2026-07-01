# Function-Calling Benchmark 复刻手册

更新时间：2026-05-23  
目标：让另一个项目可以复刻当前 `rwkv-skills` 正在跑的 function-calling benchmark，并支持之后放到服务器大规模调度。

## 范围

这里把“当前六个 benchmark”定义为之前主力运行的六个数据集：

| 家族 | dataset slug | 本地数据文件 | 样本数 | scheduler job |
|---|---|---|---:|---|
| BFCL exec | `bfcl_exec_simple_test` | `data/bfcl_exec_simple/test.jsonl` | 100 | `function_bfcl_exec` |
| BFCL exec | `bfcl_exec_multiple_test` | `data/bfcl_exec_multiple/test.jsonl` | 50 | `function_bfcl_exec` |
| BFCL exec | `bfcl_exec_parallel_test` | `data/bfcl_exec_parallel/test.jsonl` | 50 | `function_bfcl_exec` |
| BFCL exec | `bfcl_exec_parallel_multiple_test` | `data/bfcl_exec_parallel_multiple/test.jsonl` | 40 | `function_bfcl_exec` |
| ToolAlpaca | `toolalpaca_eval_simulated_test` | `data/toolalpaca_eval_simulated/test.jsonl` | 100 | `function_toolalpaca` |
| ToolAlpaca | `toolalpaca_eval_real_test` | `data/toolalpaca_eval_real/test.jsonl` | 80 | `function_toolalpaca` |

当前实际批量命令经常把 APIBank 一起跑。另一个项目如果要完全复刻这轮 32 个任务，也要加上：

| 家族 | dataset slug | 本地数据文件 | 样本数 | scheduler job |
|---|---|---|---:|---|
| APIBank | `apibank_level1_test` | `data/apibank_level1/test.jsonl` | 172 | `function_api_bank` |
| APIBank | `apibank_level2_test` | `data/apibank_level2/test.jsonl` | 217 | `function_api_bank` |

## 总体链路

最小可复刻链路是：

1. 准备 JSONL 数据集。
2. 用统一 prompt 格式生成工具调用 JSON。
3. 解析模型输出为 `[{name, arguments}]`。
4. 调用对应 sandbox 或真实 HTTP API 得到 execution result。
5. 和 reference execution result / expected call 对比。
6. 写入 `completions`、`eval`、`scores` 三类结果。
7. scheduler 按 model x dataset 派发任务。

当前项目的主入口和路径：

| 责任 | 文件路径 |
|---|---|
| benchmark 元数据注册 | `src/eval/benchmark_registry.py` |
| runner 注册和 job 映射 | `src/eval/runner_registry.py` |
| scheduler job catalog | `src/eval/scheduler/jobs.py` |
| scheduler CLI | `src/eval/scheduler/cli.py` |
| function-calling 统一入口 | `src/eval/tasks/function_calling/runner.py` |
| function-calling 公共运行逻辑 | `src/eval/tasks/function_calling/common.py` |
| RWKV 官方 JSON prompt 格式 | `src/eval/tasks/function_calling/rwkv_prompt.py` |
| 简单 tool-call prompt / decode / eval | `src/eval/tasks/function_calling/simple_tool_call.py` |
| APIBank 真实执行 | `src/eval/tasks/function_calling/api_bank.py` |
| BFCL exec 真实执行 | `src/eval/tasks/function_calling/bfcl_exec.py` |
| ToolAlpaca simulator / real HTTP | `src/eval/tasks/function_calling/toolalpaca.py` |
| ToolAlpaca 源数据解析 | `src/eval/tasks/function_calling/toolalpaca_source.py` |
| 数据准备入口 | `src/eval/datasets/data_prepper/data_manager.py` |
| APIBank 数据准备 | `src/eval/datasets/data_prepper/function_calling/api_bank.py` |
| BFCL 数据准备 | `src/eval/datasets/data_prepper/function_calling/bfcl_small.py` |
| ToolAlpaca 数据准备 | `src/eval/datasets/data_prepper/function_calling/toolalpaca.py` |
| DB schema | `scripts/schema.sql` |
| 当前榜单文档 | `docs/function_calling_open_leaderboard.zh-CN.md` |

## 数据格式

六个核心数据集都用 JSONL，一行一个 task。另一个项目应直接支持这个统一结构：

```json
{
  "task_id": "exec_simple_0",
  "instruction": "User: ...",
  "tools": [
    {
      "name": "calc_binomial_probability",
      "description": "...",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      }
    }
  ],
  "expected_tool_calls": [
    {
      "name": "calc_binomial_probability",
      "arguments": {"n": 20, "k": 5, "p": 0.6},
      "argument_options": {"n": [20], "k": [5], "p": [0.6]}
    }
  ],
  "metadata": {}
}
```

BFCL exec 额外需要：

```json
{
  "expected_executable_calls": ["calc_binomial_probability(n=20, k=5, p=0.6)"],
  "execution_result_type": ["exact_match"]
}
```

ToolAlpaca 的 `tools[*].metadata` 必须保留 OpenAPI 路由信息：

```json
{
  "metadata": {
    "path": "/search",
    "method": "get",
    "api_name": "Axolotl",
    "server_url": "https://example.com",
    "operation": {}
  }
}
```

不要把 `expected_tool_calls` 放入 prompt；它只能进入 evaluator 和 DB 的 `agent_info` / `eval.ref_answer`。

## Prompt 格式

当前 function-calling 只支持 `rwkv_official_json`，实现路径：

- `src/eval/tasks/function_calling/rwkv_prompt.py`
- `src/eval/tasks/function_calling/simple_tool_call.py::build_simple_tool_call_prompt`

关键格式：

````text
System: <system prompt>

User:<instruction>

Assistant: <think>
</think>
```json
````

注意点：

- `User:` 后面没有空格。
- `Assistant:` 后面有一个空格。
- assistant 预填固定为 `<think>`、`</think>`、JSON fenced block opener。
- 输出必须是一个 JSON object 或 JSON array。
- 单工具调用输出 `{"name": "...", "arguments": {...}}`。
- 多工具调用输出 `[{"name": "...", "arguments": {...}}, ...]`。
- 停止后缀来自 `src/eval/tasks/function_calling/rwkv_prompt.py::JSON_CALL_STOP_SUFFIXES`：

````text
\n```
```
\nUser:
\nSystem:
\nAssistant:
````

System prompt 必须包含：

- `Tools:` 后接工具目录 JSON。
- `Output JSON schema:` 后接输出 schema。
- `Return exactly one JSON value that validates against the schema.`
- `For one tool call, return one JSON object.`
- `For multiple required tool calls, return a JSON array containing every required call in execution order; do not stop after the first call.`
- `Return no prose, no markdown, and no extra text outside the JSON value.`

APIBank 额外包含日期约定：

```text
API-Bank date convention: if a month/day or relative date has no explicit year and the conversation does not state today's date, use year 2023.
```

## APIBank 复刻

涉及文件：

- runner：`src/eval/tasks/function_calling/api_bank.py`
- prompt/decode 公共层：`src/eval/tasks/function_calling/simple_tool_call.py`
- 数据准备：`src/eval/datasets/data_prepper/function_calling/api_bank.py`
- registry：`src/eval/benchmark_registry.py`
- 本地数据：`data/apibank_level1/test.jsonl`、`data/apibank_level2/test.jsonl`

源数据默认查找：

```text
references/API-Bank
../API-Bank
```

可用环境变量覆盖：

```bash
API_BANK_SOURCE_ROOT=/path/to/API-Bank
RWKV_API_BANK_SOURCE_ROOT=/path/to/API-Bank
```

实际读取目录：

```text
$API_BANK_SOURCE_ROOT/lv1-lv2-samples/level-1-given-desc
```

真实执行方式：

- `ApiBankSandbox` 在 `src/eval/tasks/function_calling/api_bank.py`。
- 动态 import 官方 `apis/*.py`。
- 读取 `$API_BANK_SOURCE_ROOT/init_database/*.json`。
- 对模型生成的 API name / arguments 调用官方 API class。
- 用官方 API class 的 `check_api_call_correctness(actual, expected)` 判分。

复刻时不能只比对 JSON 参数；必须执行 API 后比较 execution result，否则 Level-2 的链式/状态类错误会失真。

## BFCL exec 复刻

涉及文件：

- runner / sandbox：`src/eval/tasks/function_calling/bfcl_exec.py`
- prompt/decode 公共层：`src/eval/tasks/function_calling/simple_tool_call.py`
- 数据准备：`src/eval/datasets/data_prepper/function_calling/bfcl_small.py`
- 本地数据：
  - `data/bfcl_exec_simple/test.jsonl`
  - `data/bfcl_exec_multiple/test.jsonl`
  - `data/bfcl_exec_parallel/test.jsonl`
  - `data/bfcl_exec_parallel_multiple/test.jsonl`

源数据默认查找：

```text
references/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data
../gorilla/berkeley-function-call-leaderboard/bfcl_eval/data
```

可用环境变量覆盖：

```bash
RWKV_BFCL_SMALL_SOURCE_ROOT=/path/to/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data
RWKV_BFCL_V4_SOURCE_ROOT=/path/to/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data
BFCL_V4_SOURCE_ROOT=/path/to/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data
```

四个数据集对应源文件：

| dataset | question | possible answer |
|---|---|---|
| `bfcl_exec_simple` | `unused_datasets/question/BFCL_v4_exec_simple.json` | `unused_datasets/possible_answer/BFCL_v4_exec_simple.json` |
| `bfcl_exec_multiple` | `unused_datasets/question/BFCL_v4_exec_multiple.json` | `unused_datasets/possible_answer/BFCL_v4_exec_multiple.json` |
| `bfcl_exec_parallel` | `unused_datasets/question/BFCL_v4_exec_parallel.json` | `unused_datasets/possible_answer/BFCL_v4_exec_parallel.json` |
| `bfcl_exec_parallel_multiple` | `unused_datasets/question/BFCL_v4_exec_parallel_multiple.json` | `unused_datasets/possible_answer/BFCL_v4_exec_parallel_multiple.json` |

真实执行方式：

- `BfclExecSandbox` 在 `src/eval/tasks/function_calling/bfcl_exec.py`。
- reference 和 model output 都会被渲染为 Python 风格调用字符串。
- sandbox 执行 `expected_executable_calls` 得到 reference result。
- sandbox 执行模型 decoded calls 得到 actual result。
- `exec_parallel*` 类别按无序匹配。
- 其他类别按顺序匹配。
- `execution_result_type` 控制 `exact_match` / `structural_match` 等匹配方式。

重点：

- 不要只检查工具名和参数字符串。
- 必须执行 reference 和 actual，再比较 execution result。
- `bfcl_exec_parallel` / `bfcl_exec_parallel_multiple` 的多调用允许乱序匹配。
- 当前代码对 parallel 多调用会在 prompt 后追加 `[\n` 前缀，位置在 `build_bfcl_exec_prompt()`。

## ToolAlpaca 复刻

涉及文件：

- runner / sandbox：`src/eval/tasks/function_calling/toolalpaca.py`
- 源数据解析：`src/eval/tasks/function_calling/toolalpaca_source.py`
- 数据准备：`src/eval/datasets/data_prepper/function_calling/toolalpaca.py`
- 本地数据：
  - `data/toolalpaca_eval_simulated/test.jsonl`
  - `data/toolalpaca_eval_real/test.jsonl`

源数据默认查找：

```text
references/ToolAlpaca/data
../ToolAlpaca/data
```

可用环境变量覆盖：

```bash
RWKV_TOOLALPACA_SOURCE_ROOT=/path/to/ToolAlpaca/data
TOOLALPACA_SOURCE_ROOT=/path/to/ToolAlpaca/data
```

源文件：

| dataset | source file |
|---|---|
| `toolalpaca_eval_simulated` | `eval_simulated.json` |
| `toolalpaca_eval_real` | `eval_real.json` |

simulated 执行：

```bash
TOOLALPACA_SIMULATOR_URL=http://127.0.0.1:5678
```

如果设置了 `TOOLALPACA_SIMULATOR_URL`，`ToolAlpacaHttpSandbox` 会调用官方 simulator：

```text
{TOOLALPACA_SIMULATOR_URL}/{urlencoded api_name}/{openapi path}
```

real 执行：

- `toolalpaca_eval_real` 直接调用真实外部 API。
- `server_url` 来自 ToolAlpaca OpenAPI `servers[0].url`。
- HTTP timeout 环境变量：`TOOLALPACA_HTTP_TIMEOUT_S`，默认 30 秒。

real 需要的 key：

```bash
TOOLALPACA_WEATHERSTACK_API_KEY=...
WEATHERSTACK_API_KEY=...

TOOLALPACA_WOLFRAMALPHA_APP_ID=...
WOLFRAMALPHA_APP_ID=...
WOLFRAM_ALPHA_APP_ID=...
WOLFRAM_APP_ID=...

TOOLALPACA_CURRENCYBEACON_API_KEY=...
CURRENCYBEACON_API_KEY=...
CURRENCY_BEACON_API_KEY=...
```

这些 key 的注入逻辑在 `src/eval/tasks/function_calling/toolalpaca.py`：

- `_TOOLALPACA_AUTH_ENV_BY_API`
- `_inject_toolalpaca_auth_placeholders`
- `_resolve_toolalpaca_secret_placeholders`

判分方式：

- expected calls 和 decoded calls 都会执行。
- 对比 normalized HTTP request、status、response payload。
- optional call 用 `__toolalpaca_optional__` 标记。
- 跨步骤引用用 `__toolalpaca_ref__` 标记，并从前一步 execution result 中解析。

## Inference 接口

另一个项目只要提供 OpenAI-compatible `/v1/completions` 即可。当前远端客户端在：

```text
src/infer/backend.py::RemoteInferenceBackend
```

请求 URL：

```text
{--infer-base-url}/v1/completions
```

请求 body 主要字段：

```json
{
  "model": "rwkv7-g1f-13.3b-20260415-ctx8192",
  "prompt": "...",
  "max_tokens": 1024,
  "temperature": 0.3,
  "top_k": 50,
  "top_p": 0.3,
  "presence_penalty": 0.5,
  "frequency_penalty": 0.5,
  "repetition_penalty": 0.5,
  "penalty_decay": 0.99,
  "stop_tokens": [0],
  "ban_tokens": [],
  "pad_zero": true,
  "no_penalty_token_ids": [33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],
  "seed": 123,
  "stop": ["\n```", "```", "\nUser:", "\nSystem:", "\nAssistant:"]
}
```

响应必须至少兼容：

```json
{
  "choices": [
    {
      "text": "{\"name\":\"...\",\"arguments\":{}}",
      "finish_reason": "stop"
    }
  ]
}
```

当前评测侧有 HTTP retry；推理服务本身不需要、也不应该在评测代码里实现自动重启。需要自动重连时只放在本地 SSH tunnel 层。

## DB 结果格式

schema 文件：

```text
scripts/schema.sql
```

必须保留的表：

| 表 | 用途 |
|---|---|
| `benchmark` | benchmark name/split 和样本数 |
| `model` | 模型身份 |
| `task` | 一次 model x dataset x evaluator run |
| `completions` | 每个样本的 prompt、completion、agent trace、metadata |
| `eval` | 每个 completion 的 answer/ref_answer/is_passed/fail_reason |
| `scores` | 一个 task 的最终 metrics |

`completions.context` 当前存的是完整 payload，核心字段：

```json
{
  "benchmark_name": "bfcl_exec_simple",
  "dataset_split": "test",
  "sample_index": 0,
  "repeat_index": 0,
  "pass_index": 0,
  "stages": [
    {
      "prompt": "...",
      "completion": "...",
      "stop_reason": "stop_token"
    }
  ],
  "agent_result": {
    "reward": 1.0,
    "num_turns": 1,
    "cost": 0.0,
    "is_passed": true,
    "error": null
  },
  "agent_info": {
    "decoded_tool_calls": [],
    "expected_tool_calls": [],
    "fail_reason": "",
    "cot_mode": "CoT"
  },
  "agent_trace": [],
  "task_id": "exec_simple_0",
  "domain": "function_call",
  "instruction": "...",
  "metadata": {}
}
```

`eval.answer` 是 decoded tool calls JSON，`eval.ref_answer` 是 expected tool calls JSON。注意：这两个字段只能在评测完成后写入，不能拼进 prompt。

`scores.metrics` 至少包含：

```json
{
  "success_rate": 0.98,
  "avg@1": 0.98
}
```

## 调度命令

完整复刻当前 APIBank + BFCL exec + ToolAlpaca 的 8 个 dataset：

```bash
.venv/bin/python -m src.eval.scheduler.cli dispatch \
  --infer-base-url http://127.0.0.1:19081 \
  --infer-models \
    rwkv7-g1e-13.3b-20260309-ctx8192 \
    rwkv7-g1e-7.2b-20260301-ctx8192 \
    rwkv7-g1f-13.3b-20260415-ctx8192 \
    rwkv7-g1f-7.2b-20260414-ctx8192 \
  --model-select all \
  --only-jobs function_api_bank function_bfcl_exec function_toolalpaca \
  --only-datasets \
    apibank_level1 apibank_level2 \
    bfcl_exec_simple bfcl_exec_multiple bfcl_exec_parallel bfcl_exec_parallel_multiple \
    toolalpaca_eval_simulated toolalpaca_eval_real \
  --max-concurrent-jobs 4 \
  --infer-max-workers 128 \
  --remote-batch-size 128 \
  --run-mode rerun \
  --disable-checker \
  --function-prompt-style rwkv_official_json \
  --function-tool-catalog-format json \
  --function-decision-max-tokens 1024 \
  --function-final-max-tokens 3072 \
  --function-max-steps 20 \
  --function-max-rounds 20
```

只跑六个核心 dataset：

```bash
.venv/bin/python -m src.eval.scheduler.cli dispatch \
  --infer-base-url http://127.0.0.1:19081 \
  --infer-models \
    rwkv7-g1e-13.3b-20260309-ctx8192 \
    rwkv7-g1e-7.2b-20260301-ctx8192 \
    rwkv7-g1f-13.3b-20260415-ctx8192 \
    rwkv7-g1f-7.2b-20260414-ctx8192 \
  --model-select all \
  --only-jobs function_bfcl_exec function_toolalpaca \
  --only-datasets \
    bfcl_exec_simple bfcl_exec_multiple bfcl_exec_parallel bfcl_exec_parallel_multiple \
    toolalpaca_eval_simulated toolalpaca_eval_real \
  --max-concurrent-jobs 4 \
  --infer-max-workers 128 \
  --remote-batch-size 128 \
  --run-mode rerun \
  --disable-checker \
  --function-prompt-style rwkv_official_json \
  --function-tool-catalog-format json \
  --function-decision-max-tokens 1024 \
  --function-final-max-tokens 3072 \
  --function-max-steps 20 \
  --function-max-rounds 20
```

单 dataset 调试命令：

```bash
.venv/bin/python -m src.eval.tasks.function_calling.runner \
  --dataset data/bfcl_exec_simple/test.jsonl \
  --infer-base-url http://127.0.0.1:19081 \
  --infer-model rwkv7-g1f-13.3b-20260415-ctx8192 \
  --infer-timeout-s 600 \
  --infer-max-workers 128 \
  --batch-size 128 \
  --prompt-style rwkv_official_json \
  --tool-catalog-format json \
  --decision-max-tokens 1024 \
  --final-max-tokens 3072 \
  --max-rounds 20 \
  --max-steps 20
```

## 数据准备命令

如果 `data/*/test.jsonl` 已经存在，可以直接复制这些 prepared JSONL。若另一个项目要从官方源数据重新 materialize，使用同名 prepare API：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from src.eval.datasets.data_prepper.data_manager import prepare_dataset

for name in [
    "apibank_level1",
    "apibank_level2",
    "bfcl_exec_simple",
    "bfcl_exec_multiple",
    "bfcl_exec_parallel",
    "bfcl_exec_parallel_multiple",
    "toolalpaca_eval_simulated",
    "toolalpaca_eval_real",
]:
    paths = prepare_dataset(name, Path("data"), "test")
    print(name, [str(path) for path in paths])
PY
```

每个 prepared 数据集会生成：

```text
data/<dataset>/test.jsonl
data/<dataset>/test.jsonl.manifest.json
```

manifest 里会记录 `source_kind`、`row_count` 和源文件路径。

## 最小测试清单

复刻项目至少要覆盖这些测试点：

| 测试目标 | 当前项目参考测试 |
|---|---|
| prompt 使用 `rwkv_official_json` 格式 | `tests/test_function_calling_common.py` |
| APIBank / BFCL / ToolAlpaca 数据准备 | `tests/test_function_calling_dataset_prep.py` |
| BFCL exec 解析与执行判分 | `tests/test_function_calling_bfcl_v3.py` |
| function-calling runner 分发 | `tests/test_function_calling_runner.py` |
| scheduler job matrix | `tests/test_scheduler_job_matrix.py` |
| runner kind auto-detect | `tests/test_function_calling_runner.py` |

当前项目可用的快速验证命令：

```bash
.venv/bin/pytest \
  tests/test_function_calling_common.py \
  tests/test_function_calling_dataset_prep.py \
  tests/test_function_calling_bfcl_v3.py \
  tests/test_function_calling_runner.py \
  tests/test_scheduler_job_matrix.py
```

## 分数查询

大规模跑完后，按 latest task 展示分数时不要取历史最大值。应按 `scores.created_at` 取最新记录。

参考 SQL：

```sql
WITH latest AS (
  SELECT DISTINCT ON (m.model_name, b.benchmark_name, b.benchmark_split)
    m.model_name,
    b.benchmark_name || '_' || b.benchmark_split AS dataset,
    s.metrics,
    s.created_at,
    t.task_id
  FROM scores s
  JOIN task t ON t.task_id = s.task_id
  JOIN model m ON m.model_id = t.model_id
  JOIN benchmark b ON b.benchmark_id = t.benchmark_id
  WHERE b.benchmark_name IN (
    'apibank_level1',
    'apibank_level2',
    'bfcl_exec_simple',
    'bfcl_exec_multiple',
    'bfcl_exec_parallel',
    'bfcl_exec_parallel_multiple',
    'toolalpaca_eval_simulated',
    'toolalpaca_eval_real'
  )
  ORDER BY m.model_name, b.benchmark_name, b.benchmark_split, s.created_at DESC
)
SELECT
  dataset,
  model_name,
  ROUND((COALESCE((metrics->>'success_rate')::float, (metrics->>'avg@1')::float) * 100)::numeric, 2) AS score,
  task_id,
  created_at
FROM latest
ORDER BY dataset, model_name;
```

## 防作弊检查

另一个项目必须保证：

- `stages[0].prompt` 不包含 `expected_tool_calls`。
- `stages[0].prompt` 不包含 `ref_answer`。
- `stages[0].prompt` 不包含 `ground_truth`。
- `eval.ref_answer` 只在评测后生成。
- `completions.context.agent_info.expected_*` 只在评测后生成。

抽查 SQL：

```sql
SELECT COUNT(*) AS suspicious_prompt_rows
FROM completions c
JOIN task t ON t.task_id = c.task_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
WHERE b.benchmark_name IN (
  'apibank_level1',
  'apibank_level2',
  'bfcl_exec_simple',
  'bfcl_exec_multiple',
  'bfcl_exec_parallel',
  'bfcl_exec_parallel_multiple',
  'toolalpaca_eval_simulated',
  'toolalpaca_eval_real'
)
AND (
  c.context #>> '{stages,0,prompt}' ILIKE '%expected_tool_calls%'
  OR c.context #>> '{stages,0,prompt}' ILIKE '%ref_answer%'
  OR c.context #>> '{stages,0,prompt}' ILIKE '%ground_truth%'
  OR c.context #>> '{stages,0,prompt}' ILIKE '%Golden_Answers%'
);
```

正常结果应该是 `0`。

## 迁移实现建议

另一个项目如果不想复刻完整 scheduler，最少需要复制这些模块的行为：

1. 数据准备：`src/eval/datasets/data_prepper/function_calling/*.py`
2. prompt 和 decode：`src/eval/tasks/function_calling/rwkv_prompt.py`、`src/eval/tasks/function_calling/simple_tool_call.py`
3. evaluator：
   - `src/eval/tasks/function_calling/api_bank.py`
   - `src/eval/tasks/function_calling/bfcl_exec.py`
   - `src/eval/tasks/function_calling/toolalpaca.py`
   - `src/eval/tasks/function_calling/toolalpaca_source.py`
4. 远端推理协议：`src/infer/backend.py::RemoteInferenceBackend`
5. DB schema：`scripts/schema.sql`
6. scheduler job 命名：
   - `function_api_bank`
   - `function_bfcl_exec`
   - `function_toolalpaca`

大规模服务器运行时，建议把“推理服务”和“评测调度”分离：

- 服务器只跑 OpenAI-compatible `/v1/completions` 推理端。
- 本地或调度节点跑 evaluator / sandbox / DB 写入。
- SSH tunnel 可以自动重连。
- 推理端本身不要在 benchmark 代码里做自动重启。
