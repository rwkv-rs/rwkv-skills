# Function-Calling Benchmarks 说明

## 总览

| rwkv-skills 显示名 | scheduler dataset | scheduler job | 原始来源 | 当前本地评测形态 |
|---|---|---|---|---|
| `bfcl_simple_python_nocot` | `bfcl_simple_python_test` | `function_one_step_bfcl_ast` | BFCL v4 `simple_python` | 单轮 JSON tool-call exact match |
| `bfcl_multiple_nocot` | `bfcl_multiple_test` | `function_one_step_bfcl_ast` | BFCL v4 `multiple` | 单轮 JSON tool-call exact match |
| `bfcl_exec_simple_nocot` | `bfcl_exec_simple_test` | `function_one_step_bfcl_exec` | BFCL v4 `unused_datasets/question/BFCL_v4_exec_simple.json` | 单轮 JSON tool-call + 本地 BFCL executable scorer |
| `bfcl_exec_multiple_nocot` | `bfcl_exec_multiple_test` | `function_one_step_bfcl_exec` | BFCL v4 `unused_datasets/question/BFCL_v4_exec_multiple.json` | 单轮 JSON tool-call + 本地 BFCL executable scorer |
| `toolalpaca_eval_simulated_nocot` | `toolalpaca_eval_simulated_test` | `function_one_step_toolalpaca` | ToolAlpaca `data/eval_simulated.json` | 单轮 JSON tool-call exact match |
| `toolalpaca_eval_real_nocot` | `toolalpaca_eval_real_test` | `function_one_step_toolalpaca` | ToolAlpaca `data/eval_real.json` | 单轮 JSON tool-call exact match |

## 原始 Benchmark
### BFCL

论文/项目：

- Paper: The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models, ICML 2025.
- Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard
- Code: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- Dataset: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- Public result archive: https://github.com/HuanzhiMao/BFCL-Result

BFCL 官方说明里，BFCL 用来评估模型准确调用 functions/tools 的能力，覆盖不同函数调用形式、多语言和可执行场景。官方 v4 leaderboard 还区分 native function/tool calling (`FC`) 和 prompt-only 方式。

本地接入点：

- `src/eval/datasets/data_prepper/function_call/bfcl_toolalpaca.py`
- `src/eval/function_calling/one_step/simple_tool_call.py`
- `src/eval/scheduler/jobs.py`
- `configs/bfcl_simple_python.toml`
- `configs/bfcl_multiple.toml`
- `configs/bfcl_exec_simple.toml`
- `configs/bfcl_exec_multiple.toml`

本地转换：

- `bfcl_simple_python` 从 `BFCL_v4_simple_python.json` 和 `possible_answer/BFCL_v4_simple_python.json` 生成。
- `bfcl_multiple` 从 `BFCL_v4_multiple.json` 和 `possible_answer/BFCL_v4_multiple.json` 生成。
- `bfcl_exec_simple` 从 `unused_datasets/question/BFCL_v4_exec_simple.json` 和 `unused_datasets/possible_answer/BFCL_v4_exec_simple.json` 生成。
- `bfcl_exec_multiple` 从 `unused_datasets/question/BFCL_v4_exec_multiple.json` 和 `unused_datasets/possible_answer/BFCL_v4_exec_multiple.json` 生成。

本地和官方的关键差异：

- 当前 `rwkv-skills` 没有使用 BFCL 官方 prompt formatter、官方 AST checker 或 executable checker。
- `simple_python` / `multiple` 当前本地转成 `simple_tool_call`：模型只输出 JSON function call，scorer 比较函数名和参数。
- `exec_simple` / `exec_multiple` 当前转成 `bfcl_exec` scorer：先解析模型 JSON function call，再执行本地内置 BFCL 函数；未覆盖的外部/API 函数使用参数身份回退，因此仍不是 BFCL 官方完整 executable runtime。
- BFCL 的 `multiple` 表示给定多个候选函数，不必然表示需要多个 tool calls；只有题目确实需要多个调用时才应该输出 JSON array。

### ToolAlpaca

论文/项目：

- Paper: ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases.
- arXiv: https://arxiv.org/abs/2306.05301
- Repo: https://github.com/tangqiaoyu/ToolAlpaca

原始数据：

- `eval_simulated.json`: 10 个模拟 API 的 evaluation data。
- `eval_real.json`: 11 个真实 API 的 evaluation data，部分真实 API 需要认证。
- 原 repo 的 evaluation 使用模型输出、工具文档和标准答案，再用 GPT-4 等 LLM 做 procedure/response/overall 评估。

本地接入点：

- `src/eval/datasets/data_prepper/function_call/bfcl_toolalpaca.py`
- `src/eval/function_calling/one_step/simple_tool_call.py`
- `src/eval/scheduler/jobs.py`
- `configs/toolalpaca_eval_simulated.toml`
- `configs/toolalpaca_eval_real.toml`

本地转换：

- 每个 API 下的 `Instructions` 和 `Golden_Answers` 被展开为多条单轮任务。
- `Function_Description` / `Function_Projection` 转成 tool schema。
- `Golden_Answers` 中的 `Action` / `Action_Input` 转成 `expected_tool_calls`。

本地和原论文的关键差异：

- 原 ToolAlpaca 是多步工具使用任务，并用 GPT-4/人工评价 procedure、response、overall。
- 当前 `rwkv-skills` 不执行真实 API，不使用 GPT-4 judge，不评价 final natural-language response。
- 当前只要求模型输出一次 JSON tool call 或一个 JSON array，然后 exact match 函数名和参数。

## 数据记录格式

当前 6 个数据集最终都会生成 JSONL，每行核心字段为：

```json
{
  "task_id": "string",
  "instruction": "string",
  "tools": [
    {
      "name": "tool_name",
      "description": "tool description",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      }
    }
  ],
  "expected_tool_calls": [
    {
      "name": "tool_name",
      "arguments": {},
      "argument_options": {}
    }
  ],
  "env": {"type": "simple_tool_call"},
  "scorer": {"type": "simple_tool_call | bfcl_exec"},
  "metadata": {}
}
```

Loader 要求：

- `expected_tool_calls` 必须存在且非空。
- `env.type` 只能是 `simple_tool_call`。
- `scorer.type` 只能是 `simple_tool_call` 或 `bfcl_exec`。
- 如果没有 `messages`，会从 `instruction` 生成一个 user message。

## 题目示例

### BFCL simple_python

题目形态：

```text
User wants to calculate the area under y=x^2 from x=1 to x=3.
Available tool: calculate_area_under_curve(function, interval, method?)
```

当前期望输出形态：

```json
{"name":"calculate_area_under_curve","arguments":{"function":"x**2","interval":[1.0,3.0]}}
```

### BFCL multiple

题目形态：

```text
User asks how to assess deer population growth and woodland impact in Washington over the past decade.
Multiple candidate tools are shown, but the expected answer may still be one selected function call.
```

当前期望输出形态：

```json
{"name":"wildlife_population.assess_growth","arguments":{"species":"deer","location":"Washington","duration":10}}
```

### ToolAlpaca simulated / real

题目形态：

```text
Tool documentation describes an API such as Public Holidays.
User asks for a task such as planning a trip while avoiding holidays.
The model must choose the proper action and arguments from the provided API functions.
```

当前期望输出形态：

```json
{"name":"getHolidays","arguments":{"country":"Japan","year":2024}}
```

这些示例用于说明本地格式，不保证是当前数据文件里的完整原样记录。

## 当前上下文组装

入口：`src/eval/function_calling/one_step/pipeline.py::FunctionCallPipeline._make_prompt()`。

组装流程：

1. 把 `FunctionCallTaskRecord` 转成 `SimpleToolCallRecord`。
2. `instruction` 优先使用数据行里的 `instruction`；如果为空，才把 `messages` 渲染成 history。
3. 调用 `build_simple_tool_call_prompt()` 构建单轮 prompt。
4. 工具列表通过 `_render_tool_catalog()` 渲染成 JSON array。
5. 最终通过 `build_rwkv_json_call_prompt()` 组装成 RWKV chat 文本。

当前 prompt 结构：

```text
System: Tools:
[
  {
    "arguments": {
      "...": {
        "description": "...",
        "type": "..."
      }
    },
    "description": "...",
    "name": "..."
  }
]
Return only a JSON function call.
The JSON shape is {"name":"tool_name","arguments":{...}}.
If multiple tool calls are required, return a JSON array of those objects in execution order.
Use only listed tool names.

User: <instruction>

Assistant: ```json
```

工具 catalog 约束：

- tool description 最多保留 700 chars。
- tool schema 最多保留 1200 chars，超出会写 `_truncated` preview。
- `parameters.properties` 被渲染为 `arguments`。
- history 最大 24000 chars，但当前 6 个 benchmark 通常是单轮 instruction。

生成停止：

````text
\n```
```
\nUser:
\nSystem:
\nAssistant:
````

生成完成后：

- completion 会先裁掉 stop suffix。
- `final_answer = completion.strip()`。
- DB payload 写入 `prompt1`, `completion1`, `final_answer`, `events`, `stats`。

## 当前采样配置

所有 6 个 benchmark 都配置：

```toml
[default]
avg_k = [1]
report_avg_k = [1]
```

BFCL 四项和 `toolalpaca_eval_simulated` 当前 `[tool]` 配置：

```toml
template = "instruction_following_default"
max_generate_tokens = 1024
temperature = 0.2
top_k = 20
top_p = 0.2
```

`toolalpaca_eval_real` 当前 `[tool]` 配置不同：

```toml
template = "function_call_default"
max_generate_tokens = 1024
temperature = 1.0
top_k = 200
top_p = 0.0
alpha_presence = 0.0
alpha_frequency = 0.0
alpha_decay = 0.99
```

这个不对称是后续统一约束解码前需要确认的配置差异。

## 当前判分器

入口：`evaluate_function_call()` -> `_score_prediction()`。

one-step 当前有两个 scorer：

- `simple_tool_call`: `evaluate_simple_tool_calls()`，用于 `bfcl_simple_python`、`bfcl_multiple`、ToolAlpaca 两项。
- `bfcl_exec`: `evaluate_bfcl_executable_calls()`，用于 `bfcl_exec_simple`、`bfcl_exec_multiple`。

解析规则：

1. 优先从 payload 的 `final_answer` 取模型输出。
2. 如果没有，才从 `context.events` 或最后一个 `completionN` 回退。
3. 拒绝明显 prompt/template leak。
4. `decode_simple_tool_call_response()` 要求输出是完整 JSON object 或 JSON array。
5. decoder 可接受：
   - `{"name":"tool","arguments":{...}}`
   - `[{"name":"tool","arguments":{...}}, ...]`
   - `{"tool_calls":[...]}`
   - OpenAI-like nested `{"function":{"name":"...","arguments":{...}}}`
6. `arguments` 字符串必须能解析成 JSON object，否则失败。
7. JSON 后面不能有额外自然语言。

比较规则：

- tool call 数量必须和 expected 数量一致。
- 多个 tool calls 按顺序比较。
- 函数名必须完全一致。
- `arguments` 必须是 object。
- expected 的每个 argument 都必须存在，除非它的可选值包含 absent option：`null`, `""`, `{}`, `[]`。
- actual 里出现 expected 之外的非空参数会失败。
- 数字按 `1e-9` 容差比较。
- 字符串做 RWKV 文本 normalize 后精确比较。
- 字符串和非字符串之间会尝试 JSON scalar parse 后再比较。

`bfcl_exec` 额外规则：

- 从 `scorer.ground_truth` 或 `metadata.bfcl_ground_truth` 读取 BFCL ground-truth call expression。
- 对模型 call 和 ground-truth call 分别执行本地函数，再比较执行结果。
- 支持 `exact_match`、`structural_match`、`real_time_match`。
- 当前内置了常见 BFCL deterministic math/list 函数，例如 `calc_binomial_probability`、`calculate_cosine_similarity`、`calculate_density`、`calculate_displacement`、`calculate_mean`、`mat_mul`、`sort_array` 等。
- 未覆盖的函数使用 `argument_identity_fallback`，即退化为函数名和参数身份比较；这保证 benchmark 能完整跑完，但不是官方完整 executable runtime。

输出指标：

- `success_rate`
- `avg@1`
- `avg_steps`
- `avg_tool_calls`
- `env_breakdown`

当前 `steps` 固定为 1，`tool_calls` 统计当前 payload 里仍为 0；主要有效指标是 `success_rate` / `avg@1`。

## 约束解码迁移要点

后续迁移约束解码时，当前 scorer 最需要保证的是：

1. 输出必须是一个 JSON value，且第一个有效字符是 `{` 或 `[`。
2. 单调用优先使用：

```json
{"name":"tool_name","arguments":{}}
```

3. 多调用使用：

```json
[
  {"name":"tool_a","arguments":{}},
  {"name":"tool_b","arguments":{}}
]
```

4. `name` 必须被限制在当前 tool catalog 的 tool names 里。
5. `arguments` 必须是 object，不要输出字符串形式的 arguments。
6. 不要输出 `<think>`、解释文字、Markdown 结束 fence 或额外 assistant/user/system 标记。
7. 如果沿用当前 prompt 的 `Assistant: ```json\n` 前缀，约束解码只需要生成 JSON value 本体；结束 fence 可以不生成。
8. 如果未来改成无 fence prompt，需要同步检查 `extract_json_call_value_text()` 是否仍符合预期。

## 7B / 8B 参考分数

### BFCL 公开参考

BFCL 官方结果库能逐项看到公开模型分数。例如 2025-12-16 的官方结果中，`Qwen_Qwen3-8B-FC` 在 `simple_python` 是 `382/400 = 95.5%`，在 `multiple` 是 `193/200 = 96.5%`。这些是官方 BFCL prompt/decoder/checker 下的结果，不等价于当前 `rwkv-skills` 的 simplified JSON exact scorer。

另外，WizWand 汇总的 BFCL Non-Live Out-of-Domain 表给出了接近 7B/8B 模型的分项参考，包含 `Execute Simple` / `Execute Multiple`：

| Model | Params | AST Simple | AST Multiple | Execute Simple | Execute Multiple | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Hammer2.1-7B | 7B | 78.1 | 95.0 | 86.4 | 92.0 | 81.6 |
| Qwen3-8B | 8B | 76.8 | 95.5 | 94.0 | 92.0 | 83.4 |
| xLAM-7B-r | 7B | 74.2 | 95.5 | 74.0 | 96.0 | 76.6 |
| Qwen2.5-7B-Instruct | 7B | 71.8 | 95.0 | 95.4 | 94.0 | 79.6 |

### ToolAlpaca 公开参考

ToolAlpaca 论文 Table 3 给出的 7B 参考：

| Model | Sim Procedure | Sim Response | Sim Overall | Sim Human | Real Procedure | Real Response | Real Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vicuna-7B | 19.0 | 21.0 | 17.0 | 16.0 | 7.9 | 11.4 | 7.9 |
| ToolAlpaca-7B | 63.0 | 69.0 | 60.0 | 73.0 | 63.2 | 57.9 | 55.3 |
