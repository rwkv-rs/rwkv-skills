# 多轮 Agent Benchmark 复测 Runbook

日期：2026-05-14

用途：记录后续用微调权重复测多轮 agent/function-calling benchmark 的启动方式。本文不记录任何 API key 或数据库密码。

## 当前停服状态

当前中转服务器：

```bash
ssh chase@47.115.88.183 -p 8222
```

本次已停止 `/home/chase/chase-rwkv-skills` 下的 Python 推理端：

- `2467846`：`run_infer_router`，监听 `127.0.0.1:19081`
- `2467478`：`run_infer_server`，监听 `127.0.0.1:18081`
- `2467567`：`run_infer_server`，监听 `127.0.0.1:18082`
- `2467650`：`run_infer_server`，监听 `127.0.0.1:18083`
- `2467735`：`run_infer_server`，监听 `127.0.0.1:18084`

确认结果：

- `ss -ltnp | grep -E ':19081|:1808[1-4]'` 无输出
- `nvidia-smi` 中无 `/home/chase/chase-rwkv-skills/.venv/bin/python` 推理进程

未停止其他无关进程，例如 `vllm::Worker` 和 `target/release/examples/rwkv-lm-eval`。

## 停止远端推理端

先查看相关进程和端口：

```bash
ssh chase@47.115.88.183 -p 8222
ps -eo pid,ppid,stat,etime,args | grep -E 'run_infer_server|run_infer_fleet|run_infer_router|uvicorn|19081|rwkv' | grep -v grep
ss -ltnp | grep -E ':19081|:18081|:18082|:18083|:18084' || true
nvidia-smi
```

优先用 SIGTERM 正常停止 router 和 per-model infer server：

```bash
kill -TERM <PID1> <PID2> ...
sleep 10
ps -eo pid,ppid,stat,etime,args | grep -E 'run_infer_server|run_infer_router|uvicorn|19081|rwkv' | grep -v grep
```

如果还没退出，再用：

```bash
kill -KILL <PID>
```

确认端口已关闭：

```bash
ss -ltnp | grep -E ':19081|:18081|:18082|:18083|:18084' || echo "infer service stopped"
```

## 启动远端推理端

单模型服务：

```bash
cd ~/GitHub/rwkv-skills
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.bin.run_infer_server \
  --model-path /path/to/finetuned-model.pth \
  --model-name finetuned-rwkv-agent-ctx32768 \
  --device cuda:0 \
  --engine-mode classic \
  --host 127.0.0.1 \
  --port 19081 \
  --max-batch-size 8 \
  --batch-collect-ms 10 \
  --log-level info
```

多模型推荐每个模型一个后端端口，再用 router 汇总到 `19081`：

```bash
cd ~/GitHub/rwkv-skills
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.bin.run_infer_server \
  --model-path /path/to/model-a.pth \
  --model-name model-a \
  --device cuda:0 \
  --engine-mode classic \
  --host 127.0.0.1 \
  --port 18081 \
  --max-batch-size 8
```

```bash
cd ~/GitHub/rwkv-skills
CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.bin.run_infer_server \
  --model-path /path/to/model-b.pth \
  --model-name model-b \
  --device cuda:0 \
  --engine-mode classic \
  --host 127.0.0.1 \
  --port 18082 \
  --max-batch-size 8
```

```bash
cd ~/GitHub/rwkv-skills
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.bin.run_infer_router \
  --route model-a=http://127.0.0.1:18081 \
  --route model-b=http://127.0.0.1:18082 \
  --host 127.0.0.1 \
  --port 19081 \
  --timeout-s 600
```

本地转发：

```bash
autossh -M 0 -N \
  -L 19081:127.0.0.1:19081 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -p 8222 chase@47.115.88.183
```

本地确认：

```bash
curl http://127.0.0.1:19081/v1/models
```

## 多轮 Benchmark 范围

重点复测这些 job/dataset：

| job | dataset |
| --- | --- |
| `function_bfcl_v3` | `bfcl_v3_test` |
| `function_tau_bench` | `tau_bench_airline_test`, `tau_bench_retail_test`, `tau_bench_telecom_test` |
| `function_tau2_bench` | `tau2_bench_airline_base`, `tau2_bench_retail_base`, `tau2_bench_telecom_base` |
| `function_mcp_bench` | `mcp_bench_test` |

`bfcl_v3_test` 里包含 BFCL v3 多轮长上下文数据，尤其需要关注 `multi_turn_long_context`、state mismatch、tool execution error、max_steps。

## Scheduler 复测命令

远端 router 走本地 `19081` 转发时：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.eval.scheduler.cli dispatch \
  --infer-base-url http://127.0.0.1:19081 \
  --infer-models \
    finetuned-rwkv-agent-ctx32768 \
  --model-select all \
  --only-jobs \
    function_bfcl_v3 \
    function_tau_bench \
    function_tau2_bench \
    function_mcp_bench \
  --only-datasets \
    bfcl_v3_test \
    tau_bench_airline_test \
    tau_bench_retail_test \
    tau_bench_telecom_test \
    tau2_bench_airline_base \
    tau2_bench_retail_base \
    tau2_bench_telecom_base \
    mcp_bench_test \
  --function-prompt-style rwkv_official_json \
  --function-tool-catalog-format json \
  --function-history-max-chars 24000 \
  --function-max-steps 20 \
  --function-max-tool-errors 20 \
  --function-cot-max-tokens 2048 \
  --function-decision-max-tokens 1024 \
  --function-planning-max-tokens 2048 \
  --function-final-max-tokens 3072 \
  --remote-batch-size 64 \
  --infer-timeout-s 600 \
  --infer-max-workers 64 \
  --max-concurrent-jobs 4 \
  --run-mode rerun
```

多模型时把 `--infer-models` 扩展为 router 暴露的多个模型名。

## 小样本 Smoke Test

配置文件位于：

- `configs/run/bfcl_v3.toml`
- `configs/run/tau_bench_airline.toml`
- `configs/run/tau_bench_retail.toml`
- `configs/run/tau_bench_telecom.toml`
- `configs/run/tau2_bench_airline.toml`
- `configs/run/tau2_bench_retail.toml`
- `configs/run/tau2_bench_telecom.toml`
- `configs/run/mcp_bench.toml`

这些配置默认 `max_samples = 50`，适合先 smoke test。注意文件里默认 `infer_base_url = "http://127.0.0.1:18081"`；如果使用 SSH router 转发，需要改成 `http://127.0.0.1:19081` 或复制一份本地临时配置。

运行单个配置：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.main --config configs/run/bfcl_v3.toml
```

先 dry-run 看实际 runner 命令：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.main --config configs/run/bfcl_v3.toml --dry-run
```

## 结果判断

简单 BFCL/ToolAlpaca 当前已有高分基线，不适合作为多轮长上下文能力判断：

- `bfcl_simple_python_test`：约 `0.86-0.905`
- `bfcl_multiple_test`：约 `0.815-0.87`
- `bfcl_exec_simple_ast_test`：约 `0.80-0.92`（仅 AST/JSON 参数匹配）
- `bfcl_exec_multiple_ast_test`：约 `0.78-0.86`（仅 AST/JSON 参数匹配）
- `bfcl_exec_simple_test` / `bfcl_exec_multiple_test` / `bfcl_exec_parallel_test` / `bfcl_exec_parallel_multiple_test`：走 executable runner，按 BFCL 可执行调用结果判分
- `toolalpaca_eval_real_test`：约 `0.41-0.54`

多轮重点看：

- `function_bfcl_v3 / bfcl_v3_test`
- `max_steps`
- `too_many_errors`
- `invalid_decision_output`
- `unknown tool name`
- `instance_state_mismatch`
- `execution_response_mismatch`

当前未做长上下文 agent 后训练的 8192 ctx 模型在 BFCL v3 多轮上出现 `0.x%` 是合理弱项表现；微调后如果能稳定超过 `5%-10%`，说明训练方向已经有效。
