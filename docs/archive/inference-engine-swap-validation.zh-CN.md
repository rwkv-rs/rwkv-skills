# 推理引擎替换验证 Runbook

日期：2026-06-07

用途：记录本轮远端推理引擎替换到 `nano-vllm-contents` 协议后的分阶段验证、8222 当前服务状态、并发 probe 证据，以及正式测评前的固定参数。本文不记录 API key 或数据库密码。

完成边界审计见：`docs/archive/inference-engine-swap-readiness-audit.zh-CN.md`。

## 范围

当前替换目标：

- 调度侧可通过 `--infer-protocol nano-vllm-contents` 切到 nano-style `contents` batch 协议。
- 默认 seed 行为保持兼容；使用 `--infer-seed-policy omit-for-contents` 时，contents batch 会显式丢弃 prompt seed，以换取可批量发送。
- `/v1/chat/completions` 同时支持 OpenAI `messages` 请求和 nano-style `contents` 请求。
- router 对同模型多副本可以把一个 `contents` batch 拆给多个后端，再按原始 index 合并。
- scheduler 的 `probe-infer` 能输出吞吐峰值、GPU 满载建议、最大健康并发和推荐 scheduler 参数。
- fleet 可以用 `--replicas-per-model` 起同模型多副本，并用 `--router-port` 一次性启动 router。

## 阶段 1：协议替换单元测试

本阶段验证本地代码语义，不依赖 8222。

```bash
rtk uv run pytest -q \
  tests/test_infer_backend.py \
  tests/test_run_infer_server.py \
  tests/test_scheduler_remote_inference.py \
  tests/test_main_config.py \
  tests/test_infer_router.py \
  tests/test_infer_fleet.py \
  tests/test_preflight_remote_eval.py \
  tests/test_probe_remote_infer.py
```

当前验证结果：

- 当前保留的是远端推理入口 smoke 集合；历史 infer-swap formal guard 测试已清理。
- `rtk uv run python -m compileall -q ...` 通过
- `rtk git diff --check` 通过
- `rtk uv run python -m src.bin.validate_infer_swap_phases --phase-timeout-s 30` smoke：`ok=true phases=6`
- refreshed canonical bundle：`ok=true readiness_ok=true phase_gate_ok=true`，`phase_timeout_s=30.0`

覆盖点：

- `RemoteInferenceBackend` 的 `openai` 与 `nano-vllm-contents` 两条协议路径。
- contents batch 的 stop suffix 分组、choice index 合并、seed policy。
- server 端 `ContentsChatCompletionRequest` 到内部 `CompletionRequest` 的转换。
- router 对同模型多 backend 的 batch 拆分与合并。
- scheduler/admin/main config 对 `infer_protocol`、`infer_seed_policy`、`remote_batch_size` 的透传。
- fleet 多副本部署和可选 router 启动。
- `verify_remote_infer_swap` 双协议 smoke compare。
- `probe_remote_infer` 与 `preflight_remote_eval` 两个轻量入口。
- `run_infer_swap_eval` 正式测评 helper 的 preflight/queue/dispatch 参数构造。
- `run_infer_swap_eval` dispatch gate 对 phase gate 中的双协议 smoke、`gpu_full_concurrency` / `largest_successful_concurrency` 覆盖正式 profile 做复核；同时对 bundle 中 tunnel、queue、dispatch、summary/watch、evidence、speedup doc 命令、speedup output path、phase timeout 和 summary watch interval metadata 做同身份/完整性校验；bundle 中的 speedup doc 命令不能带 `--allow-incomplete`。
- `summarize_infer_swap_eval` 正式测评后的只读 DB 汇总，不伪造缺失 score。
- `audit_infer_swap_readiness` 正式启动前的一键只读审计，串联 preflight、queue、DB summary 和 GPU probe evidence。
- `draft_infer_swap_speedup_doc` 分数齐全后的提速设计文档生成；默认拒绝未完成 summary，并要求每个正式 dataset 的 latest task 为 `Completed`、有 `score_id` 和真实 `metrics`；同时复核 readiness 的 queue、GPU probe 和 `2/2` 双协议 batched smoke，并在 Evidence Gate 记录 phase timeout、summary watch interval 与 summary watch 命令。
- `prepare_infer_swap_launch_bundle` 非 dispatch 启动证据包，一次刷新 readiness/summary/evidence/phase gate 默认路径。
- `validate_infer_swap_phases` 非 dispatch 分阶段 gate，串联上述单测、compile、`git diff --check` 和可选 readiness/probe/summary evidence；每个命令阶段有超时保护，避免 gate 无限挂住。

需要重新跑完整非 dispatch 启动证据包时：

```bash
rtk uv run python -m src.bin.prepare_infer_swap_launch_bundle \
  --probe-json /tmp/rwkv-skills-infer-swap-probe-19082-gpu0-lightweight-full-20260607.json \
  --readiness-output-json /tmp/rwkv-skills-infer-swap-readiness-audit.json \
  --summary-output-json /tmp/rwkv-skills-infer-swap-eval-summary.json \
  --phase-gate-output-json /tmp/rwkv-skills-infer-swap-phase-gate.json \
  --bundle-output-json /tmp/rwkv-skills-infer-swap-launch-bundle.json \
  --phase-timeout-s 30
```

该命令不会启动正式测评；它先刷新 readiness audit、summary、evidence Markdown，再运行分阶段单测、compile、`git diff --check` 并生成 phase gate schema v2。bundle JSON 会记录本次非密钥 launch identity：profile、base URL、model、workers、remote batch、jobs、datasets、phase timeout、summary watch interval、speedup output path 和 DB target。正式分数完成后再跑 `validate_infer_swap_phases --require-summary-all-scored`，用于提速设计文档前的最终 gate。

## 阶段 2：真实 endpoint 协议对比

如果本地没有直连 8222 router，先开临时端口转发：

```bash
rtk ssh -p 8222 -N \
  -L 29082:127.0.0.1:19082 \
  -o ExitOnForwardFailure=yes \
  -o BatchMode=yes \
  chase@47.115.88.183
```

另一个终端执行：

```bash
rtk uv run python -m src.bin.verify_remote_infer_swap \
  --infer-base-url http://127.0.0.1:29082 \
  --infer-model rwkv7-g1g-2.9b-20260526-ctx8192 \
  --protocols openai,nano-vllm-contents \
  --batch-size 2 \
  --max-tokens 16 \
  --output-path /tmp/rwkv-skills-infer-swap-verify-19082.json
```

当前验证证据：

- 证据文件：`/tmp/rwkv-skills-infer-swap-verify-19082.json`
- `ok = true`
- `openai`：`2/2` 非空输出，约 `0.804s`
- `nano-vllm-contents`：`2/2` 非空输出，约 `0.710s`

这一步只证明 endpoint 协议和结果形状可用，不替代正式 benchmark 分数。

## 阶段 3：8222 当前服务拓扑

权威主机：

```bash
rtk ssh chase@47.115.88.183 -p 8222
```

当前健康检查：

```bash
rtk ssh -p 8222 -o BatchMode=yes -o ConnectTimeout=10 chase@47.115.88.183 \
  'curl -fsS http://127.0.0.1:19082/healthz'
```

当前结果：

```json
{"status":"ok","models":["rwkv7-g1g-2.9b-20260526-ctx8192"],"route_counts":{"rwkv7-g1g-2.9b-20260526-ctx8192":3}}
```

当前 router：

- `http://127.0.0.1:19082`
- model：`rwkv7-g1g-2.9b-20260526-ctx8192`
- route count：`3`
- 后端端口：`18082`、`18085`、`18086`

当前 GPU 快照：

```text
0, 81408, 97887, 0
1, 26674, 97887, 58
2, 26674, 97887, 58
3, 15120, 97887, 0
```

含义：GPU0 正在承载本轮三副本推理服务；GPU1/GPU2 有其他运行负载；GPU3 空闲但有显存占用。

## 阶段 4：服务端最大并发与满载证据

当前服务端 probe summary：

```text
/tmp/rwkv-skills-infer-swap/probe_19082_gpu0_lightweight_full_20260607.json
```

关键结果：

| 指标 | 值 |
| --- | --- |
| endpoint | `http://127.0.0.1:19082` |
| protocol | `nano-vllm-contents` |
| max tokens | `256` |
| 最大健康测试并发 | `2048` |
| 失败并发 | 无 |
| 吞吐峰值并发 | `1536` |
| 吞吐峰值 | `16066.86 output chars/s` |
| 平均 GPU 利用率 >= 90% 的并发 | `1536`, `2048` |
| 峰值 GPU 利用率 >= 90% 的并发 | `64` 起到 `2048` |
| 最大平均 GPU 利用率 | `91.43%` at `1536` |
| 最大峰值 GPU 利用率 | `96%` at `1536` |

关键点：

| concurrency | avg GPU | peak GPU | rps | output chars/s |
| ---: | ---: | ---: | ---: | ---: |
| 192 | 89.62% | 92% | 20.39 | 10970.15 |
| 1024 | 86.26% | 94% | 25.74 | 13849.17 |
| 1536 | 91.43% | 96% | 29.87 | 16066.86 |
| 2048 | 91.41% | 94% | 29.24 | 15733.65 |

结论：

- 如果目标是吞吐峰值，使用 `1536`。
- 如果目标是吃满 GPU，使用 `1536` 或 `2048`。
- 如果目标是“最大满载并发”，使用 `2048`。
- 如果目标是“满载且吞吐更优”，使用 `1536`。

当前代码里的 `probe-infer` 会区分：

- `throughput_best_concurrency`
- `gpu_full_concurrency`
- `largest_successful_concurrency`
- `selected_concurrency`
- `suggested_infer_max_workers`
- `suggested_remote_batch_size`

重新跑服务端满载 probe。优先使用轻量入口，避免远端 checkout 不完整时 scheduler CLI 额外加载 benchmark registry 依赖：

```bash
rtk uv run python -m src.bin.probe_remote_infer \
  --infer-base-url http://127.0.0.1:19082 \
  --infer-model rwkv7-g1g-2.9b-20260526-ctx8192 \
  --infer-protocol nano-vllm-contents \
  --candidates 64,128,192,256,512,1024,1536,2048 \
  --max-tokens 256 \
  --gpu-index 0 \
  --target-gpu-utilization 90 \
  --output-json /tmp/rwkv-skills-infer-swap/probe_19082_gpu0_full_load.json
```

注意：如果通过本地 SSH 转发运行 `probe-infer`，本地进程无法读取远端 NVML，GPU 利用率字段会是 `null`。需要 GPU 利用率证据时，应在 8222 服务端运行 probe。

## 阶段 5：正式测评参数

最大满载参数：

```bash
--infer-base-url http://127.0.0.1:29082
--infer-model rwkv7-g1g-2.9b-20260526-ctx8192
--infer-protocol nano-vllm-contents
--infer-seed-policy omit-for-contents
--infer-max-workers 2048
--remote-batch-size 2048
```

满载吞吐峰值参数：

```bash
--infer-base-url http://127.0.0.1:29082
--infer-model rwkv7-g1g-2.9b-20260526-ctx8192
--infer-protocol nano-vllm-contents
--infer-seed-policy omit-for-contents
--infer-max-workers 1536
--remote-batch-size 1536
```

低风险对照参数：

```bash
--infer-base-url http://127.0.0.1:29082
--infer-model rwkv7-g1g-2.9b-20260526-ctx8192
--infer-protocol nano-vllm-contents
--infer-seed-policy omit-for-contents
--infer-max-workers 1024
--remote-batch-size 1024
```

建议正式测评至少保留两组对照：

- `2048`：最大满载并发。
- `1536`：满载吞吐峰值。

如果测评稳定性优先，把 `2048` 换成 `1536`；`1024` 可以作为低风险对照，但本次 probe 中平均 GPU 利用率未达到 90%。

以上正式测评参数假设在本地工作站执行 scheduler，并通过 SSH 隧道访问 8222 router：

```bash
rtk ssh -p 8222 -N \
  -L 29082:127.0.0.1:19082 \
  -o ExitOnForwardFailure=yes \
  -o BatchMode=yes \
  chase@47.115.88.183
```

如果直接在 8222 服务器本机执行 scheduler，把 `http://127.0.0.1:29082` 改成 `http://127.0.0.1:19082`。

### 本地 DB preflight

`queue` 和 `dispatch` 都会读取本地 scheduler DB 状态；它们不是纯 parser dry-run。当前本机 `.env` 里的 `PG_PORT=5433` 不响应，本轮已在本机 `127.0.0.1:5432` 创建 `rwkv-eval` 并应用 `scripts/schema.sql`。

正式测评命令应显式覆盖 DB target：

```bash
PG_HOST=127.0.0.1
PG_PORT=5432
PG_USER=postgres
PG_DBNAME=rwkv-eval
```

`PG_PASSWORD` 继续从本地 `.env` 读取，不要写进命令行。

`src.bin.run_infer_swap_eval` 会在 import scheduler 之前同时设置 `PG_*` 与 `RWKV_EVAL_SPACE_DB_*` 两组 DB 环境变量，避免旧环境变量覆盖正式 DB target。

先确认本地 DB 可用：

```bash
rtk pg_isready -h 127.0.0.1 -p 5432 -U postgres -d rwkv-eval
```

如果改回其他 `PG_*`，必须重新跑本节 preflight 到 `ok = true`；否则 `queue`/`dispatch` 可能在初始化连接池时报 `psycopg_pool.PoolTimeout` 或认证失败。

正式启动前再跑一次完整 preflight。它会同时检查 endpoint health、`/v1/models`、协议 smoke 和本地 scheduler DB 连接：

```bash
rtk env PG_HOST=127.0.0.1 PG_PORT=5432 PG_USER=postgres PG_DBNAME=rwkv-eval \
  uv run python -m src.bin.preflight_remote_eval \
  --infer-base-url http://127.0.0.1:29082 \
  --infer-model rwkv7-g1g-2.9b-20260526-ctx8192 \
  --protocols openai,nano-vllm-contents \
  --batch-size 2 \
  --max-tokens 16 \
  --db-host 127.0.0.1 \
  --db-port 5432 \
  --db-user postgres \
  --db-name rwkv-eval \
  --db-timeout-s 5 \
  --stdout summary \
  --output-json /tmp/rwkv-skills-infer-swap-preflight-29082-localdb.json
```

只有输出里的 `ok = true` 时，才进入 `dispatch`。如果只想验证 endpoint 而暂时不检查 DB，可临时加 `--skip-db`；正式测评前不要跳过 DB。

当前本地正式路径 preflight 证据：

- 证据文件：`/tmp/rwkv-skills-infer-swap-preflight-29082-localdb-ok-20260607.json`
- endpoint 经本地 `29082 -> 8222:19082` 转发可用
- `infer_health`、`infer_models`、`protocol_smoke` 均为 `ok`
- `openai` 与 `nano-vllm-contents` batched smoke 均为 `2/2 ok`
- `scheduler_db = ok`：`127.0.0.1:5432/rwkv-eval user=postgres`

当前正式队列预览证据：

- 命令：`queue`，同下方最大满载并发参数，仅把 `dispatch` 改成 `queue`
- 结果：`待调度任务：9`
- 覆盖：`bfcl_v3_test`、`mcp_bench_test`、`tau2_bench_{airline,retail,telecom}_base`、`tau3_bench_{airline,banking_knowledge,retail,telecom}_base`

当前一键 helper 证据：

- helper：`src.bin.run_infer_swap_eval`
- 默认行为：完整 preflight 后执行 `queue`，不会启动正式测评
- 证据文件：`/tmp/rwkv-skills-infer-swap-preflight-helper-queue-audit-20260607.json`
- 结果：`ok = true`，随后 `待调度任务：9`

当前 8222 endpoint-only preflight 证据：

- 证据文件：`/tmp/rwkv-skills-infer-swap/preflight_19082_skip_db_summary_20260607.json`
- `ok = true`
- `infer_health`、`infer_models`、`protocol_smoke` 均为 `ok`
- `openai` 与 `nano-vllm-contents` batched smoke 均为 `2/2 ok`
- `scheduler_db = skipped`，因此该证据只证明 endpoint 就绪，不证明正式调度 DB 就绪

### 正式测评启动模板

推荐入口是 helper。先开 `29082 -> 8222:19082` 隧道，然后运行安全预览：

```bash
rtk uv run python -m src.bin.run_infer_swap_eval \
  --preflight-output-json /tmp/rwkv-skills-infer-swap-preflight-helper-queue.json \
  --print-scheduler-args
```

推荐用一键 launch bundle 串联 preflight、queue 预览、DB summary、GPU probe evidence 和 phase gate。它不会启动正式测评：

```bash
rtk uv run python -m src.bin.prepare_infer_swap_launch_bundle \
  --probe-json /tmp/rwkv-skills-infer-swap-probe-19082-gpu0-lightweight-full-20260607.json \
  --readiness-output-json /tmp/rwkv-skills-infer-swap-readiness-audit.json \
  --summary-output-json /tmp/rwkv-skills-infer-swap-eval-summary.json \
  --evidence-output-md /tmp/rwkv-skills-infer-swap-eval-evidence.md \
  --phase-gate-output-json /tmp/rwkv-skills-infer-swap-phase-gate.json \
  --bundle-output-json /tmp/rwkv-skills-infer-swap-launch-bundle.json
```

launch bundle 会校验 probe JSON 的 model、protocol、`largest_successful_concurrency` 和 `gpu_full_concurrency` 是否覆盖当前 profile 的 workers/batch，也会解析 scheduler 输出并默认要求 `queue_pending_count == 9`，并在 bundle JSON 中记录本次 launch identity。bundle 还会写入 `generated_at_utc`、`tunnel_argv` / `tunnel_command`、`queue_argv` / `queue_command`、`dispatch_argv` / `dispatch_command`，用于把 SSH 隧道、正式启动命令和证据包绑定；同时写入 `summary_argv` / `summary_command`、`summary_watch_argv` / `summary_watch_command`、`evidence_argv` / `evidence_command`、`speedup_doc_argv` / `speedup_doc_command` 和 `speedup_md`，用于正式测评后的只读汇总、事实报告和提速文档生成。正式 `dispatch` 还会读取 phase gate JSON 和 launch bundle JSON，只有 readiness 输出 `ready_to_dispatch=true`，phase gate schema v2 `ok=true`，包含 `git_diff_check/readiness_json/probe_json/summary_json`，phase gate 的 `protocol_smoke_protocols` 证明 `openai` 与 `nano-vllm-contents` 都有至少 `2/2` 非空输出，phase gate 的 `gpu_full_concurrency` / `largest_successful_concurrency` 覆盖本次 dispatch 的 workers/batch，且 bundle 里的 `generated_at_utc`、base URL、model、timeout、workers、remote batch、jobs、datasets、DB target、run mode、summary watch interval、speedup output path、`tunnel_argv` / `tunnel_command`、`queue_argv` / `queue_command`、`dispatch_argv` / `dispatch_command` 都与本次 dispatch 参数一致时，才显式启动正式测评。helper 会先检查 `--confirm-dispatch`、phase gate 和 launch bundle，再跑 endpoint/DB preflight，最后才进入 scheduler `dispatch`：

```bash
rtk uv run python -m src.bin.run_infer_swap_eval \
  --action dispatch \
  --confirm-dispatch \
  --profile full-load \
  --phase-gate-json /tmp/rwkv-skills-infer-swap-phase-gate.json \
  --launch-bundle-json /tmp/rwkv-skills-infer-swap-launch-bundle.json \
  --preflight-output-json /tmp/rwkv-skills-infer-swap-preflight-helper-dispatch.json
```

如果要跑吞吐峰值对照，把 `--profile full-load` 改为 `--profile throughput-peak`，并先用同样 profile 重新跑 launch bundle。helper 要求同时传 `--action dispatch --confirm-dispatch`，并通过 `--phase-gate-json` 与 `--launch-bundle-json` 参数一致性校验，才会启动正式测评。

刷新 bundle 后，也可以直接读取里面固化的隧道命令和正式启动命令：

```bash
rtk jq -r .tunnel_command /tmp/rwkv-skills-infer-swap-launch-bundle.json
rtk jq -r .dispatch_command /tmp/rwkv-skills-infer-swap-launch-bundle.json
rtk jq -r .summary_watch_command /tmp/rwkv-skills-infer-swap-launch-bundle.json
rtk jq -r .evidence_command /tmp/rwkv-skills-infer-swap-launch-bundle.json
rtk jq -r .speedup_doc_command /tmp/rwkv-skills-infer-swap-launch-bundle.json
```

### 正式测评结果汇总

正式 `dispatch` 启动后，用只读 DB helper 检查 9 个数据集是否已经产生真实 `scores`：

```bash
rtk uv run python -m src.bin.summarize_infer_swap_eval \
  --output-json /tmp/rwkv-skills-infer-swap-eval-summary.json
```

默认 DB target 与正式 helper 一致：

```text
127.0.0.1:5432/rwkv-eval user=postgres
```

输出含义：

- `score=none`：该 dataset 的最新正式 task 还没有 `scores` 行，不显示任何分数。
- `metrics={...}`：只来自 DB `scores.metrics` 的真实 JSONB。
- `tasks=x/9`：已有正式 task 的 dataset 数。
- `scored=x/9`：已有真实 score 的 dataset 数。
- 进度查询时返回码 `1` 只表示尚未全部 scored；全部 9 个 dataset 都有 score 后返回码为 `0`。

需要 JSON stdout 时：

```bash
rtk uv run python -m src.bin.summarize_infer_swap_eval --stdout json
```

需要持续观察正式测评进度时：

```bash
rtk uv run python -m src.bin.summarize_infer_swap_eval \
  --watch \
  --watch-interval-s 60 \
  --output-json /tmp/rwkv-skills-infer-swap-eval-summary.json
```

`--watch` 每轮都会重新只读查询 DB，并覆盖 `--output-json` 为最新状态。它在 9 个 dataset 都有真实 `scores` 后返回 `0`；如果加了 `--watch-timeout-s` 且超时前仍未全部 scored，则返回 `1`。

需要生成后续提速设计可引用的事实报告时，把服务端 GPU 满载 probe 一起带入 Markdown：

```bash
rtk uv run python -m src.bin.summarize_infer_swap_eval \
  --probe-json /tmp/rwkv-skills-infer-swap-probe-19082-gpu0-lightweight-full-20260607.json \
  --output-json /tmp/rwkv-skills-infer-swap-eval-summary.json \
  --output-md /tmp/rwkv-skills-infer-swap-eval-evidence.md
```

如果 probe JSON 只在 8222 上，先拉到本机：

```bash
rtk scp -P 8222 \
  chase@47.115.88.183:/tmp/rwkv-skills-infer-swap/probe_19082_gpu0_lightweight_full_20260607.json \
  /tmp/rwkv-skills-infer-swap-probe-19082-gpu0-lightweight-full-20260607.json
```

该 Markdown 报告仍然只陈列事实：并发/GPU probe、DB task/completion/eval/score、真实 `scores.metrics`。正式分数未写入前，它不会给出提速设计结论。

全部 9 个 dataset 都有真实 `scores` 后，才能生成提速设计文档：

```bash
rtk uv run python -m src.bin.draft_infer_swap_speedup_doc \
  --summary-json /tmp/rwkv-skills-infer-swap-eval-summary.json \
  --probe-json /tmp/rwkv-skills-infer-swap-probe-19082-gpu0-lightweight-full-20260607.json \
  --readiness-json /tmp/rwkv-skills-infer-swap-readiness-audit.json \
  --launch-bundle-json /tmp/rwkv-skills-infer-swap-launch-bundle.json \
  --output-md /tmp/rwkv-skills-infer-swap-speedup-design.md
```

默认情况下，如果 `summary.all_scored=false`、summary 中任一 dataset 缺 latest task / `Completed` 状态 / `score_id` / 真实 `metrics`、显式传入的 launch bundle JSON 不存在、readiness JSON 不存在，或者 launch bundle/readiness 与 summary/probe 的 model、DB、dataset list、queue count、GPU concurrency、双协议 smoke 不一致，该命令返回 `1` 且不写文档，避免把未完成或身份不一致的测评写成提速结论。
生成成功时，文档会把 launch bundle 的生成时间、`tunnel_command`、`dispatch_command`、`summary_command`、`summary_watch_command`、`evidence_command` 和 `speedup_doc_command` 写入 Evidence Gate，后续提速分析可追溯到完整运行链路。

下面保留等价的 scheduler 原生命令，方便排查或手动拆分执行。

最大满载并发：

```bash
rtk env PG_HOST=127.0.0.1 PG_PORT=5432 PG_USER=postgres PG_DBNAME=rwkv-eval \
  uv run python -m src.eval.scheduler.cli dispatch \
  --infer-base-url http://127.0.0.1:29082 \
  --infer-models rwkv7-g1g-2.9b-20260526-ctx8192 \
  --infer-protocol nano-vllm-contents \
  --infer-seed-policy omit-for-contents \
  --infer-max-workers 2048 \
  --remote-batch-size 2048 \
  --max-concurrent-jobs 1 \
  --only-jobs \
    function_tau2_bench \
    function_tau3_bench \
    function_mcp_bench \
    function_bfcl_v3 \
  --only-datasets \
    tau2_bench_airline_base \
    tau2_bench_retail_base \
    tau2_bench_telecom_base \
    tau3_bench_airline_base \
    tau3_bench_banking_knowledge_base \
    tau3_bench_retail_base \
    tau3_bench_telecom_base \
    mcp_bench_test \
    bfcl_v3_test \
  --run-mode new
```

满载吞吐峰值对照：

```bash
rtk env PG_HOST=127.0.0.1 PG_PORT=5432 PG_USER=postgres PG_DBNAME=rwkv-eval \
  uv run python -m src.eval.scheduler.cli dispatch \
  --infer-base-url http://127.0.0.1:29082 \
  --infer-models rwkv7-g1g-2.9b-20260526-ctx8192 \
  --infer-protocol nano-vllm-contents \
  --infer-seed-policy omit-for-contents \
  --infer-max-workers 1536 \
  --remote-batch-size 1536 \
  --max-concurrent-jobs 1 \
  --only-jobs \
    function_tau2_bench \
    function_tau3_bench \
    function_mcp_bench \
    function_bfcl_v3 \
  --only-datasets \
    tau2_bench_airline_base \
    tau2_bench_retail_base \
    tau2_bench_telecom_base \
    tau3_bench_airline_base \
    tau3_bench_banking_knowledge_base \
    tau3_bench_retail_base \
    tau3_bench_telecom_base \
    mcp_bench_test \
    bfcl_v3_test \
  --run-mode new
```

启动前可以把 `dispatch` 改成 `queue` 预览待调度任务；前提仍然是本地 DB 可连接。

## 阶段 6：fleet 一键部署形态

fleet 支持同模型多副本和可选 router：

```bash
rtk uv run python -m src.bin.run_infer_fleet \
  --model-paths /path/to/rwkv7-g1g-2.9b.pth \
  --model-names rwkv7-g1g-2.9b-20260526-ctx8192 \
  --base-port 18082 \
  --replicas-per-model 3 \
  --max-batch-size 64 \
  --router-port 19082 \
  --manifest-path logs/infer/fleet.json \
  --detach
```

manifest 会包含：

- `services`
- `routes_by_model`
- `router_routes`
- `router`

`router_routes` 可直接作为手动启动 `run_infer_router` 的 `--route` 参数来源。

## 当前完成边界

已完成：

- 可替换协议路径。
- 单元测试覆盖。
- 真实 endpoint 双协议 smoke compare。
- router split batch 验证。
- 服务端 GPU0 并发 probe，并证明 `1536/2048` 可以把平均 GPU 利用率推到 90% 以上。
- 测评参数已固定。

未完成：

- 正式 benchmark 分数验证。该项需要用户运行测评后提供结果或允许继续调度正式评测。
- 替换后的提速设计文档。该文档应基于正式测评瓶颈和实际 score/throughput，再给出下一轮优化方案。
