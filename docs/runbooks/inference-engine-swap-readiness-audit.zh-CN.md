# 推理引擎替换 Readiness Audit

日期：2026-06-07

本文是 `docs/runbooks/inference-engine-swap-validation.zh-CN.md` 的完成边界审计，只记录当前可由代码、测试、远端服务和本地命令直接证明的状态。

## 目标拆分

| 要求 | 当前状态 | 证据 |
| --- | --- | --- |
| 推理后端可替换到 nano contents 协议 | 已完成 | `src.infer.backend.RemoteInferenceBackend` 支持 `--infer-protocol nano-vllm-contents` 与 `--infer-seed-policy omit-for-contents` |
| server 接受 OpenAI messages 与 nano contents | 已完成 | `src.infer.server` 支持 `/v1/chat/completions` 的 `messages` 与 `contents` 请求 |
| router 对同模型多副本拆分 contents batch | 已完成 | `src.bin.run_infer_router` 对 duplicate same-model backend 拆分并按原 index 合并 |
| fleet 能部署多副本和 router | 已完成 | `src.bin.run_infer_fleet` 支持 `--replicas-per-model` 与 `--router-port` |
| 每阶段单元测试 | 已完成到当前替换边界 | focused gate: `125 passed, 2 warnings`；`src.bin.prepare_infer_swap_launch_bundle` 可重复刷新非 dispatch 启动证据包 |
| 真实 endpoint 双协议对比 | 已完成 | 8222 endpoint-only preflight: `openai:ok,nano-vllm-contents:ok` |
| 服务端最大并发并吃满 GPU | 已完成 | `/tmp/rwkv-skills-infer-swap/probe_19082_gpu0_lightweight_full_20260607.json`，`gpu_full_concurrency=2048`，`throughput_best_concurrency=1536` |
| 正式本地 scheduler 预启动 gate | 已完成 | `/tmp/rwkv-skills-infer-swap-preflight-helper-queue-audit-20260607.json`，`ok=true`，`scheduler_db=ok` |
| 正式结果只读汇总入口 | 已完成 | `src.bin.summarize_infer_swap_eval` 从 DB `task`/`completions`/`eval`/`scores` 汇总，不为缺失 score 补分 |
| 正式启动前一键审计入口 | 已完成 | `src.bin.audit_infer_swap_readiness` 只执行 preflight、queue、summary、probe evidence 检查，不 dispatch；默认硬性要求 queue 预览解析到 `待调度任务：9` |
| 正式 benchmark 分数验证 | 待用户运行 | 需要实际 `dispatch` 后的 DB `scores` / artifacts |
| 基于替换后的提速设计文档 | 待 benchmark 结果 | 需要真实 score/throughput/瓶颈证据，不能提前写成结论 |

## 当前推荐启动方式

1. 开 SSH 隧道：

```bash
rtk ssh -p 8222 -N \
  -L 29082:127.0.0.1:19082 \
  -o ExitOnForwardFailure=yes \
  -o BatchMode=yes \
  chase@47.115.88.183
```

2. 在另一个终端先跑安全预览：

```bash
rtk uv run python -m src.bin.run_infer_swap_eval \
  --preflight-output-json /tmp/rwkv-skills-infer-swap-preflight-helper-queue.json \
  --print-scheduler-args
```

3. 使用一键 launch bundle 刷新默认 readiness、summary、evidence 和 phase gate。它不会启动正式测评：

```bash
rtk uv run python -m src.bin.prepare_infer_swap_launch_bundle \
  --probe-json /tmp/rwkv-skills-infer-swap-probe-19082-gpu0-lightweight-full-20260607.json \
  --readiness-output-json /tmp/rwkv-skills-infer-swap-readiness-audit.json \
  --summary-output-json /tmp/rwkv-skills-infer-swap-eval-summary.json \
  --evidence-output-md /tmp/rwkv-skills-infer-swap-eval-evidence.md \
  --phase-gate-output-json /tmp/rwkv-skills-infer-swap-phase-gate.json \
  --bundle-output-json /tmp/rwkv-skills-infer-swap-launch-bundle.json
```

launch bundle 内部会先跑 readiness audit：校验 probe JSON 的 model、protocol、`largest_successful_concurrency` 和 `gpu_full_concurrency` 是否覆盖当前 profile 的 workers/batch；同时解析 scheduler 输出并要求 `queue_pending_count == expected_queue_count`，默认就是 9。输出的 bundle JSON 会记录本次非密钥 launch identity：profile、base URL、model、workers、remote batch、jobs、datasets 和 DB target，也会写入 `generated_at_utc`、`tunnel_argv` / `tunnel_command`、`queue_argv` / `queue_command`、`dispatch_argv` / `dispatch_command`、`summary_argv` / `summary_command`、`summary_watch_argv` / `summary_watch_command`、`evidence_argv` / `evidence_command`、`speedup_doc_argv` / `speedup_doc_command`。正式启动和后续收分/写提速文档都可以直接从 bundle 读取命令，避免手工拼命令造成参数漂移。

4. 如果只需要重新校验已有 evidence JSON，也可以单独跑非 dispatch 分阶段 gate：

```bash
rtk uv run python -m src.bin.validate_infer_swap_phases \
  --readiness-json /tmp/rwkv-skills-infer-swap-readiness-audit.json \
  --probe-json /tmp/rwkv-skills-infer-swap-probe-19082-gpu0-lightweight-full-20260607.json \
  --summary-json /tmp/rwkv-skills-infer-swap-eval-summary.json \
  --output-json /tmp/rwkv-skills-infer-swap-phase-gate.json
```

该命令不会启动正式测评。它只证明当前替换路径的分阶段单测、compile、diff hygiene 和已有 evidence JSON 仍然一致。生成的 JSON 是 phase gate schema v2，包含生成时间和 required phase 列表。

5. 只有当输出仍然是 `ok=true` / `ready_to_dispatch=true`，并且 phase gate JSON 是 schema v2、`ok=true`、包含 `git_diff_check/readiness_json/probe_json/summary_json`，同时 launch bundle 的 `generated_at_utc`、base URL、model、timeout、workers、remote batch、jobs、datasets、DB target、run mode、`tunnel_argv` / `tunnel_command`、`queue_argv` / `queue_command`、`dispatch_argv` / `dispatch_command` 都与本次 dispatch 参数一致时，才启动正式测评。`run_infer_swap_eval` 会先检查 `--confirm-dispatch`、phase gate 和 launch bundle，再跑 endpoint/DB preflight，最后才进入 scheduler `dispatch`。默认配置下这已经包含 `待调度任务：9` 的硬校验：

```bash
rtk uv run python -m src.bin.run_infer_swap_eval \
  --action dispatch \
  --confirm-dispatch \
  --profile full-load \
  --phase-gate-json /tmp/rwkv-skills-infer-swap-phase-gate.json \
  --launch-bundle-json /tmp/rwkv-skills-infer-swap-launch-bundle.json \
  --preflight-output-json /tmp/rwkv-skills-infer-swap-preflight-helper-dispatch.json
```

等价地，可以读取 launch bundle 中固化的隧道命令和启动命令：

```bash
rtk jq -r .tunnel_command /tmp/rwkv-skills-infer-swap-launch-bundle.json
rtk jq -r .dispatch_command /tmp/rwkv-skills-infer-swap-launch-bundle.json
rtk jq -r .summary_watch_command /tmp/rwkv-skills-infer-swap-launch-bundle.json
rtk jq -r .evidence_command /tmp/rwkv-skills-infer-swap-launch-bundle.json
rtk jq -r .speedup_doc_command /tmp/rwkv-skills-infer-swap-launch-bundle.json
```

吞吐峰值对照使用：

```bash
rtk uv run python -m src.bin.run_infer_swap_eval \
  --action dispatch \
  --confirm-dispatch \
  --profile throughput-peak \
  --phase-gate-json /tmp/rwkv-skills-infer-swap-phase-gate.json \
  --launch-bundle-json /tmp/rwkv-skills-infer-swap-launch-bundle.json \
  --preflight-output-json /tmp/rwkv-skills-infer-swap-preflight-helper-dispatch-throughput.json
```

5. dispatch 后用只读 DB helper 汇总真实结果：

```bash
rtk uv run python -m src.bin.summarize_infer_swap_eval \
  --output-json /tmp/rwkv-skills-infer-swap-eval-summary.json
```

该 helper 返回 `1` 表示 9 个 dataset 尚未全部写入 `scores`；全部有真实 score 后返回 `0`。输出中的 `score=none` 不是分数。

需要持续观察时使用：

```bash
rtk uv run python -m src.bin.summarize_infer_swap_eval \
  --watch \
  --watch-interval-s 60 \
  --output-json /tmp/rwkv-skills-infer-swap-eval-summary.json
```

`--watch` 仍然只读 DB；它每轮覆盖 JSON 为最新状态，不会启动、停止或修改测评任务。

需要生成提速设计前的事实证据报告时：

```bash
rtk uv run python -m src.bin.summarize_infer_swap_eval \
  --probe-json /tmp/rwkv-skills-infer-swap-probe-19082-gpu0-lightweight-full-20260607.json \
  --output-json /tmp/rwkv-skills-infer-swap-eval-summary.json \
  --output-md /tmp/rwkv-skills-infer-swap-eval-evidence.md
```

该 Markdown 报告只汇总 GPU probe 与 DB 真实结果；正式分数未完成前不产生提速结论。

全部 9 个 dataset 都有真实 `scores` 后，再生成提速设计文档：

```bash
rtk uv run python -m src.bin.draft_infer_swap_speedup_doc \
  --summary-json /tmp/rwkv-skills-infer-swap-eval-summary.json \
  --probe-json /tmp/rwkv-skills-infer-swap-probe-19082-gpu0-lightweight-full-20260607.json \
  --readiness-json /tmp/rwkv-skills-infer-swap-readiness-audit.json \
  --launch-bundle-json /tmp/rwkv-skills-infer-swap-launch-bundle.json \
  --output-md /tmp/rwkv-skills-infer-swap-speedup-design.md
```

如果 `summary.all_scored=false`、显式传入的 launch bundle JSON 不存在，或者 launch bundle 与 summary/probe 的 model、DB、dataset list 不一致，该命令默认返回 `1` 且不写文档。

## 当前已验证的运行参数

| profile | infer workers | remote batch size | 用途 |
| --- | ---: | ---: | --- |
| `full-load` | 2048 | 2048 | 最大满载并发 |
| `throughput-peak` | 1536 | 1536 | 满载吞吐峰值 |
| `low-risk` | 1024 | 1024 | 低风险对照 |

## 风险边界

- 当前 helper 默认只执行 `preflight + queue`；不会启动正式测评。
- 一键 audit helper 只执行 `preflight + queue + summary + evidence`；不会启动正式测评。
- 默认 `--expected-queue-count 9`，如果 queue 输出缺失待调度数量或不是 9，`ready_to_dispatch=false`。
- 只有同时传 `--action dispatch --confirm-dispatch`，且 `--phase-gate-json` 指向成功的 schema v2 分阶段 gate 报告、`--launch-bundle-json` 指向成功的启动证据包，并且两者记录的启动身份与本次 dispatch 参数、bundle 内固化的 `tunnel_argv` / `tunnel_command`、`queue_argv` / `queue_command`、`dispatch_argv` / `dispatch_command` 一致时，才会启动正式 benchmark；确认和证据校验会先于 preflight 执行。
- helper 会在 import scheduler 之前同时设置 `PG_*` 与 `RWKV_EVAL_SPACE_DB_*`，避免旧 DB 环境变量覆盖本地正式 DB target。
- 结果汇总 helper 只读 DB；没有 `scores` 行时只显示 `score=none`，不显示任何替代分数。
- 提速设计生成器默认要求 `summary.all_scored=true`；正式分数未完成时不会写结论文档。
- 本机 `rwkv-eval` 是按 `scripts/schema.sql` 初始化的空库；正式跑完后，分数真相以 DB `scores`、`completions`、`eval` 为准。
- 不能在没有正式结果前声称替换后提速幅度或分数变化。
