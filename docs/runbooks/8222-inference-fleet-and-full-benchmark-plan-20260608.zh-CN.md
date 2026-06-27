# 8222 推理端压测与全 benchmark 测评计划

更新时间：2026-06-08 20:40 CST

## 目标

当前任务分两条线推进：

1. 使用相同端口反复跑正式测评，找到可以把单卡性能吃满的最佳配置。
   - 不再把 CMMLU 256 样本探测当成正式结论。
   - 每次只改变一组推理端/评测端参数，用正式 benchmark 结果、`/v1/batch-metrics`、服务日志和 `nvidia-smi` 共同判断。
   - 目标是显存尽量接近满载、GPU 利用率明显抬升、有效 batch 变大、吞吐提升，同时分数不回退、无 HTTP 5xx/OOM。

2. 启动完整 benchmark，用本地数据库完成所有 benchmark 测评。
   - 评测调度在本地 `/home/chase/GitHub/rwkv-skills` 跑，使用本地 `.env` 的 DB/Judge 配置。
   - 推理只通过 SSH 转发打 8222 上的推理端。
   - 所有结果落本地 DB，远端只承载 GPU 推理服务。

## 当前代码和路径

- 本地代码源：`/home/chase/GitHub/rwkv-skills`
- 8222 目标代码路径：`/home/chase/chase-rwkv-skills`
- 8222 旧/旁路路径仍存在：`/home/chase/chase/rwkv-skills`
- 8222 权重路径：`/home/chase/rwkv-skills/weights/BlinkDL__rwkv7-g1`
- nano-vLLM RWKV runtime：`/tmp/nano-vllm-rwkv-315cf53`
- 本轮远端日志目录：`/home/chase/chase-rwkv-skills/logs/infer/fleet_20260608`

已同步到 8222 的关键代码面：

- `src/bin/run_infer_server.py`
- `src/bin/run_infer_router.py`
- `src/infer/server.py`
- `src/infer/backend.py`
- `src/eval/knowledge/pipeline.py`

同步后已在 8222 用 `.venv/bin/python -m compileall` 验证过推理服务、router 和 knowledge 相关代码可编译。

## 当前运行状态

### 8222 GPU 状态

当前快照：

| GPU | 显存 | 利用率 | 归属 |
| --- | --- | --- | --- |
| GPU0 | 14919 / 97887 MiB | 0% | `rwkv` 用户裸 `nanovllm.entrypoints.openai.api_server`，PID `4113391` |
| GPU1 | 11677 / 97887 MiB | 0% | `chase` 项目推理服务，g1g 2.9B |
| GPU2 | 11677 / 97887 MiB | 0% | `chase` 项目推理服务，g1f 2.9B |
| GPU3 | 22961 / 97887 MiB | 0% | `chase` 项目推理服务，g1g 7.2B |

GPU0 当前不是本轮项目服务，且没有 `/healthz`、`/v1/batch-metrics`。因此四模型目标目前缺 g1f 7.2B，原因是 GPU0 被其他用户进程占用。

### 8222 项目推理服务

当前已起三路项目推理服务：

| 模型 | GPU | 端口 | PID | health |
| --- | --- | --- | --- | --- |
| `rwkv7-g1g-2.9b-20260526-ctx8192` | GPU1 | `18082` | `296437` | OK |
| `rwkv7-g1f-2.9b-20260420-ctx8192` | GPU2 | `18083` | `296457` | OK |
| `rwkv7-g1g-7.2b-20260523-ctx8192` | GPU3 | `18084` | `296451` | OK |

共同参数：

```bash
--max-batch-size 512
--batch-collect-ms 50
--max-num-seqs 512
--max-num-batched-tokens 32768
--rwkv-prefill-token-budget 8192
--rwkv-prefill-max-batch-size 256
--max-state-slots 8192
--gpu-memory-utilization 0.96
```

健康检查：

```bash
curl -sS http://127.0.0.1:18082/healthz
curl -sS http://127.0.0.1:18083/healthz
curl -sS http://127.0.0.1:18084/healthz
```

三路均返回 `max_batch_size=512`、`batch_collect_ms=50`。

### 8222 router

远端 router：

- 端口：`127.0.0.1:19083`
- PID：`299530`
- `--forward-max-workers 1024`
- 当前路由：g1g 2.9B、g1f 2.9B、g1g 7.2B

健康检查：

```bash
ssh -p 8222 chase@47.115.88.183 \
  'curl -sS http://127.0.0.1:19083/healthz'
```

注意：本机 `127.0.0.1:19083` 当前不是这个远端 router 的可信视图。本机还有旧的本地 `run_infer_router` PID `13304` 占用 `19083`，只返回 g1g/g1f 两个 2.9B 路由。重新开 autossh 前应先处理这个本地端口冲突。

### 已关闭的转发

已按要求关闭刚才用于 `19083 -> 8222:19083` 的 autossh 进程：

- 已关闭：本机 autossh PID `22438`

仍存在但未动：

- 本机 autossh PID `12450`：`19082 -> 8222:8008`

该旧转发不是本轮 `19083` router 转发。

## 两种推理端方式

当前仍能看到两类服务方式：

1. 项目自带 `src.bin.run_infer_server`
   - 提供 `/healthz`
   - 提供 `/v1/chat/completions`
   - 提供 `/v1/completions`
   - 提供 `/v1/batch-metrics`
   - 写 `RWKV_INFER_BATCH_METRICS` 日志
   - 支持本项目 full benchmark 需要的 OpenAI chat、raw completions、batch metrics 和 knowledge fallback 路径

2. 裸 `nanovllm.entrypoints.openai.api_server`
   - 当前 GPU0 上有一个 `rwkv` 用户进程在跑
   - 不提供本项目的 `/healthz` 和 `/v1/batch-metrics`
   - 不适合作为所有 benchmark 的统一正式入口

结论：正式 full benchmark 和调参压测应统一走项目自带 `run_infer_server + run_infer_router`。

## 工作线 1：同端口正式测评压榨配置

固定远端统一入口：

- 远端：`http://127.0.0.1:19083/v1`
- 本地转发后：`http://127.0.0.1:19083/v1`

重新开转发前先处理本机端口冲突：

```bash
kill 13304
```

然后使用：

```bash
AUTOSSH_GATETIME=0 autossh -M 0 -N \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -p 8222 \
  -L 19083:127.0.0.1:19083 \
  chase@47.115.88.183
```

转发验证必须同时满足：

```bash
curl -sS http://127.0.0.1:19083/healthz
curl -sS http://127.0.0.1:19083/v1/models
```

本地返回的模型列表必须和远端 `8222:19083` 一致。当前远端应至少包含：

- `rwkv7-g1g-2.9b-20260526-ctx8192`
- `rwkv7-g1f-2.9b-20260420-ctx8192`
- `rwkv7-g1g-7.2b-20260523-ctx8192`

当 GPU0 释放并补起 g1f 7.2B 后，还应包含：

- `rwkv7-g1f-7.2b-20260414-ctx8192`

### 调参变量

推理端变量：

- `--max-batch-size`
- `--batch-collect-ms`
- `--max-num-seqs`
- `--max-num-batched-tokens`
- `--rwkv-prefill-token-budget`
- `--rwkv-prefill-max-batch-size`
- `--max-state-slots`
- `--gpu-memory-utilization`

router 变量：

- `--forward-max-workers`

评测端变量：

- `--infer-max-workers`
- `--remote-batch-size`
- `--max-concurrent-jobs`
- benchmark 自身的 `--batch-size`

### 判断指标

每次正式测评后记录：

- task id
- model
- dataset / benchmark
- score
- completions count
- 实际 batch 分布
- `/v1/batch-metrics`
- 服务日志中的 `RWKV_INFER_BATCH_METRICS`
- `nvidia-smi` 最大显存
- GPU utilization
- output tok/s
- HTTP 5xx / timeout / OOM

配置有效的标准：

- score 不低于同样 benchmark 的基线
- output tok/s 提升
- 实际 batch 明显变大
- VRAM 接近上限但不 OOM，目标区间约 95GB-97GB
- GPU utilization 明显抬升
- 不出现持续 HTTP 500、连接重置或 router starvation

### 下一步调参顺序

1. 先固定三路现有服务，使用远端 `19083` 或正确本地转发 `19083` 做小规模正式 benchmark smoke。
2. 在同一个 benchmark 上扫评测端并发：`infer-max-workers=64,128,256,384,512`。
3. 固定最佳评测并发后，扫 `remote-batch-size` / benchmark `batch-size`。
4. 观察实际 batch 和显存，如果显存仍低，逐步增加 `max-state-slots`、`max-num-batched-tokens`、`gpu-memory-utilization`。
5. 每次只改一组变量，保留 task id、日志路径和 batch metrics。
6. GPU0 释放后补起 g1f 7.2B，再把 router 更新为四模型路由。

## 工作线 2：本地 DB 完整 benchmark

执行原则：

- 调度和 DB 都在本地 `/home/chase/GitHub/rwkv-skills`。
- 使用本地 `.env`，不要改 DB/Judge。
- 只把推理请求通过 autossh 打到 8222。
- 使用公开协议面：优先 `--infer-protocol vllm`；需要 raw completions 的路径使用当前代码里的 `completions` 支持。
- 不把 readiness/probe artifact 当正式分数；正式分数只看本地 DB 的 task/completions/eval/scores。

建议基础命令形态：

```bash
cd /home/chase/GitHub/rwkv-skills
set -a
source .env
set +a

.venv/bin/python -m src.eval.scheduler.cli dispatch \
  --run-mode rerun \
  --infer-base-url http://127.0.0.1:19083/v1 \
  --infer-models \
    rwkv7-g1g-2.9b-20260526-ctx8192 \
    rwkv7-g1f-2.9b-20260420-ctx8192 \
    rwkv7-g1g-7.2b-20260523-ctx8192 \
  --infer-protocol vllm \
  --infer-timeout-s 1200 \
  --infer-max-workers 256 \
  --remote-batch-size 256 \
  --max-concurrent-jobs 1 \
  --run-log-dir logs/dispatch/full_benchmark_8222_20260608
```

补齐四模型后加入：

```bash
rwkv7-g1f-7.2b-20260414-ctx8192
```

是否使用 `--disable-checker` 取决于本轮正式目标：

- 如果目标是只测基础 benchmark 分数且不跑 wrong-answer checker，可加 `--disable-checker`。
- 如果目标是完整复刻当前仓库默认正式流程，不加，让本地 `.env` 中的 judge 配置生效。

## 必须先处理的问题

1. 本机 `19083` 端口被旧本地 router PID `13304` 占用。
   - 现象：本地 `curl http://127.0.0.1:19083/healthz` 只返回两个 2.9B 模型。
   - 远端 `8222:19083` 已返回三模型。
   - 处理：关闭 PID `13304` 后再启动 autossh `-L 19083:127.0.0.1:19083`。

2. GPU0 被 `rwkv` 用户裸 nano-vLLM 服务占用。
   - 当前 PID：`4113391`
   - 当前模型：裸 `rwkv7-g1g`
   - 影响：无法部署第四个目标模型 `g1f-7.2B`。
   - 处理：等 GPU0 释放，或确认可以停止该用户进程后，再用项目 `run_infer_server` 拉起 g1f 7.2B。

3. 当前 7.2B 只起了 g1g。
   - 已起：`rwkv7-g1g-7.2b-20260523-ctx8192` on GPU3。
   - 未起：`rwkv7-g1f-7.2b-20260414-ctx8192`。

4. 远端启动命令曾因 shell 后台优先级留下 bash 父进程。
   - 服务本身 OK，但后续建议用显式脚本或 `run_infer_fleet.py --detach`，避免 ssh 会话和 bash 父进程残留。

## 下一步

短期：

1. 关闭本机旧 router PID `13304`，重新用 autossh 占用本机 `19083`。
2. 验证本地 `19083/healthz` 与远端 `19083/healthz` 返回一致。
3. 用三模型先跑正式 benchmark smoke，记录 task id、score、batch metrics 和 GPU 状态。
4. 根据正式结果扫 `infer-max-workers` 与 `remote-batch-size`。

中期：

1. GPU0 释放后，用项目服务拉起 `rwkv7-g1f-7.2b-20260414-ctx8192`。
2. 重启远端 router，把四模型都挂到同一个 `19083`。
3. 固定最佳配置，启动所有 benchmark 的本地 DB 正式测评。
4. 持续用本地 DB 表和远端 batch metrics 对齐进度，不用 probe artifact 代替正式分数。

