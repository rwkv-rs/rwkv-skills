# 8222 当前已验证最高性能配置与启动方式

更新时间：2026-06-09 09:40 CST

本文记录当前已经用正式本地 DB 任务验证过的最高吞吐配置。结论只覆盖已经跑过的配置，不表示所有参数空间已经穷尽。

## 结论

当前最高已验证组合：

- 远端推理端：三路项目 `run_infer_server`，只使用 8222 的 GPU1/GPU2/GPU3。
- 远端 router：`127.0.0.1:19083`，`--forward-max-workers 4096`。
- 本地入口：用户固定的 autossh 转发 `127.0.0.1:19083 -> 8222:127.0.0.1:19083`，配置不要改。
- 本地 benchmark：本地 `/home/chase/GitHub/rwkv-skills` 调度，本地 `.env` DB/Judge，推理只打 `http://127.0.0.1:19083/v1`。
- 评测侧高吞吐参数：`--infer-max-workers 768`、`--remote-batch-size 768`、`--infer-protocol completions`、`--max-concurrent-jobs 1`、`--disable-checker`。

已验证代表任务：

- DB task `66`：`rwkv7-g1f-2.9b-20260420-ctx8192 / human_eval_test`
- 状态：`Completed`
- `completions=5248`，`eval=5248`，`scores=1`
- 分数：`avg@32=0.5687881097560976`
- 对比旧 task `38`：同模型同 benchmark 旧分数 `avg@32=0.5562118902439024`
- 服务端 batch metrics：`failed_batches=0`，最后显存 `94801 / 97250 MB`
- 服务端累计：`total_batches=32`，`total_requests=11515`，`avg_output_tok_s=410.6457`
- 最近大 batch 实测约 `batch_size=766/766/764/766/764/636`，output tok/s 约 `444.24/426.97/407.11/393.26/394.86/379.34`

## 端口和 GPU

不要使用 GPU0。GPU0 当前仍被其他用户裸 nano-vLLM 进程占用，且不是本项目统一入口。

| GPU | 模型 | 远端端口 | 本地用途 |
| --- | --- | --- | --- |
| GPU1 | `rwkv7-g1g-2.9b-20260526-ctx8192` | `18082` | 远端项目 infer server |
| GPU2 | `rwkv7-g1f-2.9b-20260420-ctx8192` | `18083` | 远端项目 infer server；当前最高吞吐验证任务使用它 |
| GPU3 | `rwkv7-g1g-7.2b-20260523-ctx8192` | `18084` | 远端项目 infer server |
| router | 三模型聚合 | `19083` | 本地 autossh 固定转发入口 |

本地还可能有旧转发端口，例如 `19082` 或单卡直连 `1808x`。正式 benchmark 统一只走本地 `127.0.0.1:19083`。

## 固定 autossh 转发

用户已在当前终端拉起这条转发。配置不能改：

```bash
AUTOSSH_GATETIME=0 autossh -M 0 -N \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -p 8222 \
  -L 19083:127.0.0.1:19083 \
  chase@47.115.88.183
```

本地验证：

```bash
curl -sS http://127.0.0.1:19083/healthz
curl -sS http://127.0.0.1:19083/v1/models
```

应返回三模型：

- `rwkv7-g1f-2.9b-20260420-ctx8192`
- `rwkv7-g1g-2.9b-20260526-ctx8192`
- `rwkv7-g1g-7.2b-20260523-ctx8192`

## 远端推理端启动方式

远端工作目录：

```bash
cd /home/chase/chase-rwkv-skills
```

共同参数：

```bash
--max-num-seqs -1
--max-num-batched-tokens 32768
--gpu-memory-utilization 0.98
--log-level info
```

GPU1 / 2.9B g1g：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m src.bin.run_infer_server \
  --model-path /home/chase/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1g-2.9b-20260526-ctx8192.pth \
  --model-name rwkv7-g1g-2.9b-20260526-ctx8192 \
  --host 127.0.0.1 \
  --port 18082 \
  --max-num-seqs -1 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.98 \
  --log-level info
```

GPU2 / 2.9B g1f：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m src.bin.run_infer_server \
  --model-path /home/chase/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1f-2.9b-20260420-ctx8192.pth \
  --model-name rwkv7-g1f-2.9b-20260420-ctx8192 \
  --host 127.0.0.1 \
  --port 18083 \
  --max-num-seqs -1 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.98 \
  --log-level info
```

GPU3 / 7.2B g1g：

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m src.bin.run_infer_server \
  --model-path /home/chase/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1g-7.2b-20260523-ctx8192.pth \
  --model-name rwkv7-g1g-7.2b-20260523-ctx8192 \
  --host 127.0.0.1 \
  --port 18084 \
  --max-num-seqs -1 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.98 \
  --log-level info
```

Router：

```bash
.venv/bin/python -m src.bin.run_infer_router \
  --host 127.0.0.1 \
  --port 19083 \
  --timeout-s 3600 \
  --forward-max-workers 4096 \
  --log-level info \
  --route rwkv7-g1g-2.9b-20260526-ctx8192=http://127.0.0.1:18082 \
  --route rwkv7-g1f-2.9b-20260420-ctx8192=http://127.0.0.1:18083 \
  --route rwkv7-g1g-7.2b-20260523-ctx8192=http://127.0.0.1:18084
```

## 重启顺序

只重启服务器上的推理端时，不要动本地 autossh 转发配置。

1. 先停远端 router `19083`。
2. 再停远端 infer server `18082`、`18083`、`18084`。
3. 按 GPU1/GPU2/GPU3 顺序重新拉起三个 infer server。
4. 最后重新拉起 router `19083`。
5. 本地验证 `http://127.0.0.1:19083/healthz` 和 `/v1/models`。

当前实测进程 PID 仅用于定位，不作为永久配置：

- GPU1 / `18082`：PID `353415`
- GPU2 / `18083`：PID `353417`
- GPU3 / `18084`：PID `353419`
- router / `19083`：PID `353810`

## 本地完整 benchmark 后台命令

本地工作目录：

```bash
cd /home/chase/GitHub/rwkv-skills
```

后台命令：

```bash
run_dir=logs/dispatch/full_benchmark_8222_20260609_stateauto_b768_full
mkdir -p "$run_dir"
set -a
source .env
set +a
export RWKV_SKILLS_DISABLE_CHECKER=1

setsid .venv/bin/python -m src.eval.scheduler.cli dispatch \
  --run-mode auto \
  --infer-base-url http://127.0.0.1:19083/v1 \
  --infer-models \
    rwkv7-g1g-2.9b-20260526-ctx8192 \
    rwkv7-g1f-2.9b-20260420-ctx8192 \
    rwkv7-g1g-7.2b-20260523-ctx8192 \
  --infer-protocol completions \
  --infer-timeout-s 1800 \
  --infer-max-workers 768 \
  --remote-batch-size 768 \
  --max-concurrent-jobs 1 \
  --disable-checker \
  --run-log-dir "$run_dir" \
  > "$run_dir/dispatcher.out" 2>&1 < /dev/null &

echo $! > "$run_dir/dispatcher.pid"
```

本轮已启动：

- PID：`6050`
- dispatcher 日志：`logs/dispatch/full_benchmark_8222_20260609_stateauto_b768_full/dispatcher.out`
- 当前恢复任务：DB task `59`，`rwkv7-g1g-7.2b-20260523-ctx8192 / human_eval_test`

## 验证命令

本地 DB 身份：

```bash
set -a
source .env
set +a
export PGHOST="${PG_HOST:-localhost}"
export PGPORT="${PG_PORT:-5432}"
export PGUSER="${PG_USER:-postgres}"
export PGDATABASE="${PG_DBNAME:-rwkv-eval}"
export PGPASSWORD="${PG_PASSWORD:-}"
psql -At -c "select current_database(), current_user, inet_server_addr(), inet_server_port();"
```

最近 task：

```bash
psql -P pager=off -F $'\t' -At -c "
select
  t.task_id,
  t.status,
  t.created_at,
  m.model_name,
  b.benchmark_name,
  b.benchmark_split,
  coalesce((select count(*) from completions c where c.task_id=t.task_id),0) as completions,
  coalesce((select count(*) from eval e join completions c on c.completions_id=e.completions_id where c.task_id=t.task_id),0) as evals,
  coalesce((select count(*) from scores s where s.task_id=t.task_id),0) as scores
from task t
join model m on m.model_id=t.model_id
join benchmark b on b.benchmark_id=t.benchmark_id
order by t.task_id desc
limit 20;
"
```

服务端 batch metrics：

```bash
curl -sS http://127.0.0.1:18083/v1/batch-metrics
curl -sS http://127.0.0.1:19083/healthz
curl -sS http://127.0.0.1:19083/v1/models
```

远端 GPU：

```bash
ssh -p 8222 chase@47.115.88.183 \
  'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
```

## 注意事项

- `run-mode auto` 用于继续 pending/failed/missing 任务，不覆盖已有 completed score。
- 当前完整 benchmark 是基础分数路径，显式禁用了 wrong-answer checker。
- 当前最佳吞吐证据来自 `g1f-2.9B / HumanEval` 正式 DB task 和 GPU2 服务端 batch metrics。
- 若后续扫到更高配置，需要用同样证据更新本文：DB task、score、completion/eval 计数、batch metrics、显存、5xx/OOM 情况。
