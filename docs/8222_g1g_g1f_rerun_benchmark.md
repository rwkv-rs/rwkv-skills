# 8222 服务器 g1g/g1f 全量测评启动文档（rerun）

> 仓库路径：`/home/chase/chase-rwkv-skills`（即远端 `~ /chase-rwkv-skills` 的实际路径）

目标：
- g1g、g1f 两个 2.9B 模型
- 使用 `rerun`，`.env` 的 DB 配置
- 不启用 checker（`--disable-checker`）
- jobs 并发 1（`--max-concurrent-jobs 1`）
- 两模型按顺序执行，均在后台运行

## 1）前置

- 只改动测评命令，不动模型文件和服务进程本体。
- 假设远端环境中已加载 `source .env`（含 DB 与 judge 信息），并已完成端口转发。

## 2）后台顺序启动命令（推荐）

在本机执行：

```bash
rtk ssh -p 8222 -o BatchMode=yes -o StrictHostKeyChecking=accept-new chase@47.115.88.183 '
set -a
cd /home/chase/chase/rwkv-skills
source .env
set +a
mkdir -p logs/manual

nohup bash -lc "
  /home/chase/chase/rwkv-skills/.venv/bin/python -m src.eval.scheduler.cli dispatch \
    --run-mode rerun \
    --infer-base-url http://127.0.0.1:18084/v1 \
    --infer-models rwkv7-g1g-2.9b-20260526-ctx8192 \
    --infer-protocol openai \
    --infer-api-key \"${OPENAI_API_KEY}\" \
    --max-concurrent-jobs 1 \
    --disable-checker \
    --run-log-dir /home/chase/chase/rwkv-skills/logs/manual/rerun_g1g 2>&1 | tee /home/chase/chase/rwkv-skills/logs/manual/rerun_g1g_dispatch.log

  /home/chase/chase/rwkv-skills/.venv/bin/python -m src.eval.scheduler.cli dispatch \
    --run-mode rerun \
    --infer-base-url http://127.0.0.1:18085/v1 \
    --infer-models rwkv7-g1f-2.9b-20260420-ctx8192 \
    --infer-protocol openai \
    --infer-api-key \"${OPENAI_API_KEY}\" \
    --max-concurrent-jobs 1 \
    --disable-checker \
    --run-log-dir /home/chase/chase/rwkv-skills/logs/manual/rerun_g1f 2>&1 | tee /home/chase/chase/rwkv-skills/logs/manual/rerun_g1f_dispatch.log
" > /home/chase/chase/rwkv-skills/logs/manual/rerun_g1g_g1f_bg.log 2>&1 & echo "rerun_seq_pid=$!"
'
```

说明：
- 两个模型顺序执行（g1g->g1f），不会并行启动 worker。
- 日志：
  - `logs/manual/rerun_g1g_dispatch.log`
  - `logs/manual/rerun_g1f_dispatch.log`
  - `logs/manual/rerun_g1g_g1f_bg.log`（外层总控）

## 3）结束后清理 GPU3 上推理端（18084）

在完成后清空 18084 端口：

```bash
rtk ssh -p 8222 -o BatchMode=yes -o StrictHostKeyChecking=accept-new chase@47.115.88.183 \
"lsof -tiTCP:18084 -sTCP:LISTEN | xargs -r kill -9"
```

## 4）为什么这样做

- `--run-mode rerun`：覆盖重跑，避免受历史任务影响
- `--disable-checker`：完全不走 checker
- `--max-concurrent-jobs 1`：jobs 参数固定为 1
- `.env` 中 DB 配置由 scheduler 启动时自动读取（`load_env_file`）
