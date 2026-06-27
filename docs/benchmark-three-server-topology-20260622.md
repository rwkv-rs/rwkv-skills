# Benchmark Server Topology - 2026-06-22

This file records the verified benchmark deployment targets and current run
authority. Do not store SSH passwords in this document.

## Current Authority

- Current DB authority: local PostgreSQL from this checkout's `.env`.
- Current DB identity verified on 2026-06-22: `rwkv-eval`, user `postgres`,
  host `127.0.0.1`, port `5432`.
- Current scheduler host: local machine, project path
  `/home/chase/GitHub/rwkv-skills`.
- Current inference entrypoint for local scheduler:
  `http://127.0.0.1:19083/v1`.
- Current run tag: `local_6gpu_resume_20260622`.
- Current dispatcher log:
  `logs/scheduler/local_6gpu_resume_20260622/dispatcher_slots6.log`.
- Current monitor log:
  `logs/monitor/local_6gpu_resume_20260622/progress.log`.
- Current monitor command:
  `.venv/bin/python scripts/watch_benchmark_progress.py --run-tag local_6gpu_resume_20260622 --interval-seconds 300 --score-id-min 182`.
- Current local score baseline before this resumed run: latest `score_id=181`.
  First new score observed after resume: `score_id=182`, task `442`,
  `code_human_eval_naive/human_eval`,
  `rwkv7-g1g-7.2b-20260523-ctx8192`.
- Second new score observed after resume: `score_id=183`, task `474`,
  `code_human_eval_naive/human_eval`,
  `rwkv7-g1f-7.2b-20260414-ctx8192`.
- 2333 is not the active authority for this run because SSH port `2333` was
  refusing connections. A later sync script should forward local DB results to
  2333 when it is reachable.

## Local Forwards

- `127.0.0.1:19083` forwards to 8222 `127.0.0.1:19083`.
- `127.0.0.1:19090` forwards to 157 `127.0.0.1:18090`.
- `127.0.0.1:19091` forwards to 157 `127.0.0.1:18091`.
- `127.0.0.1:19092` forwards to 157 `127.0.0.1:18092`.
- `127.0.0.1:19093` forwards to 157 `127.0.0.1:18093`.
- The current scheduler uses 19083 only. 19090-19093 are kept as direct health
  and metrics probes for the 157 backends.

## 8222 Server

- SSH: `ssh chase@47.115.88.183 -p 8222`
- Project path: `/home/chase/chase-rwkv-skills`
- Python env: `/home/chase/chase-rwkv-skills/.venv/bin/python`
- Current router: `127.0.0.1:19083`
- Current project-owned active GPUs:
  - GPU2: `rwkv7-g1g-7.2b-20260523-ctx8192`, port `18084`
  - GPU3: `rwkv7-g1f-7.2b-20260414-ctx8192`, port `18085`
- Do not use GPU0 for this run. It has a non-project `rwkv` user service:
  `app.py --model-path /home/rwkv/alic-li/rwkv7-g1g-7.2b-20260523-ctx8192.pth`.
- GPU1 is not used in this run to avoid contending with other users.

### 8222 Weights

- `/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1f-1.5b-20260419-ctx8192.pth`
- `/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1f-2.9b-20260420-ctx8192.pth`
- `/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1f-7.2b-20260414-ctx8192.pth`
- `/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1g-1.5b-20260526-ctx8192.pth`
- `/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1g-13.3b-20260523-ctx8192.pth`
- `/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1g-2.9b-20260526-ctx8192.pth`
- `/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1g-7.2b-20260523-ctx8192.pth`

## 157 Server

- SSH: `ssh -J chase@47.115.88.183:8222 rwkv@192.168.0.157`
- Project path: `/home/rwkv/chase/rwkv-skills`
- Python env: `/home/rwkv/chase/rwkv-skills/.venv/bin/python`
- GPUs: 4 x NVIDIA GeForce RTX 4090 D.
- Current active services:
  - GPU0: `rwkv7-g1f-1.5b-20260419-ctx8192`, port `18090`
  - GPU1: `rwkv7-g1g-1.5b-20260526-ctx8192`, port `18091`
  - GPU2: `rwkv7-g1f-2.9b-20260420-ctx8192`, port `18092`
  - GPU3: `rwkv7-g1g-2.9b-20260526-ctx8192`, port `18093`

### 157 Formal Weights

- `/home/rwkv/chase/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1f-1.5b-20260419-ctx8192.pth`
- `/home/rwkv/chase/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1f-2.9b-20260420-ctx8192.pth`
- `/home/rwkv/chase/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1g-1.5b-20260526-ctx8192.pth`
- `/home/rwkv/chase/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1g-2.9b-20260526-ctx8192.pth`
- `/home/rwkv/chase/rwkv-skills/weights/rwkv7-g1g-7.2b-20260523-ctx8192.pth`

## 2333 Server

- SSH: `ssh -i ~/.ssh/id_server_new caizus@47.115.88.183 -p 2333`
- Status on 2026-06-22: unavailable from local machine, `Connection refused`.
- Historical project path:
  `/home/caizus/Projects/MachineLearning/chase-rwkv-skills`
- Historical Python env:
  `/home/caizus/Projects/MachineLearning/rwkv-skills/.venv/bin/python`
- Historical DB target: PostgreSQL database `chase_rwkv_skills`.
- Use this only after the SSH service is reachable again. Current results should
  be synced from local DB to this DB by an explicit migration script.

## Current Router Model Map

The active `19083` router currently exposes these six models:

- `rwkv7-g1f-1.5b-20260419-ctx8192` -> 157 GPU0, port `18090`
- `rwkv7-g1g-1.5b-20260526-ctx8192` -> 157 GPU1, port `18091`
- `rwkv7-g1f-2.9b-20260420-ctx8192` -> 157 GPU2, port `18092`
- `rwkv7-g1g-2.9b-20260526-ctx8192` -> 157 GPU3, port `18093`
- `rwkv7-g1f-7.2b-20260414-ctx8192` -> 8222 GPU3, port `18085`
- `rwkv7-g1g-7.2b-20260523-ctx8192` -> 8222 GPU2, port `18084`

## Deployment Notes

- 157's `/home/rwkv/chase/rwkv-skills` is an old checkout. It is currently
  usable for the four 1.5/2.9 services, but can be replaced by uploading the
  current checkout when needed.
- Keep 8222 GPU0 and other users' processes untouched unless explicitly
  reallocated by the owner.
- Current scheduling policy at 2026-06-22 20:32 CST: six scheduler slots for
  each 1.5B/2.9B model, and one slot for each 7.2B model. Continue adjusting
  from `/v1/batch-metrics`, DB completion deltas, and timeout/error logs.
- Use `scripts/sync_eval_db_to_remote.py --score-id-min 182` as the dry-run
  check for results that must later be forwarded from local DB to the 2333 DB.
  Add `--write` only after the 2333 DB tunnel/connection is verified.
