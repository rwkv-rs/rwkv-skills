# RWKV 推理引擎性能画像（prefill vs decode）

更新时间：2026-07-01

## 目的

在继续做更大的引擎改动之前，先测清楚**时间到底花在哪**。

背景结论（已确认，不要重测）：

- decode CUDA Graph **已经实现且默认开启**。RWKV7 在 `vllm-rwkv/vllm/model_executor/models/config.py:507-520` 强制
  `CUDAGraphMode.FULL_DECODE_ONLY`（除非显式 `--enforce-eager`），捕获 decode batch 到 1024。
  已验证的 410 tok/s 峰值就是在 decode graph 开启下跑出来的。
- 真正待定的引擎级杠杆是 **chunk-parallel prefill**（prefill 在 `FULL_DECODE_ONLY` 下仍是 eager 的纯串行 WKV）。
  但它主要收益在**低并发 / 长 prompt 的 TTFT**，不一定抬高并发吞吐。

本画像用生产协议（raw `completions`）分别压 **prefill-bound** 和 **decode-bound** 两个区间，
输出用来判定 chunk-prefill 是否值得做（见文末决策门）。

## 前置

- 本机 autossh 转发已起：`http://127.0.0.1:19083/v1`（见 `8222-best-performance-config-and-startup-20260609.zh-CN.md`）。
- 服务健康：

```bash
curl -sS http://127.0.0.1:19083/healthz
curl -sS http://127.0.0.1:19083/v1/models
```

- 压测走 raw `completions`，与生产 eval 路径一致（`--infer-protocol completions`），不套 chat 模板。

## 运行

本机工作目录 `cd /home/chase/GitHub/rwkv-skills`。

### 1. Prefill-bound（TTFT vs 上下文长度，低并发）

```bash
.venv/bin/python -m src.bin.run_perf_benchmark --config rwkv7_g1f_2p9b_service_prefill
```

关注：`ttft_s`、`input_tps` 随 `ctx_len` 增长的曲线（512→8192）。这条曲线就是串行 prefill 的成本。

### 2. Decode-bound（output tok/s，生产并发）

```bash
.venv/bin/python -m src.bin.run_perf_benchmark --config rwkv7_g1f_2p9b_service_decode
```

关注：`output_tps`、`decode_window_s` 在并发 64→768 下的表现，确认 decode graph 的效率。

结果落 `results/performance/<model>/<timestamp>__completions__vllm-rwkv-service.json`。

## 归因对照（可选，各一个匹配点）

1. **协议开销**：同一个点分别用 `--protocol completions` 和 `--protocol openai-chat`，量化 chat 模板开销
   （即生产为什么走 completions）。
2. **decode graph 的价值**：远端服务分别用默认（graph 开）和 `--enforce-eager`（graph 关）重启，
   对比 decode-bound 的 `output_tps`。
3. **WKV 精度**：远端 `VLLM_RWKV7_WKV_MODE=fp16` vs `fp32io16`，看吞吐/精度取舍。

## 结果汇总（运行后填写）

| 区间 | ctx_len | 并发 | ttft_s (p50) | input_tps | output_tps | decode_window_s | failure_rate | peak VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill | 512 | 1 | | | | | | |
| prefill | 8192 | 1 | | | | | | |
| prefill | 8192 | 4 | | | | | | |
| decode | 512 | 128 | | | | | | |
| decode | 512 | 768 | | | | | | |

VRAM 从服务端 `/v1/batch-metrics` 读，不用本机 NVML（压测机 ≠ 服务机）。

## 决策门

- 若高 ctx + 低并发下 **TTFT 占 E2EL 比重大**（prefill 是瓶颈）→ chunk-parallel prefill 值得做，进入该 plan。
- 若 prefill 即使到 8k 也很便宜、decode 全程主导 → chunk-prefill 不会动指针，
  转向 decode 侧（`vllm-rwkv/vllm/v1/worker/gpu/model_states/rwkv.py:188-193` 的 state 压缩 `clone`、
  capture-size / padding 调优）。

结论记在本文件，并按 runbook 惯例留证据（config、结果 JSON 路径、batch-metrics、failure_rate）。
