# Service-Based Perf Benchmark

这个 runner 现在只做一件事：把当前服务性能指标测出来。

支持两种用法：
- 直接压测一个已经启动好的 OpenAI-compatible 服务
- 由脚本自己拉起本地 `vLLM`，测完自动退出
- 用 `TOML` 配置文件统一控制模型、范围和 vLLM 参数

当前支持的协议：
- `openai-chat`

适用对象：
- 本地 `rwkv-skills` infer service
- `vLLM` OpenAI 兼容服务
- `llama.cpp` server 的 OpenAI 兼容接口

## 结果
默认输出到：

`results/performance/<model>/<timestamp>__<protocol>__<stack>.json`

## 基本参数
- `--config`：配置文件路径，或 `configs/perf/` 下的配置名
- `--base-url`：服务地址，例如 `http://127.0.0.1:8081`
- `--model`：服务暴露的模型名；不传则尝试读取 `/v1/models`
- `--tokenizer-type`：`rwkv` 或 `hf`
- `--tokenizer-ref`：HF tokenizer 名称/路径；RWKV 可不传
- `--ctx-lens`：输入 token 长度网格
- `--concurrency-levels`：并发网格
- `--batch-sizes`：batch size 网格；不传时默认复用 `--concurrency-levels`
- `--skip-concurrency-matrix`：跳过 `ctx_len x concurrency`
- `--skip-batch-size-matrix`：跳过 `ctx_len x batch_size`
- `--output-tokens`：每个请求固定输出长度
- `--gpu-index`：本机 GPU index，可选；提供后会用 NVML 采样显存峰值

如果你传了 `--launch-vllm`，还可以用这些参数直接拉起 vLLM：
- `--vllm-python`
- `--vllm-command`
- `--vllm-host`
- `--vllm-port`
- `--vllm-dtype`
- `--vllm-tensor-parallel-size`
- `--vllm-gpu-memory-utilization`
- `--vllm-max-model-len`
- `--vllm-max-num-seqs`
- `--vllm-max-num-batched-tokens`
- `--vllm-extra-args`

推荐把范围放进配置文件里改，不要每次手敲一长串参数。
如果你走当前项目环境，先用 `uv sync --extra vllm-cu128` 把 `torch/torchaudio/torchvision/vllm` 修成一套；
如果 vLLM 需要使用另一套单独的 `torch` 环境，优先设置 `python_executable` 或 `--vllm-python`，不要再手工拼整段 `command`。

## 配置文件

配置文件默认放在：

`configs/perf/<name>.toml`

推荐结构：

```toml
[service]
model = "Qwen/Qwen2.5-1.5B-Instruct"
engine_name = "vllm"
precision = "fp16"

[tokenizer]
type = "hf"
ref = "Qwen/Qwen2.5-1.5B-Instruct"

[workload]
ctx_lens = [512, 1024, 2048, 4096, 8192]
concurrency_levels = [1, 2, 4, 8, 16]
batch_sizes = [1, 2, 4, 8, 16]
output_tokens = 128
warmup_runs = 1
measure_runs = 3
temperature = 0.0
top_p = 1.0

[vllm]
launch = true
# 直接复用当前项目环境时，留空即可；脚本会复用当前 Python 解释器。
# python_executable = "/path/to/vllm-env/bin/python"
host = "127.0.0.1"
port = 8000
dtype = "half"
tensor_parallel_size = 1
gpu_memory_utilization = 0.9
max_model_len = 8192
max_num_seqs = 16
max_num_batched_tokens = 16384
extra_args = ["--no-enable-log-requests"]

[hardware]
gpu_index = 0
hardware_label = "local-gpu"
```

范围控制主要看 `[workload]`：
- `ctx_lens` 控制上下文长度网格
- `concurrency_levels` 控制并发网格
- `batch_sizes` 控制 batch-size 网格
- `skip_concurrency_matrix` / `skip_batch_size_matrix` 可以关掉某一类矩阵

vLLM 环境控制主要看 `[vllm]`：
- `python_executable`：指定启动 vLLM 的 Python 解释器，适合切到另一套带匹配 `torch+vllm` 的环境
- `command`：只有在你确实需要自定义完整启动命令时再用

## 指标
当前输出这些基础指标：
- `TTFT`
- `E2EL`
- `Input TPS`
- `Output TPS`
- `RPS`
- `Peak VRAM`
- `Peak VRAM Delta`
- `failure rate`

## 1. RWKV 本地服务
先启动服务：

```bash
cd /home/chase/GitHub/rwkv-skills
source .venv/bin/activate
nohup python -m src.bin.run_infer_server \
  --model-path /home/chase/GitHub/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1e-1.5b-20260309-ctx8192.pth \
  --model-name rwkv7-g1e-1.5b-20260309-ctx8192 \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8081 \
  --max-batch-size 32 \
  --batch-collect-ms 5 \
  > /tmp/rwkv7-g1e-1.5b-infer.log 2>&1 &
```

再跑 benchmark：

```bash
cd /home/chase/GitHub/rwkv-skills
source .venv/bin/activate
python -m src.bin.run_perf_benchmark \
  --base-url http://127.0.0.1:8081 \
  --model rwkv7-g1e-1.5b-20260309-ctx8192 \
  --protocol openai-chat \
  --tokenizer-type rwkv \
  --engine-name rwkv-skills-infer \
  --precision fp16 \
  --ctx-lens 512,1024,2048,4096,8192 \
  --concurrency-levels 1,2,4,8,16,32 \
  --output-tokens 128 \
  --warmup-runs 1 \
  --measure-runs 3 \
  --gpu-index 0 \
  --hardware-label "local-gpu"
```

## 2. vLLM

### 2.1 脚本直接拉起 vLLM

最简单的方式是让脚本自己启动 vLLM：

```bash
cd /home/chase/GitHub/rwkv-skills
uv sync --extra vllm-cu128 \
  --reinstall-package torch \
  --reinstall-package torchaudio \
  --reinstall-package torchvision \
  --reinstall-package vllm
uv run --extra vllm-cu128 python -m src.bin.run_perf_benchmark \
  --config qwen2_1_5b_instruct
uv run --extra vllm-cu128 python -m src.bin.run_perf_benchmark \
  --config qwen2_5_1_5b_instruct
```

如果你有一套单独的 vLLM 环境，直接覆盖解释器就行：

```bash
uv run python -m src.bin.run_perf_benchmark \
  --config qwen2_1_5b_instruct \
  --vllm-python /path/to/vllm-env/bin/python
```

仓库里已经放了两份可直接改的样例：
- `configs/perf/qwen2_1_5b_instruct.toml`
- `configs/perf/qwen2_5_1_5b_instruct.toml`

### 2.2 压测一个已经启动好的 vLLM 服务

如果你已经手动把 vLLM 启在 `127.0.0.1:8000`，也可以直接打这个服务：

```bash
cd /home/chase/GitHub/rwkv-skills
source .venv/bin/activate
python -m src.bin.run_perf_benchmark \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tokenizer-type hf \
  --tokenizer-ref Qwen/Qwen2.5-7B-Instruct \
  --engine-name vllm \
  --precision bf16 \
  --ctx-lens 512,1024,2048,4096,8192 \
  --concurrency-levels 1,2,4,8,16,32 \
  --batch-sizes 1,2,4,8,16,32 \
  --output-tokens 128 \
  --warmup-runs 1 \
  --measure-runs 3 \
  --gpu-index 0 \
  --hardware-label "local-gpu"
```

## 3. llama.cpp
如果 `llama.cpp` server 提供 OpenAI-compatible chat 接口，也可以直接跑：

```bash
cd /home/chase/GitHub/rwkv-skills
source .venv/bin/activate
python -m src.bin.run_perf_benchmark \
  --base-url http://127.0.0.1:8082 \
  --model llama-3.1-8b-instruct-q4km \
  --protocol openai-chat \
  --tokenizer-type hf \
  --tokenizer-ref /path/to/llama-tokenizer \
  --engine-name llama.cpp \
  --precision q4_k_m \
  --ctx-lens 512,1024,2048,4096,8192 \
  --concurrency-levels 1,2,4,8,16,32 \
  --output-tokens 128 \
  --warmup-runs 1 \
  --measure-runs 3 \
  --gpu-index 0 \
  --hardware-label "local-gpu"
```

## 备注
- 想公平对比，三边尽量统一 `ctx-lens / concurrency / output-tokens / temperature / top-p / warmup / measure-runs`
- HF 模型建议用各自原生 tokenizer
- `Peak VRAM` 依赖本机 NVML 采样；如果 benchmark 机器和服务机器不是同一台，就不要传 `--gpu-index`
- 结果里的 `point_kind` 会区分 `concurrency` 和 `batch_size` 两类测试点
- 如果你要测 Qwen2 和 Qwen2.5 的 1.5B，对比时尽量保持同一份 `[workload]` 和 `[vllm]` 参数，只改模型名
