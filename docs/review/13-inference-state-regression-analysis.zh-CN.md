# 推理引擎 state 隐性劣化分析

日期：2026-05-03

范围：

- 本地分支：`agent-bench`，HEAD `2a92e23 Fix remote BFCL prompt completion path`
- 本地代码：`src/infer/rwkv7/rwkv7.py`、`src/infer/engine.py`、`src/infer/lightning_engine.py`、`src/infer/state_pool.py`
- 上游对照：`https://github.com/BlinkDL/Albatross`，已临时克隆到 `/tmp/Albatross`

## 初步结论

当前最可疑的点不是单纯的 `state[1]` 使用 FP16。本地 `rwkv7.py` 与 Albatross `faster_251101/reference/rwkv7.py` 的 state dtype 基本一致，都是 FP16 recurrent state + `state[2]` elapsed token counter。Albatross `reference/rwkv7.py` 的早期参考实现确实使用 FP32 kv state，但 `faster_251101`/`faster2_251201` 已经转向 FP16 state，并在 README 里说明 FP16 state 依靠 deterministic dithering 接近 FP32 loss。

更高风险的隐性错误有两个：

1. 本地 `RWKV7_*_OP` custom op 声明为 `mutates_args=()`，但底层 CUDA op 会原地修改 `state`。
2. lightning prefix cache 只按 token prefix 命中 state，没有绑定 model/path/weights hash/engine schema，可能跨模型或跨实现复用 stale state。

这两类问题都符合“只有百分之几样本出问题、问题发生后退化很隐蔽、最终表现为严重复读”的形态。

## 风险 1：custom op mutation 元数据错误

本地位置：

- `src/infer/rwkv7/rwkv7.py:83`
- `src/infer/rwkv7/rwkv7.py:101`
- `src/infer/rwkv7/rwkv7.py:119`
- `src/infer/rwkv7/rwkv7.py:138`

现状：

```python
@torch.library.custom_op("mylib::RWKV7_BATCH_OP", mutates_args=())
def RWKV7_BATCH_OP(...):
    return WKV_7_SEQ_BATCH.apply(...)
```

但同文件后面明确写着：

```python
xx = RWKV7_BATCH_OP(...)  # using CUDA to modify state in-place
```

底层 C++/CUDA 也把 `state` 当可写指针传入：

- `src/infer/rwkv7/cuda/rwkv7_state_fwd_fp16.cpp` 的 `forward_one/forward_seq`
- `src/infer/rwkv7/cuda/rwkv7_state_fwd_fp16.cu` 的 kernel 会读写 `_state`

上游对照：

- `/tmp/Albatross/faster_251101/reference/rwkv7.py` 没有把这些 WKV wrapper 注册成 `torch.library.custom_op`，而是通过 disable wrapper 避免编译器误判。
- `/tmp/Albatross/faster2_251201/reference/rwkv7.py` 已经注册为 custom op，但写的是 `mutates_args=("state",)`。

影响判断：

- 这是 correctness 级别风险，不只是性能或风格差异。
- PyTorch 编译器、TorchScript alias analysis、functionalization 看到 `mutates_args=()` 时，可以把这个 op 当成不修改输入的纯 op。
- 即使 eager 下通常能实际写入，错误 schema 在编译、图缓存、调度、别名分析或未来 PyTorch 版本中都可能产生不稳定行为。
- 这个问题由提交 `12a74a4 Update fastest rwkv7 infer` 引入或固化；该提交同步更新了 fp16 state kernel、`rwkv7.py` 和 HIP/CUDA 相关代码。

建议优先验证：

1. 在一个临时分支里只把四个 `RWKV7_*_OP` 的 `mutates_args=()` 改成 `mutates_args=("state",)`。
2. 固定 seed、固定 prompt 集、固定采样参数，对比 current vs patched 的 token 序列和复读比例。
3. 优先用 `--engine-mode classic` 验证，排除 lightning cache 干扰。

## 风险 2：lightning prefix cache 没有模型隔离

本地位置：

- `src/infer/state_pool.py`
- `src/infer/lightning_engine.py`
- 默认 DB path：`rwkv_sessions.db`

现状：

- prefix cache 的 `state_id` 是 token id 序列字符串。
- DB 表 `prefix_cache` 没有 model name、model path、weights hash、engine version、state schema version。
- `match_prefix_state()` 只按 token prefix 查 state，然后直接恢复到当前模型的 active slot。

影响判断：

- 如果同一个 `rwkv_sessions.db` 被不同模型、不同权重、不同 engine 实现或不同 state schema 复用，缓存会静默命中。
- 一旦命中，当前模型会从别的模型 state 继续生成。这个错误不会报 shape mismatch，只会表现为少数样本突然质量变差或复读。
- standalone server 如果用 `--engine-mode lightning` 且不指定 `--state-db-path`，默认会复用工作目录下的 `rwkv_sessions.db`。
- fleet 模式在指定 `--state-db-dir` 时会生成 per-model sqlite 路径，风险较低。

相关提交：

- `001ab05 feat(infer): migrate remote inference to OpenAI chat completions with MC fallback` 新增 `lightning_engine.py` 和 `state_pool.py`。
- 后续 `6d2d8aa`、`2b4aab5` 改过 lightning 的流式输出、约束解码，但没有看到模型隔离字段。

建议优先验证：

1. 复现时先删除或更换 `rwkv_sessions.db`。
2. 对每个模型强制传唯一 `--state-db-path`。
3. 同一批 prompts 对比 `classic`、`lightning fresh db`、`lightning reused db`。
4. 如果 reused db 明显更容易复读，基本可定位到 prefix cache 污染。

## 风险 3：lightning sampler state 行顺序可能错配

本地位置：

- `src/infer/lightning_engine.py` 的 `_sample_rows()`

现状：

- 当 `len(sample_rows) == active_count` 时，代码直接使用 `_sampler_states_view(active_count)`。
- 这个假设只有在 `sample_rows == [0, 1, ..., active_count-1]` 时成立。
- lightning 中 `sample_rows = direct_sample_rows + forward_sample_rows`，当某些 row 来自 prefix cache 的 `ready_logits`、另一些 row 走 forward 时，`sample_rows` 可能是排列而不是自然顺序。

影响判断：

- 这主要会错配 rapid-sampling 的随机状态，而不是 RWKV hidden state。
- 它更可能导致 seed 不稳定、采样分布轻微漂移；单独造成严重复读的可能性低于前两个问题。
- 但它会增加“少量样本非确定性劣化”的噪声，建议一并修。

建议验证：

- 只在 `sample_rows == list(range(active_count))` 时走 direct view。
- 其他情况一律按 row_index copy sampler state 子集。
- 增加一个 fake sampler 单测，构造 `sample_rows=[1,0]` 且 `active_count=2`，验证 sampler row 不串。

## 已排除或低优先级

`state[1]` FP16 本身暂不作为第一嫌疑。

- 本地 `generate_zero_state()` 与 Albatross `faster_251101` 一致，`state[1]` 是 `DTYPE=torch.half`。
- Albatross `faster2_251201/README.md` 专门说明 FP16 state + deterministic dithering 的 loss 接近 FP32。
- 如果线上只在长上下文或特殊 prompt 出现复读，仍可做 FP32 kernel A/B，但这不是当前代码和上游最快实现的直接差异。

`state[2]` missing in classic engine 之前确实修过。

- 提交：`0e1c0cc fix: include missing states[2] in state_view`
- 当前 `src/infer/engine.py` 已把 `states[2][:active_count]` 传入 `forward_batch`。
- 因此 classic continuous batching 的 elapsed_t 缺失问题不是当前 HEAD 的主要嫌疑。

## 提交记录线索

- `5d8a461 feat: Integrate RWKV7 reference implementation`
  - 初始引入 RWKV7 reference implementation、CUDA kernel、vocab。
- `12a74a4 Update fastest rwkv7 infer`
  - 大幅改 `src/infer/rwkv7/rwkv7.py` 和 CUDA/HIP kernel。
  - 引入当前 fp16 dither kernel 形态。
  - 当前可疑的 `mutates_args=()` 位于该提交后的代码中。
- `0e1c0cc fix: include missing states[2] in state_view`
  - 修复 classic continuous batching 没有传 elapsed token counter 的问题。
- `c690716 feat: 添加space更多可视化功能以及rapid-sampling`
  - 引入 rapid-sampling 相关实现和 engine 采样改动。
- `001ab05 feat(infer): migrate remote inference to OpenAI chat completions with MC fallback`
  - 新增 OpenAI-compatible backend/server/service。
  - 新增 `lightning_engine.py` 和 `state_pool.py`，引入 prefix cache。
- `6d2d8aa fix:add openai used type`
  - 大幅改流式输出、logprobs、engine/lightning 输出事件。
- `2b4aab5 Add BFCL v3 constrained decoding pipeline`
  - 改 constraints 和 engine/lightning 约束解码路径。

## 建议排查顺序

1. 用 `--engine-mode classic` + fresh process 复现。如果 classic 也复读，优先查 `RWKV7_*_OP mutates_args`。
2. 用 `--engine-mode lightning` 但换空 DB 复现。如果空 DB 正常、旧 DB 异常，优先查 prefix cache 污染。
3. 临时 patch `mutates_args=("state",)` 做 A/B。这个改动很小、风险低、和上游 `faster2_251201` 一致。
4. 给 prefix cache 加 model/schema namespace，再清理旧 DB。
5. 修 lightning sampler row order 假设，补一个排列顺序单测。

## 当前验证状态

已完成：

- 拉取官方 Albatross 到 `/tmp/Albatross` 做只读对照。
- 查看最近提交、推理目录提交、`rwkv7.py` blame。
- 对比本地 `rwkv7.py` 与 Albatross `reference`、`faster_251101`、`faster2_251201`。

未完成：

- 未跑单测。当前环境里 `pytest` 不在 PATH，执行 `pytest tests/test_state_pool.py tests/test_lightning_engine.py -q` 返回 `pytest: command not found`。
- 未做 GPU 上真实模型 A/B。用户要求先分析、不直接改 code，所以这里只给验证方案。
