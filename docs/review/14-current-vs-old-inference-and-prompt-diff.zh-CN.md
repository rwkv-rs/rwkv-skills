# 当前项目 vs `~/rwkv-skills` 推理引擎与提示词差异对比

日期：2026-05-12

对比对象：

- 当前项目：`/home/chase/GitHub/rwkv-skills`
- 旧项目：`/home/chase/rwkv-skills`
- 对比口径：两个目录当前工作树的实际文件内容，不按 git HEAD 回退；忽略 `__pycache__`、`out/`、`results/` 这类运行产物。

## 总结

最重要的结论：

1. **底层 RWKV7 模型实现和 CUDA/HIP/Triton 算子一致。**  
   `src/infer/rwkv7/rwkv7.py`、`rwkv7_state_fwd_fp16.{cpp,cu}`、rapid-sampling `{cpp,cu}`、HIP 文件、Triton 文件在两个项目中内容哈希一致。因此如果只看 RWKV block、state kernel、rapid-sampling kernel，两个项目没有实现差距。

2. **Python 推理调度层不一致。**  
   旧项目的 `src/infer/engine.py` 是较小的 classic continuous batching；当前项目在同名 classic engine 上增加了 chunked prefill、流式 token delta、top logprobs、约束解码、UTF-8 安全输出、字节级 stop suffix 截断等。当前项目还新增了 `backend.py`、`lightning_engine.py`、`state_pool.py`、`server.py`、`service.py`、`api.py`、`openai_service.py` 等远端/服务化推理层，旧项目没有这些层。

3. **如果要求“推理引擎一致”，当前必须明确使用 classic 路径，并把 prefill 行为对齐。**  
   旧 engine 逐 token prefill；当前默认 `DEFAULT_PREFILL_CHUNK_SIZE = 16`。这会走 RWKV 的 sequence/batch kernel 路径，不是旧版的逐 token 调度。理论上 state 应一致，但如果要做 bit-level 或问题定位级别的一致性，建议把当前 `prefill_chunk_size` 设置为 `1` 做 A/B。

4. **提示词差距很大，尤其是知识选择题、数学、代码生成。**  
   旧项目提示词散落在各 evaluator 中；当前项目集中到 `src/eval/prompt_builders.py` 的 expected-context/placeholder 模式。function-calling 的简单 JSON 工具调用 prompt 基本一致，但当前项目新增了 BFCL v3、tau、MCP、BrowseComp 等复杂 prompt。

5. **旧项目并不比当前项目在 CUDA/kernel 层更完整；旧项目更“完善”的部分主要体现在部分 benchmark prompt 习惯和老 pipeline 语义。**  
   需要迁移的是 prompt 与 classic engine 调度语义，不是 CUDA 算子。

## 文件级对比

### 推理目录

| 范围 | 结论 |
| --- | --- |
| `src/infer/rwkv7/rwkv7.py` | 完全一致。包括 `SPMV_OP`、`RWKV7_ONE_OP`、`RWKV7_SEQ_OP`、`RWKV7_ONE_BATCH_OP`、`RWKV7_BATCH_OP` 和 `RWKV_x070`。 |
| `src/infer/rwkv7/cuda/*` | 完全一致。 |
| `src/infer/rwkv7/hip/*` | 完全一致。 |
| `src/infer/rwkv7/rwkv_mm_op_triton.py` | 完全一致。 |
| `src/infer/rapid_sampling/*` | 完全一致。 |
| `src/infer/engine.py` | 不一致，是主要差异。当前 823 行，旧版 400 行。 |
| `src/infer/sampling.py` | 不一致。当前新增 config 校验、token/logprob delta 结构。 |
| `src/infer/model.py` | 小差异。旧版有模型枚举元数据，当前移除；模型加载逻辑一致。 |
| `src/infer/rwkv7/utils.py` | 只有类型标注和 `Sequence[int]` 接口差异，语义基本一致。 |
| 当前新增 `backend.py`/`lightning_engine.py`/`state_pool.py`/`server.py`/`service.py`/`api.py`/`openai_service.py`/`constraints/*` | 旧项目没有对应实现。属于当前项目新增推理服务、远端调用、OpenAI-compatible API、prefix cache、约束解码能力。 |

底层算子哈希抽样：

| 文件 | SHA256 |
| --- | --- |
| `src/infer/rwkv7/rwkv7.py` | `64b374ceb645abb396efc47cd3ed0226042b4d9375ae9953b1f56232fb8ffa58` |
| `src/infer/rwkv7/rwkv_mm_op_triton.py` | `b36b8665cd5e94220ad35bd606f3d4332351ca6a77686984c701fc8002fb1f88` |
| `src/infer/rwkv7/cuda/rwkv7_state_fwd_fp16.cpp` | `d84402f585abe304c963459f494a8124196cd5c093b94e85fbbf70f69c080889` |
| `src/infer/rwkv7/cuda/rwkv7_state_fwd_fp16.cu` | `4b2518435e14883ecb47364d528a52c536a5cdeaa373b797facdcaefcb9e262d` |
| `src/infer/rapid_sampling/sampling.cpp` | `9aab24fad7863708b1f0f5e4a7a21a8567749968538f01f601a1a055bb7eb6d8` |
| `src/infer/rapid_sampling/sampling.cu` | `f7c213ae4a05dcd1e5b96924cc901ac7175b097df7ca09e4335083af2758081e` |

两个项目上述哈希相同。

## 函数级推理对比

### `src/infer/engine.py`

| 函数/类 | 旧项目行为 | 当前项目行为 | 影响 |
| --- | --- | --- | --- |
| `TokenizerProtocol` | `encode/decode` 协议。 | 相同。 | 无差异。 |
| `RWKVModelProtocol` | `generate_zero_state/forward/forward_batch` 协议。 | 相同。 | 无差异。 |
| `InferenceEngine.__init__` | 保存 model/tokenizer。 | 相同。 | 无差异。 |
| `InferenceEngine.generate` | 参数：`prompts`、`sampling`、`batch_size`、`progress_desc`、`probe_only`、`on_complete`、`prompt_stop_suffixes`、`prompt_seeds`、`preserve_prompt_whitespace`。 | 移除 `preserve_prompt_whitespace`，新增 `prefill_chunk_size`、`on_token`、`prompt_constraints`、`top_logprobs`、`show_progress`。 | 当前生成路径默认保留 prompt 原始空白；旧版默认 `strip()`。若旧 benchmark prompt 依赖首尾空白，需要对齐。 |
| `_normalize_prompt` | 旧版存在，默认 `prompt.strip()`。 | 当前删除。 | 生成 token 序列可能不同。 |
| `_ActiveTask` | 只记录 prompt、pending token、generated token、stop suffix、新 token、finish reason。 | 增加 prompt 剩余 token 数、generated count、token event、UTF-8 buffer、pending stop token、bytes stop suffix、constraint runtime、pending generated token。 | 当前支持 chunked prefill、流式输出、字节级 stop、约束解码。 |
| `_continuous_batching` 参数校验 | 校验 seeds 和 stop suffix 长度。 | 额外校验 constraints；规范化 `prefill_chunk_size`、`top_logprobs`；调用 `sampling.checked(vocab_size)`。 | 当前对非法采样参数更稳，但和旧版非法参数行为不同。 |
| prompt 编码 | 默认 `prompt.strip()`，除非 `preserve_prompt_whitespace=True`。 | 永远 `tokenizer.encode(prompt)`。 | 这是 classic engine 最直接的输入差异。 |
| prompt prefill | 每轮每个 active task 只 pop 一个 token，统一走一 token step。 | prompt 阶段一次取 `prefill_chunk_size` 个 token；prompt 完成的 row 才进入 sampling。 | 默认 16 token chunk 会改变 kernel 调用形态。要求和旧版严格一致时应设置为 1。 |
| `_sampler_states_view` | 取 active rows 对应随机状态。 | 相同。 | 无差异。 |
| `_set_sampler_seed` | 每个 slot 用 `rapid_sampler.setup_rand(seed, 1)` 覆盖。 | 相同，但 `prompt_seeds` 可为 `None` 元素。 | 当前兼容性更强。 |
| `_validate_sampled_tokens` | 校验 sampled token 数量与范围。 | 相同。 | 无差异。 |
| `_reset_slot` | 清 penalties、`states[0]`、`states[1]`、`states[2]`。 | 相同。 | 无差异。 |
| `_swap_sampler_state_rows` | slot remove 时交换 sampler state。 | 相同。 | 无差异。 |
| `_remove_slot` | 将最后一个 slot 搬到空 slot，并同步 state/penalty/sampler state。 | 相同。 | 无差异。 |
| `_sample_subset` | 不存在。旧版每步对所有 active rows sampling。 | 当前只对需要 sampling 的 rows 采样，支持 prompt chunk 后部分 row sampling。 | 当前新增能力；classic 路径在 full rows 时通常自然顺序安全。 |
| stop token 处理 | 若 sampled token 是 stop token，不加入 generated tokens。 | 相同。 | 无差异。 |
| stop suffix 处理 | append token 后 decode 全部 generated text，`suffix in text` 即停；输出包含触发 stop suffix 的 token。 | 用 token bytes 做 UTF-8 安全缓冲；命中 stop suffix 后只输出 stop 之前的文本，stop suffix 本身不进入 output text/token_ids。 | 对 JSON fence、`\nUser:` 等 stop suffix，当前输出更干净，但和旧版 completion 字面内容不同。 |
| `_matches_stop_suffix` | 旧版存在，文本级包含匹配。 | 当前删除，由 `_find_stop_suffix` 替代。 | 当前字节级更稳。 |
| no-penalty token | sampler 更新 penalty 后，旧版立即把 `no_penalty_ids` 清零。 | 当前在采样前对 subset 清零，采样后不立即清零，但下次采样前会再清零。 | 对下一次 sampling 分布基本等价；内部 penalty tensor 瞬时值不同。若未来中间读取 penalties，会有差异。 |
| penalties sampler | 总是调用 repetition sampler。 | 若 penalty 全为 0，走无 repetition sampler。 | 默认配置 penalty 非 0，不影响默认 benchmark。 |
| `_infer_vocab_size` | 相同。 | 相同。 | 无差异。 |
| `_infer_device` | 相同。 | 相同。 | 无差异。 |
| `_prepare_state_container` | 相同。 | 相同。 | 无差异。 |
| `_decode_tokens` | 反复裁掉末尾 token，直到可 decode。 | 相同。 | 无差异。 |
| `_decode_token_bytes` | 不存在。 | 当前新增，优先用 tokenizer `decodeBytes/decode_bytes`。 | 支撑 UTF-8 streaming、stop suffix bytes、logprob token bytes。 |
| `_normalize_stop_suffixes` | 返回 `tuple[str, ...]`。 | 返回 `tuple[(suffix, suffix_bytes), ...]` 和最大 bytes 长度。 | 当前字节级 stop。 |
| `_record_generated_text_delta` | 不存在。 | 当前新增，更新 emitted text/token events，并调用 `on_token`。 | 支持 streaming。 |
| `_push_generated_token_text` | 不存在。 | 当前新增，处理 UTF-8 完整 token prefix。 | 防止半个 UTF-8 字符被提前输出。 |
| `_push_stop_tokens` | 不存在。 | 当前新增，维护 stop suffix 延迟窗口。 | 防止 stop suffix 被输出。 |
| `_finish_generated_text` | 不存在。 | 当前新增，收尾 pending bytes。 | stop/max/constraint 结束时 flush 文本。 |
| `_take_stop_output` | 不存在。 | 当前新增，按 bytes emit token/text。 | 支持 stop suffix 截断。 |
| `_token_bytes` | 不存在。 | 当前新增。 | token event 到 bytes。 |
| `_collect_token_bytes` | 不存在。 | 当前新增。 | stop suffix bytes 匹配。 |
| `_longest_valid_utf8_token_prefix` | 不存在。 | 当前新增。 | streaming 稳定性。 |
| `_longest_valid_utf8_prefix_len` | 不存在。 | 当前新增。 | partial bytes 输出保护。 |
| `_find_stop_suffix` | 不存在。 | 当前新增，查找最早 stop bytes。 | 替代旧 `_matches_stop_suffix`。 |
| `_build_generated_token` | 不存在。 | 当前新增，构造 token text/bytes/logprob/top_logprobs。 | 支持 OpenAI-compatible logprobs/streaming。 |
| `_build_generated_token_candidate` | 不存在。 | 当前新增。 | 支持 top logprobs。 |

需要重点注意的行为差异：

- **prompt strip 差异**：旧版默认 `strip()`，当前生成不 strip。  
- **prefill 差异**：旧版逐 token，当前默认 16 token chunk。  
- **stop suffix 输出差异**：旧版 completion 可能包含 stop suffix，当前默认截掉。  
- **completion event 差异**：当前 `GenerationOutput.tokens` 可能有 token/logprob 信息，旧版没有。

### `src/infer/sampling.py`

| 函数/类 | 旧项目 | 当前项目 | 影响 |
| --- | --- | --- | --- |
| `SamplingConfig` 字段 | `max_generate_tokens`、`temperature`、`top_k`、`top_p`、`alpha_presence`、`alpha_frequency`、`alpha_decay`、`stop_tokens`、`ban_tokens`、`pad_zero`、`no_penalty_token_ids`。 | 字段相同。 | 默认采样配置一致。 |
| `SamplingConfig.clamp` | 限制最大生成 token。 | 相同。 | 无差异。 |
| `SamplingConfig.max_new_tokens` | 无。 | 新增 alias。 | API 兼容层。 |
| `presence_penalty/repetition_penalty/penalty_decay` | 无。 | 新增 OpenAI 命名 alias。 | API 兼容层。 |
| `penalties_enabled` | 无。 | 判断是否启用 penalty sampler。 | penalty 全 0 时当前会走更简单 sampler。 |
| `checked` | 无。 | 修正 temperature/top_k/top_p/max_generate_tokens。 | 当前非法参数更稳，但不完全复刻旧错误行为。 |
| `GenerationOutput` | `prompt_index/prompt/token_ids/text/finish_reason`。 | 额外 `tokens: list[GeneratedToken]`。 | 当前可携带 token-level metadata。 |
| `GeneratedTokenCandidate` | 无。 | 新增。 | top-logprobs 输出。 |
| `GeneratedToken` | 无。 | 新增。 | token id/text/bytes/logprob。 |
| `GeneratedTextDelta` | 无。 | 新增。 | streaming delta。 |

### `src/infer/model.py`

| 函数/类 | 旧项目 | 当前项目 | 影响 |
| --- | --- | --- | --- |
| `ArchVersion` | 有，只有 `RWKV7`。 | 删除。 | 配置/展示元数据差异，加载无影响。 |
| `DataVersion` | 有 `g0...g1f`。 | 删除。 | 配置/展示元数据差异。 |
| `ParamSize` | 有 `0_1b...13_3b`。 | 删除。 | 配置/展示元数据差异。 |
| `ModelLoadConfig` | `weights_path/device/tokenizer_path/arch_version/data_version/num_params`。 | 只保留 `weights_path/device/tokenizer_path`。 | 若旧配置代码依赖 enum 字段，需要迁移。 |
| `load_rwkv_model` | 展开 `.pth`、加载默认 vocab、构造 `RWKV_x070` 和 `TRIE_TOKENIZER`。 | 相同。 | 核心加载无差异。 |

### `src/infer/rwkv7/utils.py`

| 函数/类 | 差异 | 影响 |
| --- | --- | --- |
| `TRIE` | 无语义差异。 | 无。 |
| `TRIE_TOKENIZER.encodeBytes` | 当前增加类型标注。 | 无。 |
| `TRIE_TOKENIZER.decodeBytes` | 当前接受 `Sequence[int]`，内部别名为 `tokens`。 | 语义一致，但更适配 current engine bytes decode。 |
| `TRIE_TOKENIZER.encode` | 当前增加类型标注。 | 无。 |
| `TRIE_TOKENIZER.decode` | 当前增加类型标注。 | 无。 |

### `src/infer/rwkv7/rwkv7.py`

该文件两个项目完全一致。函数/类实现如下：

| 函数/类 | 作用 | 一致性 |
| --- | --- | --- |
| `SPMV.forward` | 调用 `torch.ops.rwkv7_state_fwd_fp16.spmv_forward`。 | 一致。 |
| `SPMV_OP` | custom op 包装 sparse vector/matrix multiply；`mutates_args=()`。 | 一致。 |
| `WKV_7_ONE.forward` | 单 batch 单 token RWKV7 recurrent state op。 | 一致。 |
| `RWKV7_ONE_OP` | custom op 包装 `forward_one(1, ...)`；`mutates_args=("state",)`。 | 一致。 |
| `WKV_7_SEQ.forward` | 单 batch 多 token sequence op。 | 一致。 |
| `RWKV7_SEQ_OP` | custom op 包装 `forward_seq(1, T, ...)`；`mutates_args=("state",)`。 | 一致。 |
| `WKV_7_BATCH.forward` | 多 batch 单 token op。 | 一致。 |
| `RWKV7_ONE_BATCH_OP` | custom op 包装 `forward_one(B, ...)`；`mutates_args=("state",)`。 | 一致。 |
| `WKV_7_SEQ_BATCH.forward` | 多 batch 多 token sequence op。 | 一致。 |
| `RWKV7_BATCH_OP` | custom op 包装 `forward_seq(B, T, ...)`；`mutates_args=("state",)`。 | 一致。 |
| `RWKV_x070.__init__` | 加载权重、转 dtype/device、推断层数、准备参数。 | 一致。 |
| `generate_zero_state` | 生成 `state[0]`、`state[1]`、`state[2]`。 | 一致。 |
| `forward` | 单样本入口，按长度走 one/seq。 | 一致。 |
| `forward_batch` | 多样本入口，按 token list 情况走 same-length/seq-batch。 | 一致。 |
| `forward_batch_same_length` | 同长度 batch 快路径。 | 一致。 |
| `forward_one` | 单 token transformer layer loop。 | 一致。 |
| `forward_seq` | 单样本 sequence loop。 | 一致。 |
| `forward_seq_batch` | 多样本 sequence loop。 | 一致。 |
| `RWKV_x070_TMix_one/seq/seq_batch` | time-mix 计算，调用 RWKV7 custom op。 | 一致。 |
| `RWKV_x070_CMix_one/seq/seq_batch` | channel-mix 计算。 | 一致。 |

说明：之前需要警惕的 `mutates_args=()` state mutation 元数据问题，在当前两个目录里都已经是 `mutates_args=("state",)`，不是当前这次两个项目之间的差异。

## CUDA/HIP/rapid-sampling 算子级对比

所有下面列出的算子在两个项目中实现一致。

### RWKV7 state forward extension

注册位置：

- CUDA：`src/infer/rwkv7/cuda/rwkv7_state_fwd_fp16.cpp`
- HIP：`src/infer/rwkv7/hip/rwkv7_state_fwd_fp16_op.hip`

| Torch op | C++ wrapper | CUDA/HIP wrapper | kernel | 作用 | 一致性 |
| --- | --- | --- | --- | --- | --- |
| `rwkv7_state_fwd_fp16.forward_one` | `forward_one` | `cuda_forward_one` | `kernel_forward_w0_fp16_dither_one` | 多 batch 单 token recurrent update；读写 `state`；用 `elapsed_t` 做 deterministic dithering。 | 一致。 |
| `rwkv7_state_fwd_fp16.forward_seq` | `forward_seq` | `cuda_forward_seq` | `kernel_forward_w0_fp16_dither_seq` | 多 batch 多 token recurrent update；读写 `state`；sequence 内按 `_t` 推进。 | 一致。 |
| `rwkv7_state_fwd_fp16.spmv_forward` | `spmv_forward` | `cuda_spmv_forward` | `spvecmatmul_noindices` | 稀疏 vector x matrix，用非零 half 元素和 `atomicAdd(half2)` 聚合。 | 一致。 |

实现要点：

- `kernel_forward_w0_fp16_dither_one/seq` 使用 `_N_=64` head size、`half2` shared memory state tile。
- `w` 的衰减项包含 `rotator1(_elapsed_t[bbb] + _t)` deterministic dithering。
- state 会被原地写回 `_state`。
- Python custom op wrapper 已标记 `mutates_args=("state",)`。

### rapid-sampling extension

注册位置：

- CUDA wrapper：`src/infer/rapid_sampling/sampling.cpp`
- CUDA kernel：`src/infer/rapid_sampling/sampling.cu`
- HIP wrapper/kernel：`src/infer/rapid_sampling/hip/sampling_op.hip`、`sampling.hip`

| Python exposed op | kernel/wrapper | 作用 | 一致性 |
| --- | --- | --- | --- |
| `setup_rand(seed, B)` | `setup_rand_kernel` | 为每个 batch row 初始化 `curandStatePhilox4_32_10_t`。 | 一致。 |
| `batch_sampling_repetition_temperature_topk_topp` | `batch_sampling_repetition_temperature_topk_topp_kernel` | logits - penalties 后 temperature softmax，top-k/top-p 阈值采样，更新 penalties。 | 一致。 |
| `batch_sampling_temperature_topk_topp` | `batch_sampling_temperature_topk_topp_kernel` 或 `batch_sampling_topp_kernel` | 无 repetition penalty 的 temperature/top-k/top-p 采样；当 `temperature == 1 && top_k == V` 时走 topp 专用 kernel。 | 一致。 |

实现要点：

- logits 必须 FP32，vocab size 必须 `V % 4 == 0` 且 `V <= 1048576`。
- top-p/top-k 通过概率阈值二分和 block-level scan 处理。
- repetition kernel 会在采样后按 sampled token 更新 `penalties`。
- 两个项目 kernel 文件完全一致；采样差异只可能来自 Python 调用参数或调用顺序。

## 当前项目新增推理层

这些文件旧项目没有，对“两个项目推理一致”构成额外变量：

| 当前文件 | 作用 | 与旧版关系 |
| --- | --- | --- |
| `src/infer/backend.py` | 统一 local/remote backend；支持 OpenAI-compatible remote `/v1/completions`；local 可选 classic/lightning。 | 旧版 evaluator 直接加载本地模型和 `InferenceEngine`。 |
| `src/infer/lightning_engine.py` | 新增 lightning engine adapter，支持 prefix state cache。 | 旧版没有。需要单独 A/B，不应默认认为和旧 classic 一致。 |
| `src/infer/state_pool.py` | L1/L2/sqlite prefix/session state cache。 | 旧版没有。若跨模型复用 DB，会引入隐性 state 污染风险。 |
| `src/infer/constraints/*` | 字符串/JSON/function-call 约束解码。 | 旧版没有。remote backend 不支持 constraints。 |
| `src/infer/api.py`、`server.py`、`service.py`、`openai_service.py`、`sse.py` | OpenAI-compatible HTTP API、batch worker、SSE streaming、chat/tool prompt 渲染。 | 旧版没有。 |

当前-only 风险：

- `lightning_engine` 依赖 `state_pool` prefix cache。若同一 `rwkv_sessions.db` 被不同模型或不同 engine schema 复用，可能静默命中旧 state。需要为每个模型指定独立 `--state-db-path`，或关闭 lightning 用 classic 对齐旧版。
- `lightning_engine` 有自己的 sampling rows 调度；若要排查模型退化，应先用 `--engine-mode classic`，再单独验证 lightning。
- remote backend 的 `GenerationOutput.token_ids` 为空，constraints 不支持，`top_logprobs` 不由 eval client 直接控制；多选 logits scoring 依赖服务端返回 `candidate_token_texts` 的 logprobs。

## 提示词差异总览

### 旧项目：提示词分散

旧项目主要 prompt 来源：

- 多选：`src/eval/evaluators/multi_choice.py`
- 自由问答/数学：`src/eval/evaluators/free_response.py`
- 指令跟随：`src/eval/evaluators/instruction_following.py`
- 代码：`src/eval/evaluators/coding.py` 和 `src/eval/evaluators/coding_prompts.py`
- 简单 function-calling：`src/eval/function_calling/rwkv_prompt.py`、`simple_tool_call.py`

### 当前项目：统一 expected-context

当前项目主要 prompt 来源：

- 通用 builder：`src/eval/prompt_builders.py`
- 多选 pipeline：`src/eval/knowledge/pipeline.py`
- 数学 pipeline：`src/eval/maths/pipeline.py`
- 指令跟随 pipeline：`src/eval/instruction_following/pipeline.py`
- 代码 pipeline：`src/eval/coding/pipeline.py`
- function-calling：`src/eval/function_calling/*.py`

当前统一的核心模式是：

```text
User: ...

Assistant: ...
```

并在 expected context 中使用 placeholder：

- `<|completions_of_cot|>`
- `<|logprobs_of_choices|>`
- `<|final_answer|>`
- `<|completions|>`

pipeline 根据任务阶段用 `prompt_for_cot` 或 `prompt_for_marker` 截取到 placeholder 前作为真实 prompt。

## Benchmark prompt 逐类对比

### 多选题 / MMLU / MMLU-Pro

旧英文 direct：

```text
User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
<CHOICES>

Assistant:The answer is
```

旧英文 CoT：

```text
User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
<CHOICES>

Assistant:<think
```

旧 final：

```text
<Q><COT>
Therefore, the answer is
```

当前 expected context：

```text
User: You are a very talented expert in {subject}.
Answer this question and finish with a single option letter.
Question: {question}
Choices:
A. ...
B. ...

Assistant: <think><|completions_of_cot|></think>
Therefore, the answer is<|logprobs_of_choices|>
```

当前 no-cot 截到：

```text
User: ...

Assistant: Therefore, the answer is
```

差异：

- 旧版 `Assistant:The answer is` 无空格；当前 `Assistant: Therefore, the answer is` 有空格且 wording 不同。
- 旧版题目直接放 `<Q>`，当前加 `Question:` 和 `Choices:` label。
- 旧版 choices 由 `_format_prompt` 注入；当前统一 `A. choice` 格式。
- 旧版 CoT prompt 以 `Assistant:<think` 结尾；当前 CoT prompt 以 `Assistant: <think>` 结尾。
- 当前支持 `fake_cot` expected context，旧版没有统一 fake-cot 模式。

对 MMLU-Pro 的影响：如果要复现旧版分数，应使用旧模板；如果要跟当前 field-oriented/rwkv-rs 风格一致，应使用当前 `multi_choice_plain`/`multi_choice_cot`。

### 数学 / Free-response

旧默认 CoT：

```text
User: <Q>

Assistant:<think
```

旧默认 final：

```text
<Q><COT>
Therefore, the answer is \(\boxed{
```

当前 `maths/pipeline.py` legacy default：

```text
User: <Q>

Assistant: <think>
```

final：

```text
<Q><COT></think>
Therefore, the answer is \(\boxed{
```

当前 canonical expected context：

```text
User: You are a very talented expert in {subject}.
Solve the problem and output the final answer in \boxed{}.
Problem: {question}

Assistant: <think><|completions_of_cot|></think>
Therefore, the answer is \(\boxed{<|final_answer|>}\).
```

差异：

- 旧版 CoT 起始是 `Assistant:<think`，当前是 `Assistant: <think>`。
- 旧版 final 不自动补 `</think>`，当前默认补。
- 当前 canonical prompt 增加 subject、Problem label、boxed 格式说明和句号。
- 当前仍保留 legacy template 分支：只要传入非默认 `cot_prompt_template/final_answer_template`，pipeline 会走旧式替换逻辑。

### 指令跟随

旧项目：

```text
User: {prompt}

Assistant:{suffix}
```

其中 `enable_think=True` 时 `suffix="<think"`。

当前项目：

```text
User: {prompt}

Assistant:{suffix}
```

其中 `enable_think=True` 时 `suffix=" <think"`。

差异：

- 非 think 模式基本一致。
- think 模式当前多了一个空格：`Assistant: <think` vs `Assistant:<think`。

### HumanEval / MBPP

旧项目 HumanEval 主 prompt：

````text
User:You are a top-level code master. Complete the following code without any additional text or explanation:
{clean_code}

Assistant: <think>
</think>
```python
````

旧 no-echo/bugfix prompt：

````text
User: You are a top-level code master. Complete the following code without any additional text or explanation:
{clean_code}

Assistant:<think></think>
```python
````

当前 HumanEval expected context：

````text
User: You are a top-level code master.
{prompt}
Complete the code without any additional text or explanation:

Assistant: ```python
{assistant_code_prefix}<|completions|>
````

当前 `CodingPipeline._build_human_eval_prompt` 默认对非 `human_eval_fix` 使用 `assistant_code_prefix=prompt`，也就是 assistant code block 中会 echo 原 prompt。

差异：

- 旧版通常带空 think block；当前 no-cot 不带 `<think></think>`。
- 旧版 HumanEval 主 prompt 的 `User:You` 少一个空格；当前统一 `User: ...`。
- 当前 HumanEval 默认在 assistant code fence 后 echo prompt；旧版 no-echo 路径不 echo。
- MBPP 当前支持 `cot_mode`，旧版主要是 no-cot 代码块 prompt。

如果旧版代码生成分数更好，优先回放旧 prompt，特别是空 think block 和 no-echo/echo 行为。

### LiveCodeBench

旧项目 CoT prompt：

````text
User: You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.
### Question:
{question}

### Format: ...
```python
{starter_or_placeholder}
```

### Answer: (use the provided format with backticks)

Assistant:<think
````

旧 final：

````text
{cot_prompt}{cot_completion}
</think>
```python
````

当前 expected context：

````text
User: You are an expert Python programmer.
Solve the following programming problem and output only the final code.
Problem:
{question}
Use the following starter code and complete it into a full solution:
```python
{starter_code}
```

Assistant: <think><|completions_of_cot|></think>
```python
{starter_code}
<|completions|>
````

差异：

- 旧版用 `### Question` / `### Format` / `### Answer` 结构，当前用更短的 `Problem:` 和 starter-code 描述。
- 旧版 CoT 起始 `Assistant:<think`；当前 `Assistant: <think>`。
- 当前 final code prompt 会把 starter code 放入 assistant code prefix；旧版 final 只打开 code block。

### Function-calling / simple BFCL + ToolAlpaca

简单 function-calling 核心 prompt 在两个项目中一致：

````text
System: Tools:
[
  ...
]
Return only a JSON function call.
The JSON shape is {"name":"tool_name","arguments":{...}}.
If multiple tool calls are required, return a JSON array of those objects in execution order.
Use only listed tool names.

User: {instruction}

Assistant: ```json
````

共同 builder：

- `rwkv_prompt.assistant_json_prefix`
- `rwkv_prompt.build_rwkv_json_call_prompt`
- `simple_tool_call.build_simple_tool_call_prompt`
- `extract_json_call_value_text`
- `coerce_json_function_call_payload(s)`

差异：

- 当前 `simple_tool_call.py` 增加了 `_run_simple_tool_call`，直接接入 scheduler/DB/partial flush；旧版只有数据规范化、prompt 构造、decode/evaluate 这类核心函数。
- 当前 function-calling 目录新增 BFCL v3、tau、MCP、BrowseComp 等复杂 agent prompt；旧版没有这些完整模块。

### 当前新增 BFCL v3 / tau / MCP prompt

当前 BFCL v3 支持两种风格：

- `rwkv_official_json`：直接用 `System/User/Assistant` 加 JSON fence 的 prompt，输出 JSON function call。
- `staged_cot_router`：先 CoT，再 router 输出 `TOOL/ASK/HANDOFF`，再 tool/ask/handoff prompt。

关键 BFCL v3 prompt 函数：

| 函数 | 作用 |
| --- | --- |
| `build_bfcl_system_block` | `System: ...`。 |
| `build_bfcl_user_block` | `User: Request...`，可附 previous tool result 和 state snapshot。 |
| `build_bfcl_rwkv_prompt` | 调 `build_rwkv_json_call_prompt` 生成 official JSON call prompt。 |
| `build_bfcl_cot_prompt` | `System + User + Assistant: <think>`。 |
| `build_bfcl_router_prompt` | 要求只输出 `TOOL/ASK/HANDOFF`。 |
| `build_bfcl_tool_prompt` | 要求输出 JSON tool call。 |
| `build_bfcl_ask_prompt` | 要求输出 clarification question。 |
| `build_bfcl_handoff_prompt` | 要求输出 plain-language handoff/final response。 |
| `build_bfcl_system_prompt` | 渲染 Tools、JSON shape、ask_user/final_answer 规则。 |

当前 tau prompt：

- `build_tau_system_prompt` 渲染 assistant/user tools，追加 `final_answer`，要求 `Return only a JSON function call`。
- `build_expected_context` 用 `Assistant: <think><|completions_of_cot|>`。
- `build_turn_completion_prompt` 在 CoT 后追加 `</think>\nReturn only a JSON function call.\n`。

当前 MCP prompt：

- `build_planning_context` 用 `System/User/Assistant: <think><|completions_of_cot|>`。
- `build_planning_json_call_prompt` 可走 official JSON function call 风格。
- tool name 格式为 `server:tool`。

这些 current-only prompt 没有旧版直接等价物。

## 需要对齐的实际改动建议

如果目标是“当前项目行为尽量复刻旧项目 classic 本地推理”：

1. 本地推理使用 `--engine-mode classic`，不要用 `lightning` 做一致性基线。
2. 对需要精确复现的任务，把 `prefill_chunk_size` 设为 `1`。
3. 明确是否恢复旧版 prompt `strip()`。当前生成路径不 strip；旧版默认 strip。可以在 caller 侧传 `prompt.strip()`，或重新加兼容参数。
4. 对 stop suffix 字面输出做兼容判断。当前会截掉 stop suffix；旧版可能把 stop suffix 留在 completion 中。
5. 多选/数学/代码 benchmark 如要复现旧分数，应优先复制旧模板，而不是只看模型/采样。
6. 如果使用 current-only `lightning`，为每个模型设置独立 `--state-db-path`，并先清理旧 sqlite cache。
7. function-calling simple prompt 无需迁移，核心已一致；只需确认当前 configs 是否选择同一个 `prompt_style` 和 dataset。

## 建议验证矩阵

为了把“引擎差异”和“prompt 差异”拆开，建议按下面顺序做 A/B：

| 组 | 当前设置 | 目的 |
| --- | --- | --- |
| A | current classic + current prompt + `prefill_chunk_size=16` | 当前默认表现。 |
| B | current classic + current prompt + `prefill_chunk_size=1` | 隔离 chunked prefill 差异。 |
| C | current classic + old prompt + `prefill_chunk_size=1` + prompt strip | 最接近旧 classic。 |
| D | old classic + old prompt | 旧基线。 |
| E | current lightning + current prompt + fresh per-model state DB | 单独验证 lightning。 |

如果 C 和 D 接近，而 A/B 偏差大，主要是 prompt 或 prefill 调度差异。  
如果 B/C 与 D 仍明显偏差，再查 stop suffix 截断、sampling 参数、remote 服务端实现。

## 当前验证状态

已完成：

- 文件级 diff。
- 推理相关核心文件 hash 对比。
- 函数级静态对比。
- CUDA/HIP/rapid-sampling 算子注册与 kernel 对照。
- benchmark prompt 静态对比。

未完成：

- 未做 GPU 上逐 token A/B。
- 未跑完整 benchmark。
- 未修改代码。
