# 07 推理模块审阅（`src/infer`）

## 高优先级问题（稳定性/安全性）

### INF-1. 模块导入阶段就强绑定 `flashinfer` 与环境变量副作用
- 位置：
  - `src/infer/engine.py:13-22`
  - `src/infer/engine.py:22`
- 问题：import `engine` 时直接修改环境变量并强制 `import flashinfer`。
- 影响：
  - 非 CUDA/非 flashinfer 环境无法平滑降级（导入即失败）。
  - 单测或工具脚本仅想复用数据结构也会被迫带 GPU 依赖。
- 建议：
  - 延迟导入 backend（在运行时按配置选择 `flashinfer/torch`）。
  - 导入期不改环境；把缓存目录等参数放入 `RunContext`。

### INF-2. Tokenizer 词表解析使用 `eval`，存在代码执行风险
- 位置：`src/infer/rwkv7/utils.py:115`
- 问题：词表行内容直接 `eval(...)`。
- 影响：若词表文件被污染，可触发任意代码执行。
- 建议：改为安全解析（例如 `ast.literal_eval` + 严格格式校验）。

### INF-3. 状态张量结构依赖硬编码索引，协议脆弱
- 位置：
  - `src/infer/engine.py:224-225`
  - `src/infer/engine.py:230-231`
  - `src/infer/engine.py:307-311`
- 问题：生成循环默认 `states[0]/states[1]/states[2]` 且假定固定 shape。
- 影响：
  - 一旦模型实现或 state 布局变化，运行期深处才崩溃。
  - 难以切换不同推理后端。
- 建议：引入显式 `StateAdapter`/`BackendAdapter`，在初始化阶段做结构校验并失败前置。

## 中优先级问题

### INF-4. `ModelLoadConfig` 的关键字段定义后未使用，配置可读性失真
- 位置：`src/infer/model.py:50-52`
- 问题：`arch_version/data_version/num_params` 仅定义不生效。
- 影响：用户以为这些配置会影响加载，实际无效，容易产生“配置看起来对但行为不对”。
- 建议：
  - 要么在加载逻辑里真正校验并应用。
  - 要么从该配置移除，统一由上层元数据管理。

### INF-5. `_continuous_batching` 过大且职责混杂
- 位置：`src/infer/engine.py:133-361`
- 问题：同一函数包含队列推进、state 管理、采样、惩罚、吞吐统计、回调、decode。
- 影响：难测试、难替换、难并行化。
- 建议：拆成 `TaskQueue` / `Sampler` / `StateManager` / `CompletionEmitter` 四个组件。

### INF-6. decode 异常被静默吞掉并“截断重试”，可能造成隐式内容丢失
- 位置：`src/infer/engine.py:401-406`
- 问题：decode 报错后直接删末尾 token 重试，且不记录任何异常。
- 影响：生成文本可能被截断但上层无法感知。
- 建议：记录 decode_error 计数与样本 id，并将“截断恢复”作为显式降级路径。

### INF-7. 运行时告警以 `print` 输出，缺乏统一日志与可观测性
- 位置：
  - `src/infer/engine.py:149`
  - `src/infer/engine.py:163`
  - `src/infer/engine.py:341`
- 建议：统一接入结构化日志（包含 model/dataset/task_id/prompt_index）。

## 低优先级问题

### INF-8. `assert` 被用于运行时输入约束，优化模式下会失效
- 位置：`src/infer/rwkv7/rwkv7.py:165-166`, `src/infer/rwkv7/rwkv7.py:249-250`
- 建议：改为显式异常，避免 `python -O` 下检查被跳过。

## 模块重构建议（面向多进程统一管理）

1. **后端分层**：`InferenceBackend` 协议化，`FlashinferBackend` 与 `TorchBackend` 可独立选择。
2. **进程边界清晰**：模型与 CUDA 资源只在 worker 进程内初始化，controller 进程仅下发任务。
3. **结果契约标准化**：`GenerationOutput` 补充 `decode_error`, `backend`, `latency_ms`, `token_count`。
4. **可回放**：将采样参数快照与随机种子强制写入每次 run 的元数据。
