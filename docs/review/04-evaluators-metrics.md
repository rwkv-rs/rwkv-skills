# 04 评估器与指标模块审阅（`src/eval/evaluators`, `src/eval/metrics`）

## 高优先级问题（会导致结果错误）

### EM-1. `pass@k/avg@k` 未对重复 `(sample_index, repeat_index)` 去重
- 位置：`src/eval/metrics/at_k.py:27-30`, `src/eval/metrics/at_k.py:54-56`
- 问题：直接按 sample 聚合布尔值，重复样本会被重复计入。
- 影响：在断点续跑/重复写入场景会放大分数（尤其 pass@k）。
- 建议：先按 `(sample_index, repeat_index)` 去重（保留最新或最可信记录）再聚合。

### EM-2. LiveCodeBench 评估函数会二次消费输入迭代器
- 位置：`src/eval/metrics/code_generation/livecodebench/evaluation.py:170`, `src/eval/metrics/code_generation/livecodebench/evaluation.py:233`
- 问题：同一个 `completions` 源被遍历两次；若传入 generator，第二次为空。
- 影响：`eval_payloads` 可能为空或不完整，导致入库不一致。
- 建议：开头物化为 list，再复用。

### EM-3. 指令遵循模块在 import 阶段触发 NLTK 下载
- 位置：`src/eval/metrics/instruction_following/instructions_util.py:27-35`
- 问题：模块导入时下载资源，测试/离线环境会直接失败。
- 影响：`pytest` 收集阶段就报错（已复现）。
- 建议：改为惰性初始化 + 可配置 data dir + 失败降级。

## 中优先级问题

### EM-4. 评估 pipeline 入口参数过多，函数可维护性差
- 位置：
  - `src/eval/evaluators/free_response.py:39`
  - `src/eval/evaluators/multi_choice.py:145`
  - `src/eval/evaluators/coding.py:113`, `src/eval/evaluators/coding.py:228`, `src/eval/evaluators/coding.py:349`
- 问题：大量可选参数与流程控制交织。
- 建议：引入 `RunOptions`/`GenerationPlan` 对象，简化签名。

### EM-5. coding 三套流程代码重复明显
- 位置：`src/eval/evaluators/coding.py`（HumanEval/MBPP/LiveCodeBench）
- 问题：`probe/resume/chunk/on_complete` 模式重复。
- 建议：抽象公共模板函数，保留 dataset-specific hook。

### EM-6. 多个评估器对异常样本索引采用隐式兜底
- 位置：例如 `src/eval/metrics/free_response.py:222`, `src/eval/metrics/multi_choice.py:70`
- 问题：`sample_index` 默认 0，错误数据可能被误映射到第一题。
- 建议：严格校验，非法行标记并单独统计为数据错误。

## 低优先级问题

### EM-7. 指令遵循规则库体量过大且混入大量 TODO
- 位置：`src/eval/metrics/instruction_following/instructions.py`, `src/eval/metrics/instruction_following/instructions_registry.py`
- 建议：将规则库与执行框架解耦，支持按数据集裁剪加载。

