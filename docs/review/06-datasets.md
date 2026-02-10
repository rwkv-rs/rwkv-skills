# 06 数据集模块审阅（`src/eval/datasets`）

## 高优先级问题（会影响结果可信度或重构可行性）

### DS-1. 数据缓存路径协议不统一，`output_root` 与缓存根混用
- 位置：
  - `src/eval/datasets/data_prepper/code_generation/human_eval.py:19`
  - `src/eval/datasets/data_prepper/free_answer/gsm_plus.py:39`
  - `src/eval/datasets/data_prepper/instruction_following/ifbench.py:18`
  - `src/eval/datasets/data_prepper/free_answer/mawps.py:18`
  - （同类问题分布在多个 prepper）
- 问题：大量 preparer 直接写死 `dataset_cache_dir(Path("data"), ...)`，而不是从调用方传入的运行目录/缓存策略生成。
- 影响：
  - 多进程并发准备数据时，缓存目录冲突概率高。
  - 未来切到 `main.py + 配置文件` 后，配置化的输出根与缓存根无法严格隔离。
  - 在不同工作目录运行同一任务时，行为不可预测（隐式依赖 CWD）。
- 建议：
  - 明确拆分 `data_root` / `cache_root` / `artifact_root` 三个路径，并全部来自 `RunConfig`。
  - preparer 接口从 `Callable[[Path, str], list[Path]]` 升级为接收结构化上下文（如 `DatasetPrepareContext`）。

### DS-2. split 语义与真实数据源不一致，存在“名义 test / 实际 train”风险
- 位置：
  - `src/eval/datasets/data_prepper/instruction_following/wmt24pp.py:33`
  - `src/eval/datasets/data_prepper/instruction_following/wmt24pp.py:58-59`
  - `src/eval/datasets/data_prepper/free_answer/answer_judge.py:11-12`
- 问题：对外只暴露 `split="test"`，但内部读取源数据的 `train` split。
- 影响：
  - 评测报告里 `*_test` 标签与数据事实不一致，结果可解释性差。
  - 容易引发“训练集污染评测”的争议。
- 建议：
  - 在输出 JSONL 元数据中强制写入 `source_split` 与 `source_revision`。
  - 在配置层显式声明 split 映射（例如 `target_split: test`, `source_split: train`），并在最终 score 中透传。

### DS-3. 多选 loader 对越界答案做隐式“减一修复”，会掩盖脏数据
- 位置：`src/eval/datasets/data_loader/multiple_choice.py:77-79`
- 问题：`answer_index >= len(choices)` 且 `answer_index - 1` 合法时自动减一，不报错。
- 影响：
  - 上游数据格式错误不会暴露，问题被静默吞掉。
  - 不同来源数据质量不可比，可能导致错误答案被当作合法样本。
- 建议：
  - 默认严格模式：越界直接报错并记录坏样本。
  - 如需兼容历史数据，提供显式 `compat_mode` 开关并输出修复计数。

### DS-4. Qwen 数据加载后删除原始缓存文件，破坏幂等与并发稳定性
- 位置：`src/eval/datasets/data_prepper/data_utils.py:257`
- 问题：`load_qwen_dataset` 每次读取后都会 `raw_path.unlink(...)`。
- 影响：
  - 重跑时重复下载，增加外部依赖抖动。
  - 多进程下可能出现“另一个进程正在读文件但文件被删”的竞态。
- 建议：
  - 原始文件持久缓存，配合 checksum/revision 管理。
  - 由专门清理策略（TTL 或显式命令）控制回收，而非读后即删。

## 中优先级问题

### DS-5. 解压逻辑存在路径穿越风险
- 位置：`src/eval/datasets/data_prepper/data_utils.py:135-140`
- 问题：`tar.extractall(...)` / `zip.extractall(...)` 未做成员路径校验。
- 影响：恶意归档可写出目标目录之外的文件。
- 建议：实现安全解压（校验解压后路径必须位于 destination 下）。

### DS-6. LiveCodeBench 依赖 `trust_remote_code=True` 且版本由环境变量覆盖
- 位置：
  - `src/eval/datasets/data_prepper/code_generation/livecodebench.py:54`
  - `src/eval/datasets/data_prepper/code_generation/livecodebench.py:59`
- 问题：运行时行为受环境变量与远端实现共同影响，难以复现。
- 建议：
  - 配置中固定 revision/hash。
  - 将“允许 remote code”作为显式风险开关并默认关闭。

### DS-7. MMMLU 逐 subject 调 `load_dataset`，准备过程放大网络/IO 开销
- 位置：`src/eval/datasets/data_prepper/multiple_choice/mmmlu.py:62-64`
- 问题：外层遍历 subject，内层每次重新请求一个 dataset config。
- 影响：慢、易触发 rate limit，且失败恢复成本高。
- 建议：增加本地缓存层与断点元数据，避免重复全量请求。

### DS-8. `data_prepper/common.py` 仍是占位模块，缺失真正共享逻辑
- 位置：`src/eval/datasets/data_prepper/common.py:1`
- 问题：目前大量预处理规则散落在各数据集脚本，无法共享校验/标准化。
- 建议：把字段映射、答案归一化、split 元数据注入沉淀到 `common.py`。

## 低优先级问题

### DS-9. `data_manager` 对 perf logger 采用裸兜底，可能掩盖真实导入错误
- 位置：`src/eval/datasets/data_prepper/data_manager.py:20`
- 建议：仅捕获 `ModuleNotFoundError`，其他异常要透出并打诊断日志。

### DS-10. 错误信息仍带历史命名（NeMo），语义与仓库不一致
- 位置：`src/eval/datasets/data_prepper/data_manager.py:106`
- 建议：统一为当前项目语义，避免线上排障误导。

## 模块重构建议（对齐 `main.py + 配置 + 多进程`）

1. **定义标准数据契约**：`DatasetManifest`（dataset/source/split/revision/checksum/row_count）。
2. **统一准备入口**：`DatasetService.prepare(manifest, context)`，所有 dataset prepper 仅实现纯转换。
3. **并发安全**：下载与解压引入文件锁，缓存目录按 `dataset@revision` 分层。
4. **可追溯输出**：每个输出 JSONL 旁边写 `manifest.json`，供后续评估和 UI 回放。
