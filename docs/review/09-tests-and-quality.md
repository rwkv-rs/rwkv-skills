# 09 测试与质量保障审阅（`tests`, CI 可测性, 静态正确性）

## 高优先级问题（当前就会阻断质量闭环）

### TQ-1. 测试在收集阶段即失败，CI 不能提供有效反馈
- 证据（已复现）：`pytest -q` 直接在 collection 报错。
- 具体位置：
  - `tests/test_db_integration.py:5` 依赖已不存在的 `DatabaseManager`。
  - `src/db/database.py:1-12` 当前只导出 `init_db/is_initialized`。
  - `src/eval/metrics/instruction_following/instructions_util.py:27-35` import 阶段触发 NLTK 下载。
- 影响：
  - 大部分测试根本没执行。
  - 开发者容易误判“代码没问题，只是环境问题”。
- 建议：
  - 先恢复“可收集”状态：移除过时代码路径、取消 import side-effect。
  - 把资源下载迁移到显式 setup 步骤。

### TQ-2. 数据库正确性关键路径缺少回归测试
- 已知高风险点：
  - `src/db/eval_db_repo.py:417-426` 非法索引回退为 0。
  - `src/db/eval_db_repo.py:476-479` `insert_eval` 宽条件 update。
- 现状：没有对应的 repo/service 级回归用例。
- 影响：后续重构很容易再次引入“分数统计错误/覆盖写入”问题。
- 建议：为每个高风险 bug 建“先失败后修复”的回归测试模板。

### TQ-3. 指标正确性缺少“重复样本/续跑”场景覆盖
- 已知问题：`src/eval/metrics/at_k.py:27-30`, `src/eval/metrics/at_k.py:54-56`。
- 现有测试：`tests/test_free_response_metrics.py:8-29` 仅覆盖理想输入。
- 影响：最关键的线上场景（resume/retry）未被验证。
- 建议：新增重复键、脏数据、断点续跑组合用例，并要求与 DB 写入契约联测。

## 中优先级问题

### TQ-4. 测试体系风格混用，夹杂历史 `unittest` 与 `pytest` 习惯
- 位置：`tests/test_db_integration.py`、`tests/test_space_data.py` 等。
- 问题：fixture/monkeypatch 复用率低，维护成本高。
- 建议：统一迁移为 pytest 风格，沉淀公共 fixture（DB session、临时数据集、fake writer）。

### TQ-5. 新增用例虽覆盖部分调度语义，但整体 E2E 仍缺失
- 已有改进：
  - `tests/test_resume_context_status_filter.py`
  - `tests/test_scheduler_overwrite_semantics.py`
- 缺口：没有“从任务创建到分数入库再到 Space 展示”的端到端可重复用例。
- 建议：增加最小 E2E 场景（本地 sqlite/临时目录）作为发布前门禁。

### TQ-6. `param_search` 暴露符号错误未被任何测试捕获
- 位置：`src/eval/param_search/cot_grid.py:121`
- 问题：`__all__` 包含不存在的 `total_grid_size`。
- 影响：`from ... import *` 时会触发 `AttributeError`，属于静态正确性错误。
- 建议：补充“模块导出契约测试”（检查 `__all__` 内符号均存在）。

### TQ-7. Space/UI 关键路径测试不足
- 现状：仅 `tests/test_space_data.py` 覆盖数据归一化；`src/space/app.py` 核心交互逻辑基本无测试。
- 建议：把 `app.py` 中纯函数部分拆出来并单测（selection/pivot/chart-data builders）。

## 低优先级问题

### TQ-8. 缺少统一质量门禁（lint/type/security）
- 现状：仓库未体现稳定执行的 lint/type/safety CI 流程。
- 建议：增加 `ruff + mypy/pyright + pytest` 的最小流水线，并把“collection 必须成功”设为第一门槛。

## 建议的测试修复优先级

1. **先修收集失败**：`test_db_integration` 迁移 + NLTK import side-effect 去除。
2. **补正确性回归**：DB upsert、`pass@k/avg@k` 去重、generator 二次消费。
3. **建最小 E2E**：主流程跑通并断言最终 score 与 task 状态。
4. **加静态契约测试**：`__all__`、配置 schema、结果 payload schema。
