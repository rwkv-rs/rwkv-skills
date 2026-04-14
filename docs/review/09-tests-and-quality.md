# 09 测试与质量保障审阅（`tests`, CI 可测性, 静态正确性）

## 高优先级问题（当前就会阻断质量闭环）

### TQ-1. collection 阻塞问题已解除，但文档结论需要更新
- 当前状态（2026-04-03 复核）：
  - 使用项目虚拟环境执行 `.venv/bin/pytest -q`，结果为 `133 passed`
  - `src/eval/metrics/instruction_following/instructions_util.py` 的 import 阶段 NLTK 下载副作用已去除
  - 历史 `tests/test_db_integration.py` 已不再构成 collection 阻塞项
- 当前真正的问题：
  - 不是“测试跑不起来”，而是“关键 correctness 场景仍缺少定向回归”
- 建议：
  - 继续把资源下载限制在显式 setup / lazy path
  - 把测试重点从“恢复 collection”切到“补高风险行为回归”

### TQ-2. 数据库正确性关键路径缺少回归测试
- 已知高风险点：
  - completion / eval 幂等写入语义是否持续保持唯一约束与精确插入
  - task identity / resume 语义在 `run_mode=auto|new|resume|rerun` 下是否稳定
- 现状：没有对应的 repo/service 级回归用例。
- 影响：后续重构很容易再次引入“分数统计错误/覆盖写入”问题。
- 建议：为每个高风险 bug 建“先失败后修复”的回归测试模板。

### TQ-3. 指标正确性缺少“重复样本/续跑”场景覆盖
- 已知问题：`src/eval/metrics/at_k.py:27-30`, `src/eval/metrics/at_k.py:54-56`。
- 已补回归：`LiveCodeBench` 生成器输入的二次消费问题已修，并新增专门测试覆盖。
- 现有测试缺口：仓库中仍缺少重复键、脏数据、断点续跑场景的专门指标用例。
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

### TQ-6. 静态导出契约仍缺系统性测试
- 现状：仓库还没有统一检查 `__all__`、入口函数导出、配置 schema 的静态契约测试。
- 影响：这类错误即使已在个别模块修复，后续也容易在别处再次出现。
- 建议：补充“模块导出契约测试”（检查 `__all__` 内符号均存在）。

### TQ-7. Space/UI 关键路径测试不足
- 现状：仅 `tests/test_space_data.py` 覆盖数据归一化；`src/space/app.py` 核心交互逻辑基本无测试。
- 建议：把 `app.py` 中纯函数部分拆出来并单测（selection/pivot/chart-data builders）。

## 低优先级问题

### TQ-8. 缺少统一质量门禁（lint/type/security）
- 现状：仓库未体现稳定执行的 lint/type/safety CI 流程。
- 建议：增加 `ruff + mypy/pyright + pytest` 的最小流水线，并把“collection 必须成功”设为第一门槛。

## 建议的测试修复优先级

1. **补正确性回归**：DB 幂等写入、`pass@k/avg@k` 去重、generator 二次消费。
2. **建最小 E2E**：主流程跑通并断言最终 score 与 task 状态。
3. **加静态契约测试**：`__all__`、配置 schema、结果 payload schema。
4. **继续收敛测试基建**：为 SQL/service 主链、dataset runtime、Space 消费层沉淀共用 fixture。
