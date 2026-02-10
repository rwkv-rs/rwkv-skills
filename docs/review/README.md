# rwkv-skills 全量代码审阅（模块化）

本目录记录对当前仓库（`src/`, `tests/`, `scripts/`）的分模块审阅结果，目标是找出**必须重构/应尽快修复**的问题，服务于你提出的方向：

- 废除 CLI 分裂入口
- 收敛为 `main.py + 配置文件`
- 用多进程统一调度执行与评估
- 修复数据库与计分逻辑中会导致业务结果错误的问题

## 审阅范围

- 代码规模：`src` 约 23k+ 行 Python
- 模块：`bin`, `db`, `eval`, `infer`, `space`, `tests`
- 输出方式：每个模块单独审阅文档 + 总体重构路线图

## 文档索引

1. `docs/review/01-architecture.md`：总体架构与入口层问题
2. `docs/review/02-db.md`：数据库层（高风险）
3. `docs/review/03-scheduler.md`：调度器与任务编排
4. `docs/review/04-evaluators-metrics.md`：评估流水线与计分指标
5. `docs/review/05-bin-entrypoints.md`：CLI 脚本入口层
6. `docs/review/06-datasets.md`：数据准备与加载模块
7. `docs/review/07-infer.md`：推理引擎
8. `docs/review/08-space.md`：可视化（Space）
9. `docs/review/09-tests-and-quality.md`：测试与质量保障
10. `docs/review/10-refactor-roadmap-mainpy.md`：`main.py + 配置驱动 + 多进程` 重构路线

## 重点结论（摘要）

- 当前项目的**工程复杂度主要来自“入口分裂 + 业务逻辑重复 + 隐式环境变量协议”**。
- 已识别多个会导致结果偏差或不稳定的点，尤其在：
  - completion/eval 入库幂等策略
  - `pass@k/avg@k` 的重复样本处理
  - 评估器对输入迭代器的消费语义
  - 测试在收集阶段即失败
- 现有结构适合进入“**一次架构收敛 + 分阶段迁移**”而不是继续补丁式修复。

