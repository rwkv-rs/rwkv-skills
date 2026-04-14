# 08 Space 可视化模块审阅（`src/space`）

## 高优先级问题

### SPC-1. UI 逻辑高度集中在单文件，维护成本过高
- 位置：`src/space/app.py`（总长约 1411 行），核心构建函数 `src/space/app.py:1188-1402`
- 问题：数据准备、筛选、图表、导出、事件绑定都堆在一个模块。
- 影响：
  - 任意改动都容易触发连锁回归。
  - 很难做单元测试与局部替换（例如将来换成后端 API 拉取）。
- 建议：拆为 `view_model.py` / `charts.py` / `tables.py` / `io.py` / `app.py`。

### SPC-2. 每次交互都全量重读分数索引，扩展性仍然一般
- 状态更新（2026-04-03）：**原先“rglob 全盘扫分数目录”问题已修复**
- 位置：
  - `src/space/app.py`
  - `src/space/data.py`
- 当前行为：`load_scores()` 已改为只读取 `score_index.jsonl`，不再扫 DB latest-score 视图或结果目录。
- 当前问题：refresh 仍会重读整个 index 文件，还没有增量缓存或内存快照。
- 影响：结果文件规模稍大时，UI 切换模型会明显卡顿。
- 建议：
  - 增加增量缓存（mtime/index）。
  - refresh 才强制全量重建，普通筛选只在内存数据上操作。

### SPC-3. CSV 导出不断创建临时目录且不回收
- 位置：`src/space/app.py:1134-1137`
- 问题：每次导出调用 `tempfile.mkdtemp(...)`，没有回收策略。
- 影响：长时间运行 Space 会累积大量临时文件。
- 建议：改为固定导出目录 + 覆盖写，或在会话结束时清理。

## 中优先级问题

### SPC-4. 分数根目录在 import 阶段固定，运行期间配置变更不会生效
- 位置：`src/space/data.py:112`
- 问题：`SPACE_SCORES_ROOT = _resolve_scores_root()` 在模块导入时计算一次。
- 影响：
  - 动态切换环境变量/挂载目录后，UI 仍指向旧目录。
  - 诊断“为什么读取不到最新结果”时容易困惑。
- 建议：在每次 refresh 时重算 root，或允许 UI 显式选择根目录。

### SPC-5. Coding 示例只读日志首行，展示代表性不足
- 位置：`src/space/app.py:1093-1095`
- 问题：仅读取 `readline()`，并且 JSON 解析失败后静默跳过（`src/space/app.py:1099-1102`）。
- 影响：示例区域经常不能代表实际最优/最新样本。
- 建议：按规则选样（例如最新一条通过样本）并在失败时给出显式提示。

### SPC-6. 多处异常采用“吞掉+print”，缺乏可观测性
- 位置：
  - `src/space/data.py:319-324`
  - `src/space/data.py:339-343`
  - `src/space/app.py:370-373`
- 建议：统一日志接口并给出 error code，便于线上排障。

## 低优先级问题

### SPC-7. 领域/标签映射硬编码在代码中，扩展数据集需要改代码
- 位置：`src/space/app.py:47-78`, `src/space/app.py:133-167`
- 建议：把域映射迁移到配置文件（例如 `configs/space_domains.toml`）。

### SPC-8. `data.py` 与 `app.py` 的数据契约未强约束
- 位置：`src/space/data.py:220-238`, `src/space/app.py:169-177`
- 建议：引入稳定的 `ViewModel` dataclass，减少隐式字段依赖。

## 模块重构建议

1. **后端先行**：这一条已完成第一步，Space 主分数已改读统一 `score_index`。
   剩余问题是把 index 的维护职责进一步收敛到 orchestrator。
2. **缓存层**：`load_scores` 改为增量索引 + 内存快照。
3. **组件化 UI**：按“摘要/表格/图表/导出”拆函数与文件，降低变更风险。
4. **可观测性**：添加 `space_metrics`（扫描时延、加载文件数、解析失败数）。
