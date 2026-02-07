# 05 CLI 入口脚本审阅（`src/bin`）

## 高优先级问题

### BIN-1. 入口脚本复制粘贴严重，已形成“多源真相”
- 证据：`src/bin/eval_*.py` 与 `src/bin/param_search_*.py` 中 task 创建、writer 生命周期、异常回滚、score 入库几乎同构。
- 典型位置：
  - `src/bin/eval_free_response.py:166-289`
  - `src/bin/eval_multi_choice.py:52-118`
  - `src/bin/eval_code_livecodebench.py:99-193`
- 影响：任何流程修复需要改 N 个脚本，易漏改。
- 建议：只保留一个 `main.py`，任务类型通过配置分发到 runner。

### BIN-2. 异常分支状态判定存在误标 completed 风险
- 位置：例如 `src/bin/eval_code_human_eval.py:131-134`, `src/bin/eval_free_response.py:229-231`
- 问题：异常时使用 `actual == expected_count` 判定 completed，未区分“历史已完成数据”与“本次运行新增数据”。
- 建议：以 `run_attempt_id` 或本次写入计数判定，不依赖全量计数。

### BIN-3. 参数体系不统一，默认值漂移
- 证据：`db-write-queue` 在不同脚本中默认值不一致（16 / 4096）。
- 位置：如 `src/bin/eval_free_response.py:54`, `src/bin/eval_multi_choice.py:33`
- 影响：性能与背压行为不一致，线上表现不可预测。
- 建议：统一到配置层集中管理。

## 中优先级问题

### BIN-4. 兼容参数和无效参数占比高
- 位置：`src/bin/eval_multi_choice_cot.py:89-92`, `src/bin/eval_instruction_following.py:81-84`
- 建议：迁移期提示后移除，避免参数面继续膨胀。

### BIN-5. 多处导入未使用
- 位置：如 `ensure_job_id` 在多个 eval 脚本被导入但未使用（`src/bin/eval_multi_choice.py:14` 等）。
- 建议：统一做静态清理，建立 lint gate。

### BIN-6. 参数网格输入直接 JSON 字符串，缺少结构校验
- 位置：`src/bin/param_search_free_response.py:95-102`
- 风险：异常输入导致运行中断或行为不可预测。
- 建议：定义显式 schema，解析失败时给出结构化报错。

## 低优先级问题

### BIN-7. 迁移脚本与运行脚本混在同一目录
- 位置：`src/bin/migrate_*`, `src/bin/backfill_*`
- 建议：归档至 `tools/legacy/`，并从生产入口剥离。

