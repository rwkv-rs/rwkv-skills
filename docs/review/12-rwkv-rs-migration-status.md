# 12 `rwkv-rs -> rwkv-skills` 迁入状态

目标：按 `rwkv-rs -> rwkv-skills` 的方向重新收口当前主干迁移状态，避免继续用“只实现了一部分”的早期表述误判现状。

## 1. 核心结论

`rwkv-skills` 现在已经明显吸收了 `rwkv-rs` 的 evaluator 主干语义，尤其是：

- task persistence 语义
- attempt / checker / score 执行语义
- benchmark 到 job/runner 的统一注册分发语义
- task history 查询语义

当前没有对齐的，主要不是这些 evaluator 主干本身，而是：

- `rwkv-rs` 的集中式内存 scheduler 形状
- `rwkv-rs` 的集中式 HTTP / admin 服务形状
- `main.py + RunConfig + orchestrator` 这一层仍未完全做成单一主干

另外，之前被归为“未完成”的几项现在应从主缺口里拿掉：

- function-calling 已不再是“统一门面转发旧 runner”，而是单一统一 runner 内部承载 BrowseComp / MCP-Bench / tau 两类实现。
- dataset runtime 已把 `DatasetPrepareContext` 真正下传到 loader，并清掉了 `Path("data")` / `data/hf_cache` 这类硬编码入口。
- Space 主分数展示已切到 `score_index.jsonl`，不再依赖 DB 的 latest-score 聚合查询。
- instruction-following 的 NLTK 资源初始化已改成惰性执行，不再污染 import / pytest collection。

另外，`rwkv-skills` 现在已经补上 PostgreSQL-backed 的 scheduler claim/lease 协调；
这属于 Python 侧新增能力，不是从 `rwkv-rs` 直接平移过来的结构。

## 2. 已迁入的主干

### 2.1 第一阶段：task persistence 语义

`rwkv-skills` 已经吸收了 `rwkv-rs` 的 `run_mode` 语义和 TaskIdentity 查找语义：

- `src/eval/evaluating/task_persistence.py`
  - 统一提供 `auto/new/resume/rerun`
  - `new` 遇到 matching task 拒绝
  - `resume` 遇到 completed 拒绝
  - `resume` 找不到唯一 resumable task 时拒绝
- `src/db/eval_db_service.py`
  - 先确保 benchmark/model 存在
  - 再按 `config_path + evaluator + git_hash + model_id + benchmark_id + sampling_config` 查 matching task
- `src/db/sql_repo.py`
  - TaskIdentity 底层查询已经落在显式 SQL repo 层完成

resume 复用旧 task 并清 score 也已经对齐：

- `src/db/eval_db_service.py`
  - `ctx.can_resume` 时把 task 改回 `running`
  - 删除旧 `scores`

### 2.2 第二阶段：attempt / checker / finalization 语义

`rwkv-rs` 里由 scheduler 内存态维护的 AttemptKey / skip completed / failed-only checker 语义，已经被 `rwkv-skills` 吸收，只是落地方式改成了 runner-local + DB-backed：

- `src/eval/evaluating/task_persistence.py`
  - `TaskExecutionState.skip_keys` 来自 `ResumeContext.completed_keys`
- `src/eval/knowledge/runner.py`
  - runner 从 `prepare_task_execution(...)` 读取 `task_id` 与 `skip_keys`
  - 结束后统一 `eval -> checker -> score -> task completion`
- `src/eval/knowledge/pipeline.py`
  - pipeline 按 `attempt_keys` 与 `skip_keys` 跳过已完成 attempt
- `src/db/async_writer.py`
  - completion 改为异步写入
- `src/eval/evaluating/checker.py`
  - 只拉 wrong answers
  - 再减去已有 checker keys
  - 只补没做过的失败项

所以迁入的是执行语义，不是 `rwkv-rs` 的集中式 `TaskRunState + run_scheduler()` 结构本身。

### 2.3 第三阶段：benchmark / job / runner 注册分发语义

这一层已经不是早期“准备迁”的状态，而是已经在 `rwkv-skills` 里形成了 scheduler 主导的 metadata matrix：

- `src/eval/benchmark_registry.py`
  - benchmark 元数据：`field / cot_modes / default_split / scheduler_jobs / avg_ks / pass_ks`
- `src/eval/runner_registry.py`
  - runner 元数据：`module / group / scheduler_domain / extra_args / probe flags`
- `src/eval/scheduler/jobs.py`
  - 将 benchmark metadata 与 runner metadata 组合成最终 `JobSpec`

这一步迁入的是“统一注册 + 统一分发语义”，不是把 `rwkv-rs` 的 trait/factory 形状原样照搬。

### 2.4 第四阶段：task history 查询语义

`rwkv-rs` 里 HTTP 壳下面那层“历史查询”能力，已经在 `rwkv-skills` 里沉成 service API：

- `src/db/eval_db_service.py`
  - `get_task_bundle`
  - `list_completions_rows`
  - `list_eval_rows`
  - `list_checker_rows`
  - `list_scores_rows`
  - `list_eval_records_for_space`

因此已迁入的是“任务历史可查询”的底层能力，不是 `rwkv-rs` 的独立 HTTP router 外壳。

### 2.5 第五阶段：统一入口的最小配置壳

当前主干已经补上最小版统一入口：

- 仓库根 `main.py`
- `src/main.py`
- `pyproject.toml` 中的 `rwkv-skills` script

这层目前已具备：

- `RunConfig` 的最小强类型解析
- benchmark -> runner 的统一分发
- dataset auto-prepare / existing-only resolve
- `run_mode` / `job_name` 的统一环境注入

但这层目前仍是“配置驱动单次 eval shell”，还不是文档里设想的多进程 `Orchestrator`。

## 3. 明确还没有迁入的部分

### 3.1 没有迁入 `rwkv-rs` 的集中式 scheduler 形状

`rwkv-skills` 当前没有把这套结构整体搬过来：

- `TaskRunState`
- `pending_attempts`
- `task_results`
- `pending_checks`
- `checker_running`
- `run_scheduler()`

当前收口仍主要发生在各领域 runner 内部，而不是统一 scheduler 内存态。

### 3.2 尚未迁入 `rwkv-rs` 风格的集中式 HTTP / admin 服务形状

这部分需要修正文档口径：

- `rwkv-skills` **已经有** 轻量级 scheduler admin HTTP/control：
  - `DesiredState / ObservedStatus`
  - `pause / resume / cancel`
  - `control.json / runtime.json` runtime control 文件
- 但它仍**不是** `rwkv-rs` 那种集中式 router / service / openapi 组织方式

因此准确表述应当是：

- 已有一层轻量 HTTP / admin 外壳
- 但还没收敛成与 `rwkv-rs` 对等的服务形状

### 3.3 不应把“分布式层”算成 `rwkv-rs` 未迁入项

`rwkv-rs` 当前提供的是单进程内并发调度器，不是 evaluator 语义上的多机 claim / lease 框架。

因此第五阶段更准确的描述应当是：

- `rwkv-rs` 本来就没有真正分布式主干可迁
- `rwkv-skills` 现有 dispatcher 已经在 evaluator 语义之上新增了 PostgreSQL-backed claim/lease 外壳

## 4. `rwkv-skills` 当前 dispatcher 的真实形态

这层是 `rwkv-skills` 自己新增的调度包装，而不是 `rwkv-rs` 的直接平移：

- `src/eval/scheduler/state.py`
  - 完成态来自 DB 最新 score
  - 运行态来自本地 `.pid` 文件和本机 PID 存活检查
- `src/db/pool.py`
  - Python 侧已经补上 `Db { pool }` 形状，对应 `rwkv-rs` 的 `Db { pool: PgPool }`
  - 主运行链默认经由 PostgreSQL connection pool，而不是 ORM session factory
- `src/db/sql_repo.py` / `src/db/eval_db_service.py`
  - `task/completions/eval/checker/scores` 的主运行路径已经切到显式 SQL
  - `get_or_create_task`、resume、completion 写入、eval/checker 落库、score 落库都不再依赖 ORM
- `src/eval/scheduler/process.py`
  - GPU 空闲判断来自本机 `nvidia-smi`
  - 任务启动来自本地 `subprocess.Popen`
- `src/eval/scheduler/lease.py`
  - scheduler claim/lease 通过 PostgreSQL pooled DB + 显式 SQL 协调
  - 多节点靠 lease 过期 / 续租 / 释放避免重复启动
- `src/eval/scheduler/actions.py`
  - 调度循环本质是“扫 DB 完成态 + 扫本地 pid + 按资源槽位起本地子进程”
  - 本地推理模式下资源槽位来自空闲 GPU
  - 远端推理模式下资源槽位来自 `infer_base_url + infer_models + max_concurrent_jobs`
  - 分布式模式下还会把 foreign active leases 视作全局运行态，避免多个 scheduler 节点重复派发
  - 启动时注入 `job/model/dataset/run_mode`，远端模式还会注入 `infer_*` 环境变量

所以它目前是“pooled PostgreSQL + 显式 SQL 的 evaluator 主运行链 + 本地评测 worker 调度 + PostgreSQL claim/lease 协调”，并且已经支持把测评端和推理端拆开；但它仍不是 `rwkv-rs` 那种集中式内存 scheduler 状态机。

## 5. 重新测评时应采用的判断口径

如果后续重新测评 `rwkv-rs -> rwkv-skills` 的迁入程度，更准确的口径应当是：

- 已经迁入的主干有四层：
  - evaluator 持久化语义
  - attempt 执行语义
  - benchmark/job 注册分发语义
  - task history 查询语义
- 还没迁入的主干主要有三层：
  - 集中式内存 scheduler 状态机
  - `rwkv-rs` 风格的集中式 HTTP / admin 服务形状
  - `main.py + orchestrator + worker pool` 的统一主干
  - `rwkv-rs` 并不存在的多机分布式层不应误算为“待迁移主干”

这也意味着后续工作重点应该放在：

- benchmark naming / catalog 的进一步收敛
- runner 收口
- `main.py` 继续向 orchestrator 主干推进
- 是否真的需要把现有 admin 层再升级成更完整的服务壳
