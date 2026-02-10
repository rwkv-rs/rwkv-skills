# 03 调度模块审阅（`src/eval/scheduler`）

## 高优先级问题

### SCH-1. `action_dispatch` 复杂度过高，改动风险极大
- 位置：`src/eval/scheduler/actions.py:122`
- 现状：单函数约 294 行，承担调度主循环、失败恢复、日志、队列、GPU 分配、子进程启动等职责。
- 影响：
  - 回归风险大，单点故障。
  - 很难过渡到你要求的“多进程统一管理”架构。
- 建议：拆分为 `plan -> allocate -> launch -> observe -> reconcile` 五段式服务。

### SCH-2. 队列策略硬编码过多，业务策略与基础设施混在一起
- 位置：`src/eval/scheduler/queue.py:96`
- 问题：`build_queue` 同时处理模型筛选、param-search 特例、dataset 特例、running/cooldown 排除。
- 影响：新增规则易破坏已有路径。
- 建议：将策略拆成可组合规则（Rule objects）。

### SCH-3. GPU 空闲判定只看显存阈值，误判概率高
- 位置：`src/eval/scheduler/process.py:184-186`
- 问题：`list_idle_gpus` 未结合 util/load，仅按显存使用判断。
- 影响：可能把仍在高负载计算的 GPU 判为空闲。
- 建议：同时使用 utilization + mem + 可选进程白名单判断。

## 中优先级问题

### SCH-4. `scan_completed_jobs` 形参与实现语义不一致
- 位置：`src/eval/scheduler/state.py:50-56`
- 问题：函数接收 `log_dir`，但主要依赖数据库读取 latest scores。
- 影响：命名和行为偏差，后续维护容易误解。
- 建议：重命名为 DB-oriented API，或恢复日志扫描语义。

### SCH-5. profiler 对损坏缓存吞异常，诊断困难
- 位置：`src/eval/scheduler/profiler.py:53-54`
- 问题：`load_batch_cache` 使用裸 `except`。
- 建议：捕获具体异常并打印错误来源，避免“静默退化”。

### SCH-6. CLI 层仍是主入口，不利于配置驱动
- 位置：`src/eval/scheduler/cli.py:37-63`
- 问题：核心控制流深耦合 argparse 子命令。
- 建议：CLI 只做参数映射，调用统一 orchestrator API。

## 低优先级问题

### SCH-7. 兼容参数/no-op 参数积累
- 位置：如 `src/bin/eval_multi_choice_cot.py:89-92`, `src/bin/eval_instruction_following.py:81-84`
- 建议：在主流程收敛后移除兼容垃圾参数，保留一层迁移提示即可。

