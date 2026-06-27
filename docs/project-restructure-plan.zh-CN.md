# 项目结构整理计划

## 背景与目标

`rwkv-skills` 经历了从 Gradio → Next.js+Rust → FastAPI+Vite 的多轮前端演进，留下了大量结构债：

- **`src/space/` 命名与职责混乱**：名字是 HF Space/Gradio 时代遗留；内部把"纯逻辑层"和"FastAPI Web 层"混在同一层目录。
- **Next.js 残留散落根目录**：`public/`、`types/next-types-js.d.ts`、`.gitignore` 里的 `.next/` 等条目。
- **根目录散落文件**：临时调试脚本 `test.py`、与 `src/main.py` 重复的转发壳 `main.py`、旧规划 `plan.md`。
- **`src/bin/` 入口与一次性脚本混杂**：6 个 pyproject 注册入口混着 ~22 个 migrate/audit/probe 等一次性脚本。

目标：**前端（`frontend/`）与后端（`src/`）严格分离、命名语义化、Web 层与纯逻辑层分层**，并清除历史残留。

> 关键事实（已探查）：外部仅 `src/bin/run_dashboard.py` 与 `src/space/serialize.py` 引用 `src.space`；`src/space` 内部 41 处全为相对 import。重命名风险可控。

---

## 目标结构

```
rwkv-skills/
├── frontend/                      # 前端（已自包含，保持不动）
│   ├── src/{App.tsx, api.ts, components/, ...}
│   ├── package.json, vite.config.ts, tsconfig.json, index.html
│   └── dist/                      # 构建产物（应 gitignore）
├── src/                           # 后端
│   ├── dashboard/                 # ← 原 src/space，改名+分层
│   │   ├── __init__.py
│   │   ├── web/                   # FastAPI 层（HTTP / 序列化 / 服务封装）
│   │   │   ├── api.py
│   │   │   ├── admin_api.py
│   │   │   ├── serialize.py
│   │   │   ├── charts_json.py
│   │   │   └── eval_service.py
│   │   └── core/                  # 纯逻辑（框架无关，可测）
│   │       ├── data.py  metrics.py  selection.py  tables.py
│   │       ├── charts.py  domains.py  constants.py
│   │       ├── vocab.py  score_index.py  eval_records.py
│   ├── bin/                       # 仅保留 pyproject 注册的正式入口
│   ├── eval/  infer/  db/  plugins/  infra/  main.py
│   └── ...
├── scripts/
│   └── oneoff/                    # ← src/bin 里的一次性脚本归档于此
├── docs/  tests/  configs/  vendor/  lexical_chunk_router/
└── pyproject.toml  README*.md  uv.lock  .gitignore
```

---

## 分阶段执行

每阶段独立可验证、可独立提交。**当前仅为计划，未执行。**

### 阶段 1 — 清 Next.js 根残留（零风险）
- 删除 `public/`（`next.svg`、`vercel.svg`、`public/vendor/` 整套 KaTeX/highlight/markdown-it/fonts）
- 删除 `types/next-types-js.d.ts`（空目录则删 `types/`）
- `.gitignore`：移除 `.next/`、`next-env.d.ts`、`.vercel/`；把 `# Node/Next frontend` 注释改为 `# Node/Vite frontend`；新增 `frontend/dist/`、`*.tsbuildinfo` 已覆盖
- **验证**：`git status` 干净；`grep -rn "next\|vercel" .gitignore` 无残留

### 阶段 2 — 根目录收口（低风险）
- 删 `test.py`（硬编码 `/home/alic-li/...` 的临时脚本）
- 删根 `main.py`（仅 `from src.main import main`；`rwkv-skills` 入口走 `src.main:main`，不依赖它）
- `plan.md` → 移到 `docs/archive/plan-2026-06-16.md`（保留历史，不留根目录）
- **验证**：`rwkv-skills --help` 仍可运行；`python -c "from src.main import main"` 通过

### 阶段 3 — `src/space/` → `src/dashboard/`，分 web/core（中风险，核心）
1. `git mv src/space src/dashboard`
2. 在 `dashboard/` 下建 `web/` 与 `core/`，按上方目标结构 `git mv` 各文件
3. 重写 import：
   - 跨层引用补包路径：`web/*` 引用纯逻辑改 `from ..core.xxx import ...`；`core` 内部相对 import 不变
   - `web/api.py` 中 `_REPO_ROOT = Path(__file__).resolve().parents[2]` → 因层级加深改为 `parents[3]`，`_SPA_DIST` 指向 `frontend/dist` 不变（需重算层级）
   - `__init__.py` 更新 docstring 与（如有）re-export
4. `pyproject.toml`：`rwkv-skills-dashboard = "src.bin.run_dashboard:main"` 不变；但 `run_dashboard.py` 内 `src.space.api:app` → `src.dashboard.web.api:app`
5. `src/bin/run_dashboard.py`、`src/dashboard/web/serialize.py` 的 `src.space` 引用全部改 `src.dashboard...`
- **验证**：
  - `python -c "from src.dashboard.web.api import create_app; create_app()"`
  - `TestClient` 跑 `/api/meta`、`/api/leaderboard`、`/api/admin/eval/status` 均 200
  - `pnpm --dir frontend build` 后访问 `/` 返回 SPA

### 阶段 4 — `src/bin/` 分入口与脚本（中风险）
- `src/bin/` 仅保留：`run_dashboard`、`run_infer_server`、`run_infer_fleet`、`run_infer_router`、`run_perf_benchmark`、`download_weights`（+ `__init__.py`、`data/`）
- 其余一次性脚本 `git mv` 到 `scripts/oneoff/`：`migrate_*`、`audit_*`、`*_infer_swap_*`、`param_search_*`、`probe_remote_infer`、`backfill_*`、`convert_*`、`summarize_*`、`validate_*`、`verify_*`、`preflight_*`、`prepare_*`、`clean_old_imports`、`run_function_calling_matrix`、`run_llm_checker`、`run_openai_tool_call_adapter`
- **移动前必做**：`grep -rn "src.bin.<name>" src tests pyproject.toml` 确认无被 import（已初步确认 pyproject 未引用这些）
- **验证**：6 个入口 `--help` 正常；`pytest -q` 不因路径变动失败

### 阶段 5 — 顶层杂物归位（低风险，可选）
- 确认 tracked 的 `albatross/` 是否为空/废弃，是则删
- README（中/英）新增"项目结构"小节，说明 `frontend/` + `src/dashboard/{web,core}` 布局
- `config_backup/`、`summary-6/`、`tmp/`、`logs/` 等本地产物已被 gitignore，保留本地、确保不入库

---

## 风险与回滚

- 全程用 `git mv` 保留历史；每阶段单独 commit，便于 `git revert` 单阶段回滚。
- 阶段 3 是唯一牵动 import 的改动，且引用面极小（2 处外部 + 41 处内部相对 import），改完即用 TestClient 端到端验证。
- 阶段 4 移动前先 grep 确认无交叉 import，避免运行期 `ModuleNotFoundError`。

## 总验证（全部完成后）
```bash
.venv/bin/python -c "from src.dashboard.web.api import create_app; create_app()"
.venv/bin/python -m pytest -q
pnpm --dir frontend build
.venv/bin/python -m src.bin.run_dashboard --host 127.0.0.1 --port 7862  # 手测 /, /api/*, /api/admin/*
```
