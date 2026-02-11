# RWKV Skills

[English](README.md) | 中文

面向 RWKV7 的推理与评测脚手架，包含批量推理引擎、常见评测数据集的准备器以及一个 GPU 调度器骨架。

## 目录速览
- `src/infer`：RWKV 模型加载、采样策略与连续批量生成引擎。
- `src/infer/rwkv7`：上游 RWKV7 参考实现（含 CUDA 扩展、词表）已内置，无需额外子模块。
- `src/eval/datasets`：数据结构定义、JSONL 加载器以及各类数据集的准备脚本。
- `src/eval/evaluators`：多选 / 自由问答 / 指令遵循 / 代码生成（HumanEval、MBPP）评测管线。
- `src/eval/scheduler`：评测任务排队、GPU 侦测与调度的 CLI（现已附带 multi-choice / free-response / instruction-following / human-eval / mbpp 入口脚本）。
- `weights`、`data`、`results`（可选）：模型权重、数据集与评测产物的默认存放位置。

## 环境要求
- Python 3.12+，推荐安装 `uv` 以管理依赖。
- NVIDIA/AMD GPU（使用 `triton` 与内置 rapid-sampling 内核），需要与所选 PyTorch 发行版匹配的 CUDA/ROCm。

## 安装
```bash
# 安装依赖（示例：CUDA 12.9，对应 pyproject 中 torch-cu129 可选项）
uv sync --extra torch-cu129

# 开发模式安装，暴露 CLI 入口
uv pip install -e .
```
如需其他 CUDA/CPU 发行版，请改用 `--extra torch-cu126` / `--extra torch-cpu` 等。

## 下载模型权重
`rwkv-download-weights` 会从 Hugging Face 镜像枚举并并发下载 `.pth` 权重：
```bash
rwkv-download-weights /path/to/weights
# 或指定额外仓库：
rwkv-download-weights --repo BlinkDL/rwkv7-g1 --repo your/repo
```
可通过环境变量覆盖默认镜像与 Token（`HF_ENDPOINT`、`HF_TOKEN`）。

## 数据集准备
数据集默认存放在 `data/`。可以直接调用准备器生成 JSONL：
```bash
python - <<'PY'
from pathlib import Path
from src.eval.datasets.data_prepper.data_manager import prepare_dataset

prepare_dataset("mmlu", Path("data"))  # 会生成 data/mmlu/<split>.jsonl
PY
```
支持的数据集别名可通过 `available_*_datasets()` 系列函数查看。

## 评测与推理示例
目前推荐直接调用管线类：
```python
from src.eval.evaluators.multi_choice import MultipleChoicePipeline
from src.infer.model import ModelLoadConfig

pipeline = MultipleChoicePipeline(ModelLoadConfig(weights_path="weights/rwkv7-*.pth", device="cuda:0"))
result = pipeline.run_direct(
    dataset_path="data/mmlu/test.jsonl",
    output_path="results/completions/mmlu_direct.jsonl",
    sample_limit=50
)
print(result)
```
自由问答与指令遵循的用法类似，分别使用 `FreeResponsePipeline` 与 `InstructionFollowingPipeline`。

## 调度器 CLI
`rwkv-skills-scheduler` 暴露了一组命令（队列预览、调度、状态、停止、日志轮播）：
```bash
rwkv-skills-scheduler queue
rwkv-skills-scheduler dispatch --run-log-dir results/logs
```
其中 `queue` 是 `dispatch` 的 dry-run，会接受与 `dispatch` 一致的参数（包含 `--overwrite`）并输出将被调度的任务列表。
默认会跳过已有分数的任务。
若需强制重跑，可在 dispatch 时附上 `--overwrite`；调度器会创建新一轮/新版本结果，不会删除历史 completion / score / eval 记录。
评测脚本在配置好 API_KEY/JUDGE_MODEL 时默认会运行 LLM wrong-answer checker；如需关闭，可在 dispatch 时附上 `--disable-checker`。
可以用 `--only-datasets aime24 aime25` 这类参数仅重测指定 benchmark（名称即可，不需要 `_test` 后缀），也可以用 `--skip-datasets mmlu` 排除特定集合。若想只跑部分模型，无需填写完整路径，可使用 `--model-regex '^rwkv7-.*7\\.2b$'` 等正则过滤模型文件名，配合默认的权重 glob 即可。
默认模型 glob 在 `src/eval/scheduler/config.py` 中配置（仅指向仓库内 `weights/rwkv7-*.pth`，请按需覆盖）。调度器依赖的入口脚本已提供：
`src/bin/eval_multi_choice.py`、`eval_multi_choice_cot.py`、`eval_free_response.py`、`eval_free_response_judge.py`、`eval_instruction_following.py`、`eval_code_human_eval.py`、`eval_code_mbpp.py`、`eval_code_livecodebench.py`。
其中 `gsm8k_test` / `math_500_test` / `answer_judge_test` / `gaokao2023en_test` 这类需要 LLM 评分的 free-response benchmark 会自动被派发到 `eval_free_response_judge.py`，其余 free-response 仍走 `eval_free_response.py` 的 exact match 逻辑。
采样参数的网格搜索通过 param-search 流程完成：
- runner job 会把完整网格每个 trial 的 completions/eval/scores 写到 `results/param_search/{completions,eval,scores}/{model}/{benchmark}/trial_*.{jsonl,json}`。
- selector job 会统计 `results/param_search/scores/...`（默认综合 `gsm8k_test` + `hendrycks_math_test`，其中 `math` 会自动映射到 `hendrycks_math_test`），并把唯一最佳格点复制/写入到不带后缀的 `{benchmark}` 产物路径。

调度器在评测最新 2.9B 模型时，会自动对 `gsm8k_test` + `hendrycks_math_test` 启用 param-search。

## HumanEval 代码生成评测
- 数据集准备：`prepare_dataset("human_eval", Path("data"))` 会下载官方 `HumanEval.jsonl.gz` 并写出 `data/human_eval/test.jsonl`。
- 直接运行 CLI：
  ```bash
  python -m src.bin.eval_code_human_eval \
    --model-path weights/rwkv7-*.pth \
    --dataset data/human_eval/test.jsonl \
    --batch-size 128 \
    --pass-k 1 --pass-k 2 --pass-k 4 --pass-k 8 --pass-k 16 \
    --eval-timeout 3
  ```
  生成的样本写入 results/completions 结构，并会自动执行官方测试用例输出 pass@k 结果（生成次数等于最大的 k）。

## MBPP 代码生成评测
- 数据集准备：`prepare_dataset("mbpp", Path("data"))` 会使用 EvalPlus 版本的 MBPP+，并将 prompt 中的 4 空格转换为制表符。
- 运行 CLI：
  ```bash
  python -m src.bin.eval_code_mbpp \
    --model-path weights/rwkv7-*.pth \
    --dataset data/mbpp/test.jsonl \
    --batch-size 128 \
    --pass-k 1 --pass-k 2 --pass-k 4 --pass-k 8 --pass-k 16 \
    --eval-timeout 3
  ```
  会生成多样本并用 EvalPlus 测试用例执行，输出 pass@k（生成次数等于最大的 k）。

## LiveCodeBench 代码生成评测
- 数据集准备：`prepare_dataset("livecodebench", Path("data"))` 会下载 LiveCodeBench release_v6（lite）并写出 `data/livecodebench/test.jsonl`（可用 `RWKV_SKILLS_LIVECODEBENCH_VERSION_TAG` 覆盖版本）。
- 运行 CLI：
  ```bash
  python -m src.bin.eval_code_livecodebench \
    --model-path weights/rwkv7-*.pth \
    --dataset data/livecodebench/test.jsonl \
    --batch-size 64 \
    --pass-k 1 --pass-k 5 \
    --eval-timeout 6 \
    --eval-workers 12
  ```
  会抽取代码块并执行 LiveCodeBench 测试，输出 pass@k（生成次数等于最大的 k）。

## 已知缺口 / TODO
- 尚未支持其他代码基准（BigCodeBench 等）。

欢迎根据上述缺口补全实现并更新文档。

## 历史结果迁移
若已将旧版本 `rwkv-mmlu` / `rwkv-skills` 生成的 JSON 汇总到 `results_old/`，可以通过下述命令一次性迁移到当前 `results/scores/` 布局，避免重复跑全量评测：

```bash
python -m src.bin.migrate_old_results --source results_old
# 只想看看会写哪些文件，可加 --dry-run；已有结果但需要覆盖可加 --overwrite
```

## C. 仅使用调度器（DB）
### C.1 一次性准备
1. 准备 PostgreSQL 并可连接（写好 .env / 环境变量）  
   .env参考 `.env.example`
2. 准备模型权重：  
   `/home/jay/workspace/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1a-0.1b-20250728-ctx4096.pth`
3. 准备数据集目录：`/home/jay/workspace/rwkv-skills/data`（包含各任务 JSONL）

### C.2 调度器队列预览
用途：确认 8 个入口都会被调度、数据集路径可解析。
```bash
uv run rwkv-skills-scheduler queue \
  --model-select all \
  --models "<MODEL_PATH>" \
  --only-jobs code_human_eval code_livecodebench code_mbpp free_response free_response_judge instruction_following multi_choice_plain multi_choice_cot
```

### C.3 执行调度
用途：实际运行 8 个入口并写入数据库。
```bash
uv run rwkv-skills-scheduler dispatch \
  --model-select all \
  --models "<MODEL_PATH>" \
  --only-jobs code_human_eval code_livecodebench code_mbpp free_response free_response_judge instruction_following multi_choice_plain multi_choice_cot \
  --skip-missing-dataset
```

### C.4 监控/停止
```bash
rwkv-skills-scheduler status
rwkv-skills-scheduler logs
rwkv-skills-scheduler stop --all
```

### 多模型续跑逻辑
以 `model + dataset(+cot)` 为单位判断：默认会跳过已有分数；若无分数则续跑最近的未完成 task。若传 `--overwrite`，则会强制新建 task 重跑并写入新版本，不删除历史记录。任务失败会标记为 `failed`，下次调度在未产出分数前仍会续跑该 task。
