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
- NVIDIA GPU（使用 `flashinfer`、`triton` 等依赖），需要与所选 PyTorch 发行版匹配的 CUDA/ROCm。

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

pipeline = MultipleChoicePipeline(ModelLoadConfig(weights_path="weights/rwkv7-*.pth"))
result = pipeline.run_direct(
    dataset_path="data/mmlu/test.jsonl",
    output_path="results/completions/mmlu_direct.jsonl",
)
print(result)
```
自由问答与指令遵循的用法类似，分别使用 `FreeResponsePipeline` 与 `InstructionFollowingPipeline`。

## 调度器 CLI
`rwkv-skills-scheduler` 暴露了一组命令（队列预览、调度、状态、停止、日志轮播）：
```bash
rwkv-skills-scheduler queue
rwkv-skills-scheduler dispatch --completion-dir results/completions --run-log-dir results/logs --eval-result-dir results/eval
```
若需无视 `results/scores` 中已存在的结果并强制重跑，可在 dispatch 时附上 `--overwrite`，调度器会在启动前删除旧的 completion / score / eval 产物再重新评测。
可以用 `--only-datasets aime24 aime25` 这类参数仅重测指定 benchmark（名称即可，不需要 `_test` 后缀），也可以用 `--skip-datasets mmlu` 排除特定集合。若想只跑部分模型，无需填写完整路径，可使用 `--model-regex '^rwkv7-.*7\\.2b$'` 等正则过滤模型文件名，配合默认的权重 glob 即可。
默认模型 glob 在 `src/eval/scheduler/config.py` 中配置（仅指向仓库内 `weights/rwkv7-*.pth`，请按需覆盖）。调度器依赖的入口脚本已提供：
`src/bin/eval_multi_choice.py`、`eval_multi_choice_cot.py`、`eval_free_response.py`、`eval_free_response_judge.py`、`eval_instruction_following.py`、`eval_code_human_eval.py`、`eval_code_mbpp.py`。
其中 `gsm8k_test` / `math_500_test` / `answer_judge_test` / `gaokao2023en_test` 这类需要 LLM 评分的数学问答会自动被派发到 `eval_free_response_judge.py`，其余 free-response 仍走 `eval_free_response.py` 的 exact match 逻辑。
采样参数的网格搜索通过 param-search 流程完成：
- runner job 会把完整网格（先 `normal` 后 `simple`，不截断）每个 trial 的 completions/eval/scores 写到 `results/param_search/{completions,eval,scores}/{model}/{benchmark}/trial_*.{jsonl,json}`。
- selector job 会统计 `results/param_search/scores/...`（默认综合 `gsm8k_test` + `hendrycks_math_test`，其中 `math` 会自动映射到 `hendrycks_math_test`），并保留两套互相独立的选参结果：
  - `normal` 最优格点 -> 复制/写入到 `results/{completions,eval,scores}`，benchmark 名称追加后缀 `__ps_normal`
  - `simple` 最优格点 -> 复制/写入到 `results/{completions,eval,scores}`，benchmark 名称追加后缀 `__ps_simple`
  -（兼容旧逻辑）两种模式里总分最高的格点仍会写入不带后缀的 `{benchmark}` 产物路径。

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

## 已知缺口 / TODO
- 尚未支持其他代码基准（LiveCodeBench/BigCodeBench 等）；当前代码生成仅覆盖 HumanEval 与 MBPP。

欢迎根据上述缺口补全实现并更新文档。

## 历史结果迁移
若已将旧版本 `rwkv-mmlu` / `rwkv-skills` 生成的 JSON 汇总到 `results_old/`，可以通过下述命令一次性迁移到当前 `results/scores/` 布局，避免重复跑全量评测：

```bash
python -m src.bin.migrate_old_results --source results_old
# 只想看看会写哪些文件，可加 --dry-run；已有结果但需要覆盖可加 --overwrite
```

迁移脚本会自动识别多选 / 数学自由问答 / instruction-following 等任务类型，保留科目细分指标，并在默认情况下跳过仓库里已有的 score JSON。
