## Eval DB

### 表结构说明
- `eval_dataset`: 数据集维表，按 `dataset_slug` 唯一，记录 domain/version/meta。
- `eval_split`: 数据集 split 维表，按 `(dataset_id, split_name)` 唯一。
- `eval_sample`: 样本维表，按 `(dataset_id, split_id, sample_index)` 唯一，存题目/参考答案/原始 meta。
- `eval_model`: 模型维表，按 `(model_slug, model_revision)` 唯一（revision 为空时按空串归一）。
- `eval_task`: 任务维表，`task_id=<dataset_slug>__<model_slug>` 唯一，关联 dataset/model。
- `eval_run`: 一次运行/实验记录，关联 task，存采样/运行配置、代码版本、状态与时间。
- `eval_run_sample`: run 下样本实例（含 repeat），按 `(run_id, sample_id, repeat_index)` 唯一。
- `eval_attempt`: run_sample 的尝试记录，按 `(run_sample_id, attempt_index)` 唯一。
- `eval_stage_output`: 阶段化输出（cot/final/other），支持 partial 与最终段标记。
- `eval_cot_checkpoint`: CoT 续跑检查点，`(attempt_id, stage)` 的 latest 仅一条。
- `eval_metric`: 评测指标记录，挂在 run_sample。
- `eval_run_event`: 运行事件/汇总记录，挂在 run 或 run_sample。

### 环境变量
- DB 连接配置从 `.env.example` 读取
- `RWKV_DB_ENABLED=1`
- `PG_HOST`, `PG_PORT`, `PG_USER`, `PG_PASSWORD`, `PG_DBNAME`

### 初始化数据库
```bash
python -m src.bin.init_eval_db --force
```

### 评测命令（写入 DB + 产出 JSONL/score）
```bash
python -m src.bin.eval_free_response --model-path <pth> --dataset <jsonl>
python -m src.bin.eval_free_response_judge --model-path <pth> --dataset <jsonl>
python -m src.bin.eval_multi_choice --model-path <pth> --dataset <jsonl>
python -m src.bin.eval_multi_choice_cot --model-path <pth> --dataset <jsonl>
python -m src.bin.eval_instruction_following --model-path <pth> --dataset <jsonl>
python -m src.bin.eval_code_human_eval --model-path <pth> --dataset <jsonl>
python -m src.bin.eval_code_mbpp --model-path <pth> --dataset <jsonl>
python -m src.bin.eval_code_livecodebench --model-path <pth> --dataset <jsonl>
python -m src.bin.param_search_free_response --model-path <pth> --dataset <jsonl>
python -m src.bin.param_search_free_response_judge --model-path <pth> --dataset <jsonl>
python -m src.bin.param_search_select --model-path <pth>
```
```bash
uv run src/bin/eval_free_response.py \
  --device cuda:0 \
  --model-path /home/jay/workspace/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1a-0.4b-20250905-ctx4096.pth \
  --dataset /home/jay/workspace/rwkv-skills/data/gsm8k/test.jsonl \
  --output /home/jay/workspace/rwkv-skills/results/gsm8k_test_db_run.jsonl \
  --batch-size 8 \
  --max-samples 10
```