## Eval DB

### 表结构说明
- `version`: 评测版本/运行元信息（job/dataset/model/git）
- `completions`: 模型输出轨迹（prompt/completion/stop/context）
- `eval`: 样本级评测结果（pass/fail + answer/ref）
- `score`: 汇总指标（dataset/model/cot）
- `logs`: 调度/运行事件
- `view_score_latest`: 最新 score 视图（按 dataset/model/cot）
- `view_eval_completion`: eval 与 completions 的样本级 join 视图

### 环境变量
- DB 连接配置从 `.env` 读取
- `RWKV_DB_ENABLED=1`
- `PG_HOST`, `PG_PORT`, `PG_USER`, `PG_PASSWORD`, `PG_DBNAME`

### 初始化数据库
```bash
python -m src.bin.init_eval_db --force
```

### 评测命令（写入 DB）
DB 模式下不写入 `results` 目录
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
  --batch-size 8 \
  --max-samples 10
```