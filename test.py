from src.eval.evaluators.multi_choice import MultipleChoicePipeline
from src.infer.model import ModelLoadConfig

pipeline = MultipleChoicePipeline(ModelLoadConfig(weights_path="/home/alic-li/RWKV-v7/rwkv7-g1a3-2.9b-20251103-ctx8192.pth", device="cuda:0"))
result = pipeline.run_direct(
    dataset_path="data/mmlu/test.jsonl",
    output_path="results/completions/mmlu_direct.jsonl",
    sample_limit=50
)
print(result)