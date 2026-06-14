# Parallel Candidate Router Experiment

Experimental JSON-only TAU runner for testing a parallel candidate tool-call
router. This is not part of formal score dispatch.

Reproduce a small run against the current forwarded router:

```bash
rtk .venv/bin/python scripts/experiments/run_parallel_candidate_router_experiment.py \
  --dataset tau2_bench_airline/base \
  --infer-base-url http://127.0.0.1:19083 \
  --max-samples 2 \
  --max-steps 8
```

Artifacts are written under `out/parallel_candidate_router/<timestamp>/`:

- `health.json`
- `summary.json`
- `<model>/completions.jsonl`
- `<model>/eval.jsonl`
- `<model>/score.json`

The shared JSON/tool-call parsing and schema contract lives in
`src/eval/function_calling/tool_call_contract.py`. Experiment-only logic should
stay in this package.
