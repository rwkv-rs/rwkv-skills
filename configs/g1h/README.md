# G1h Benchmark Config Override

The formal 157 evaluation checkout is the authoritative caller for this
directory. From that checkout, use the relative path below:

```bash
cd /home/rwkv/chase/rwkv-skills-g1h-normal-20260719
PYTHONPATH=/home/rwkv/chase/rwkv-skills-g1h-normal-20260719
RWKV_BENCHMARK_CONFIG_ROOT=configs/g1h
```

Do not point a formal scheduler at the shared main checkout. The shared
checkout owns symlinked data/results and service code; the formal checkout
owns the scheduler, runner, benchmark code, and G1h config used for a run.

Policy from prompt probes on 2026-07-17:

- Prefer compact empty think for answer benchmarks: `Bot✿<think></think>`.
- Avoid `Assistant: <think` and `Assistant: <think>\n` for G1h runs.
- `✿` stop token id is `10060`.
- Function-calling fenced JSON is not handled by `stop_tokens` alone. FC runners should stop on fenced JSON suffixes (`\n````, ```), role boundaries, and `✿` when official prompt style is enabled.

Function-calling policy after the 2026-07-17 fix:

- Use runner prompt style `rwkv_flower_json` for formal g1h FC reruns.
- The assistant prefix is `Bot✿<think></think>` followed by a fenced JSON prefill (` ```json ` then `{`), so generation continues inside the final JSON object content.
- FC runners keep `stop_tokens = [0]` after clamping and rely on string stop suffixes for code fences, role boundaries, and `✿`.
- BrowseComp-Plus formal reruns must keep `candidate_router_mode = "parallel"` so normal search decisions use `parallel_candidate`.
