# vLLM RWKV Bugs

## rapid sampler greedy crash in knowledge CoT stage2

- Time found: 2026-07-06
- Symptom: knowledge CoT tasks reached `Generating CoT: 100%`, then failed or hung during the silent `Generating MC answer` stage.
- Fatal error: `rapid-sampling does not support greedy requests. Set VLLM_USE_RAPID_SAMPLER=0 to use the native greedy path.`
- Bad workaround: globally setting `VLLM_USE_RAPID_SAMPLER=0`. That fixes greedy stage2 but breaks stage1 because current CoT sampling sends `penalty_decay=0.99`, which requires rapid sampling.
- Working fix: keep rapid sampler enabled and make the short MC-answer stage non-greedy while still deterministic:
  - `temperature=1.0`
  - `top_k=1`
- Code touched:
  - local: `src/eval/tasks/knowledge/pipeline.py`
  - hotfix on 157 runtime project: `/home/rwkv/chase/rwkv-skills/src/eval/knowledge/pipeline.py`
- Operational cleanup done:
  - stale/failed `supergpqa` 13.3 tasks were replaced with patched reruns.
  - pre-patch `gpqa_extended` 1.5/2.9 runners were stopped early and relaunched with patched code to avoid wasting hours before stage2 failure.
