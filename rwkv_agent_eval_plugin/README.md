# RWKV Agent Eval Plugin

Reusable prompt-preparation helpers for RWKV ecosystem multi-turn agent
benchmarks. The plugin owns lexical tool-window routing, long-context evidence
selection, and trace payloads. It does not own model loading, scheduler state,
database writes, or benchmark scoring.

Use the TOML gate in rwkv-skills-like projects:

```toml
[default]
agent_plugin_enabled = true
long_context_min_chars = 3000
long_context_chunk_chars = 1000
long_context_max_evidence_chunks = 4
long_context_max_evidence_chars = 6000
prompt_max_chars = 24576
```

`prompt_max_chars` is the final rendered agent prompt hard cap. It is not the
per-chunk size. Long-document chunking is controlled by
`long_context_chunk_chars`, `long_context_max_evidence_chunks`, and
`long_context_max_evidence_chars`.

Direct use:

```python
from rwkv_agent_eval_plugin import agent_plugin_config_from_sources, route_agent_prompt_inputs

config = agent_plugin_config_from_sources({"agent_plugin_enabled": True})
route = route_agent_prompt_inputs(
    domain_policy=policy,
    tools=tools,
    messages=messages,
    config=config,
)

selected_tools = route.selected_tools
messages_for_prompt = route.messages
policy_for_prompt = route.domain_policy
trace = route.trace_payload()
```
