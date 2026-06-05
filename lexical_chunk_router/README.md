# Lexical Chunk Router

Small, model-free helpers for rwkv-skills-like projects that need to fit long
documents, tool outputs, or large tool catalogs into a smaller prompt window.

The package is intentionally independent from eval runners, databases, and RWKV
inference code. The first version only implements lexical routing.

Public API is exported from `lexical_chunk_router.__all__`, and the package ships
`py.typed` for type-checker visibility.

Use it from this repository with:

```python
from lexical_chunk_router import LongDocConfig, ToolRouterConfig, compact_text, route_tools
```

For another rwkv-skills-like project, copy the `lexical_chunk_router/` directory
or install a wheel that includes this package. The old
`src.plugins.lexical_chunk_router` path in this repository is only a compatibility
re-export.

## Long Document Compaction

```python
from lexical_chunk_router import LongDocConfig, compact_text

result = compact_text(
    long_text,
    query="Which invoice is paid?",
    config=LongDocConfig(max_chunk_chars=900, max_evidence_chunks=4, max_evidence_chars=3000),
)
prompt_context = result.text
```

## Tool Catalog Routing

```python
from lexical_chunk_router import ToolRouterConfig, route_tools

route = route_tools(
    tools,
    messages,
    config=ToolRouterConfig(max_tools=8, trigger_tool_count=12, trigger_catalog_chars=4000),
)
visible_tools = route.selected_tools
trace = route.trace_payload()
```

## Notes

- `compact_text` preserves line numbers and emits an evidence window with chunk
  ids, line spans, and lexical scores.
- `route_tools` accepts OpenAI-style function schemas, mappings with
  `name`/`description`/`parameters`, or objects with `name` and `description`.
- `enable_domain_hints=True` keeps a small set of tau-style airline/retail
  anchors that worked well in rwkv-skills experiments. Disable it for generic
  projects that only want token overlap scoring.
