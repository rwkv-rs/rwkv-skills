# Lexical Chunk Router

RWKV long-context agent benchmark helpers for fitting long documents, tool
outputs, domain policies, and large tool catalogs into a smaller prompt window.

The package is RWKV-oriented, but it is intentionally independent from one
project's eval runner, database, scheduler, or inference-server implementation.
Projects pass their own backend object for model-based routing.

Public API is exported from `src.plugins.lexical_chunk_router.__all__`, and the
package ships `py.typed` for type-checker visibility.

Use it from this repository with:

```python
from src.plugins.lexical_chunk_router import LongDocConfig, ToolRouterConfig, compact_text, route_tools
```

For another rwkv-skills-like project, copy the `src/plugins/lexical_chunk_router/`
directory or install a wheel that includes this package.

## Long Document Compaction

```python
from src.plugins.lexical_chunk_router import LongDocConfig, compact_text

result = compact_text(
    long_text,
    query="Which invoice is paid?",
    config=LongDocConfig(max_chunk_chars=900, max_evidence_chunks=4, max_evidence_chars=3000),
)
prompt_context = result.text
trace = result.trace_payload()
```

## Tool Catalog Routing

```python
from src.plugins.lexical_chunk_router import ToolRouterConfig, route_tools

route = route_tools(
    tools,
    messages,
    config=ToolRouterConfig(max_tools=8, trigger_tool_count=12, trigger_catalog_chars=4000),
)
visible_tools = route.selected_tools
trace = route.trace_payload()
```

## RWKV Model Routing

Long-document compaction is lexical only. Tool routing supports a single model
router call through `ToolRouterConfig(mode="model")`; parallel candidate
selection lives in the formal function-calling runner API, not in this package.

```python
route = route_tools(
    tools,
    messages,
    config=ToolRouterConfig(mode="model"),
    backend=rwkv_backend,
    sampling=router_sampling,
)
```

The backend protocol is available as `RwkvRouterBackend`, and the package also
exports RWKV JSON-call helpers such as `build_rwkv_json_call_prompt`,
`extract_json_call_value_text`, `JSON_CALL_STOP_SUFFIXES`, and
`clamp_router_sampling`.

## Notes

- `compact_text` preserves line numbers and emits an evidence window with chunk
  ids, line spans, and lexical scores.
- `route_tools` accepts OpenAI-style function schemas, mappings with
  `name`/`description`/`parameters`, or objects with `name` and `description`.
- `enable_domain_hints=True` keeps a small set of tau-style airline/retail
  anchors that worked well in rwkv-skills experiments. Disable it for generic
  projects that only want token overlap scoring.
- `src.plugins.lexical_chunk_router` is the stable in-repository API.
