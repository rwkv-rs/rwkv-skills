from __future__ import annotations

"""BrowseComp-Plus agent adapter boundary."""

from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.function_calling.agent.env import AgentObservation, AgentStepResult
from src.eval.function_calling.common.action import ToolAction

OFFICIAL_BROWSECOMP_PLUS_SOURCE = "texttron/BrowseComp-Plus"
DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT = Path("/tmp/rwkv-official-refs/BrowseComp-Plus")


@dataclass(frozen=True, slots=True)
class BrowseCompPlusAdapterConfig:
    official_root: Path = DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT
    index_path: Path | None = None
    k: int = 5
    include_get_document: bool = True
    snippet_max_chars: int = 2000


BROWSECOMP_PLUS_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "search",
    "description": "Search the fixed BrowseComp-Plus corpus and return relevant documents.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
        },
        "required": ["query"],
    },
}
BROWSECOMP_PLUS_GET_DOCUMENT_SCHEMA: dict[str, Any] = {
    "name": "get_document",
    "description": "Retrieve a full BrowseComp-Plus document by docid.",
    "parameters": {
        "type": "object",
        "properties": {
            "docid": {"type": "string", "description": "Document id."},
        },
        "required": ["docid"],
    },
}
BROWSECOMP_PLUS_FINAL_SCHEMA: dict[str, Any] = {
    "name": "final_answer",
    "description": "Finish the task with the final answer.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Final answer with concise supporting explanation."},
        },
        "required": ["answer"],
    },
}
BROWSECOMP_PLUS_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    BROWSECOMP_PLUS_SEARCH_SCHEMA,
    BROWSECOMP_PLUS_GET_DOCUMENT_SCHEMA,
    BROWSECOMP_PLUS_FINAL_SCHEMA,
)


def require_browsecomp_plus_assets(config: BrowseCompPlusAdapterConfig | None = None) -> Path:
    cfg = config or BrowseCompPlusAdapterConfig()
    root = cfg.official_root.expanduser().resolve()
    if not (root / "scripts_evaluation" / "evaluate_run.py").exists():
        raise FileNotFoundError(f"BrowseComp-Plus official evaluator not found under {root}")
    return root


def browsecomp_plus_index_path(root: str | Path | None = None) -> Path:
    override = (
        os.environ.get("RWKV_BROWSECOMP_PLUS_BM25_INDEX")
        or os.environ.get("BROWSECOMP_PLUS_BM25_INDEX")
    )
    if override:
        return Path(override).expanduser().resolve()
    resolved_root = Path(root).expanduser().resolve() if root else DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT
    return (resolved_root / "indexes" / "bm25").resolve()


def require_browsecomp_plus_bm25_index(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"BrowseComp-Plus BM25 index not found: {resolved}")
    if not any(resolved.glob("segments_*")):
        raise FileNotFoundError(f"BrowseComp-Plus BM25 index does not look like a Lucene index: {resolved}")
    return resolved


def load_browsecomp_plus_rows_from_decrypted_jsonl(
    path: str | Path,
    *,
    official_root: str | Path | None = None,
    dataset_name: str = "browsecomp_plus",
    max_steps: int = 16,
) -> list[dict[str, Any]]:
    root = Path(official_root).expanduser().resolve() if official_root else DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT
    rows: list[dict[str, Any]] = []
    source_path = Path(path).expanduser().resolve()
    with source_path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, Mapping):
                continue
            query_id = str(item.get("query_id") or item.get("id") or index)
            query = str(item.get("query") or item.get("question") or "").strip()
            if not query:
                continue
            documents = _list_of_dicts(item.get("documents") or item.get("evidence_documents"))
            if not documents:
                documents = _official_row_documents(item)
            index_path = browsecomp_plus_index_path(root)
            rows.append(
                {
                    "task_id": f"{dataset_name}__{query_id}",
                    "instruction": query,
                    "messages": [{"role": "user", "content": query}],
                    "tools": [dict(tool) for tool in BROWSECOMP_PLUS_TOOL_SCHEMAS],
                    "expected_tool_calls": [],
                    "env": {
                        "type": "browsecomp_plus",
                        "query_id": query_id,
                        "k": 5,
                        "official_root": str(root),
                        "index_path": str(index_path),
                    },
                    "scorer": {"type": "browsecomp_plus_official"},
                    "max_steps": max_steps,
                    "metadata": {
                        "source_format": "official_browsecomp_plus",
                        "official_source": OFFICIAL_BROWSECOMP_PLUS_SOURCE,
                        "browsecomp_plus_official_root": str(root),
                        "browsecomp_plus_bm25_index_path": str(index_path),
                        "browsecomp_plus_source_path": str(source_path),
                        "query_id": query_id,
                        "query": query,
                        "answer": item.get("answer"),
                        **({"browsecomp_plus_documents": documents} if documents else {}),
                    },
                }
            )
    return rows


class BrowseCompPlusEnv:
    """Local agent environment for BrowseComp-Plus run-file generation."""

    def __init__(
        self,
        record: FunctionCallTaskRecord | Mapping[str, Any],
        *,
        config: BrowseCompPlusAdapterConfig | None = None,
    ) -> None:
        self.record = record
        self.config = config or BrowseCompPlusAdapterConfig()
        metadata = _record_metadata(record)
        env = _record_env(record)
        self.query_id = str(env.get("query_id") or metadata.get("query_id") or _record_task_id(record))
        self.query = str(metadata.get("query") or _record_instruction(record) or "")
        self.answer = metadata.get("answer")
        self.k = _positive_int(env.get("k")) or self.config.k
        root_value = (
            env.get("official_root")
            or metadata.get("browsecomp_plus_official_root")
            or self.config.official_root
        )
        self.official_root = Path(str(root_value)).expanduser().resolve()
        index_value = (
            env.get("index_path")
            or metadata.get("browsecomp_plus_bm25_index_path")
            or self.config.index_path
        )
        self.index_path = (
            Path(str(index_value)).expanduser().resolve()
            if index_value
            else browsecomp_plus_index_path(self.official_root)
        )
        self.snippet_max_chars = _positive_int(env.get("snippet_max_chars")) or self.config.snippet_max_chars
        self.documents = _list_of_dicts(metadata.get("browsecomp_plus_documents") or env.get("documents"))
        if not self.documents and metadata.get("browsecomp_plus_source_path"):
            self.documents = _load_documents_for_query(
                Path(str(metadata["browsecomp_plus_source_path"])),
                self.query_id,
            )
        self._prefer_record_documents = bool(self.documents) and not index_value
        self._searcher: _PyseriniBM25Searcher | None = None
        self.tool_call_counts: dict[str, int] = {}
        self.retrieved_docids: set[str] = set()
        self.final_output = ""
        self.status = "incomplete"
        self.actions: list[dict[str, Any]] = []

    def reset(self) -> AgentObservation:
        self.tool_call_counts = {}
        self.retrieved_docids = set()
        self.final_output = ""
        self.status = "incomplete"
        self.actions = []
        if not self._has_official_index() and not self.documents:
            raise FileNotFoundError(
                "BrowseComp-Plus requires the official BM25 index or per-record documents. "
                f"Expected index at {self.index_path}. Set RWKV_BROWSECOMP_PLUS_BM25_INDEX "
                "or BROWSECOMP_PLUS_BM25_INDEX if the index is elsewhere."
            )
        return AgentObservation(
            _initial_observation(self.query),
            {
                "benchmark": "browsecomp_plus",
                "query_id": self.query_id,
                "available_tools": [tool["name"] for tool in BROWSECOMP_PLUS_TOOL_SCHEMAS],
            },
        )

    def step(self, action: ToolAction) -> AgentStepResult:
        self.actions.append({"name": action.name, "arguments": dict(action.arguments)})
        if action.name == "search":
            return self._search(action)
        if action.name == "get_document":
            return self._get_document(action)
        if action.name == "final_answer":
            return self._final_answer(action)
        details = {"fail_reason": "unknown_tool", "actual_tool": action.name, **self._run_details()}
        return AgentStepResult(
            AgentObservation(f"Unknown tool: {action.name}", {"error": True, "query_id": self.query_id}),
            done=True,
            score=0.0,
            success=False,
            details=details,
        )

    def _search(self, action: ToolAction) -> AgentStepResult:
        query = str(action.arguments.get("query") or "").strip()
        self._count_tool("search")
        results = self._search_documents(query, self.k)
        for result in results:
            docid = result.get("docid")
            if docid is not None:
                self.retrieved_docids.add(str(docid))
        content = json.dumps(results, ensure_ascii=False, separators=(",", ":"))
        return AgentStepResult(
            AgentObservation(
                f"Search results: {content}",
                {
                    "query_id": self.query_id,
                    "tool": "search",
                    "retrieved_docids": sorted(self.retrieved_docids),
                },
            ),
            done=False,
            details={"tool": "search", "result_count": len(results), **self._run_details()},
        )

    def _get_document(self, action: ToolAction) -> AgentStepResult:
        docid = str(action.arguments.get("docid") or "").strip()
        self._count_tool("get_document")
        if self._has_official_index():
            document = self._official_searcher().get_document(docid)
        else:
            document = next((item for item in self.documents if str(item.get("docid") or item.get("id") or "") == docid), None)
        if document is None:
            content = json.dumps({"docid": docid, "error": "not_found"}, ensure_ascii=False)
        else:
            self.retrieved_docids.add(docid)
            content = json.dumps(_document_payload(document), ensure_ascii=False, separators=(",", ":"))
        return AgentStepResult(
            AgentObservation(
                f"Document: {content}",
                {
                    "query_id": self.query_id,
                    "tool": "get_document",
                    "retrieved_docids": sorted(self.retrieved_docids),
                },
            ),
            done=False,
            details={"tool": "get_document", **self._run_details()},
        )

    def _final_answer(self, action: ToolAction) -> AgentStepResult:
        answer = str(action.arguments.get("answer") or "").strip()
        self.final_output = answer
        self.status = "completed" if answer else "incomplete"
        success = bool(answer)
        details = {
            "finish_reason": "final_answer",
            "official_score_unavailable": True,
            **self._run_details(),
        }
        return AgentStepResult(
            AgentObservation("Final answer recorded.", {"query_id": self.query_id, "done": True}),
            done=True,
            score=None if success else 0.0,
            success=success,
            details=details,
        )

    def _search_documents(self, query: str, k: int) -> list[dict[str, Any]]:
        if self._has_official_index():
            return [
                _search_result_payload(item, score, snippet_max_chars=self.snippet_max_chars)
                for item, score in self._official_searcher().search(query, k)
            ]
        if not self.documents:
            raise FileNotFoundError(
                "BrowseComp-Plus search has no official BM25 index and no per-record documents"
            )
        query_tokens = _tokenize(query)
        scored = sorted(
            self.documents,
            key=lambda item: _document_score(query_tokens, item),
            reverse=True,
        )
        return [
            _search_result_payload(item, score, snippet_max_chars=self.snippet_max_chars)
            for item, score in (_score_pair(query_tokens, doc) for doc in scored[:k])
        ]

    def _count_tool(self, name: str) -> None:
        self.tool_call_counts[name] = self.tool_call_counts.get(name, 0) + 1

    def _run_details(self) -> dict[str, Any]:
        return {
            "browsecomp_plus_run": {
                "query_id": self.query_id,
                "tool_call_counts": dict(self.tool_call_counts),
                "status": self.status,
                "retrieved_docids": sorted(self.retrieved_docids),
                "result": [{"type": "output_text", "output": self.final_output}] if self.final_output else [],
            },
            "retrieved_docids": sorted(self.retrieved_docids),
            "tool_call_counts": dict(self.tool_call_counts),
            "retriever": "bm25" if self._has_official_index() else "record_documents",
            "index_path": str(self.index_path),
        }

    def _has_official_index(self) -> bool:
        if self._prefer_record_documents:
            return False
        return self.index_path.exists() and any(self.index_path.glob("segments_*"))

    def _official_searcher(self) -> "_PyseriniBM25Searcher":
        if self._searcher is None:
            self._searcher = _get_pyserini_bm25_searcher(self.index_path)
        return self._searcher


def create_browsecomp_plus_env(
    record: FunctionCallTaskRecord | Mapping[str, Any],
    *,
    config: BrowseCompPlusAdapterConfig | None = None,
) -> BrowseCompPlusEnv:
    return BrowseCompPlusEnv(record, config=config)


def browsecomp_plus_run_from_agent_details(details: Mapping[str, Any]) -> dict[str, Any] | None:
    run = details.get("browsecomp_plus_run")
    if isinstance(run, Mapping):
        return dict(run)
    final_details = details.get("final_env_details")
    if isinstance(final_details, Mapping):
        run = final_details.get("browsecomp_plus_run")
        if isinstance(run, Mapping):
            return dict(run)
    return None


def _initial_observation(query: str) -> str:
    return (
        "You are answering a BrowseComp-Plus deep-research question against a fixed corpus.\n"
        "Use search or get_document as needed. When ready, call final_answer.\n"
        "Final answer should include the exact answer and concise evidence citations using [docid] when available.\n"
        f"Question: {query}"
    )


def _score_pair(tokens: set[str], document: Mapping[str, Any]) -> tuple[Mapping[str, Any], float]:
    return document, float(_document_score(tokens, document)[0])


def _document_score(tokens: set[str], document: Mapping[str, Any]) -> tuple[int, int, str]:
    text = _document_text(document).lower()
    doc_tokens = _tokenize(text)
    overlap = len(tokens & doc_tokens)
    substring_bonus = sum(1 for token in tokens if token and token in text)
    return overlap, substring_bonus, str(document.get("docid") or document.get("id") or "")


def _search_result_payload(
    document: Mapping[str, Any],
    score: float,
    *,
    snippet_max_chars: int,
) -> dict[str, Any]:
    payload = _document_payload(document)
    text = str(payload.get("text") or payload.get("snippet") or "")
    return {
        "docid": str(payload.get("docid") or ""),
        "score": score,
        "snippet": text[:snippet_max_chars] if snippet_max_chars > 0 else text,
    }


def _document_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    docid = str(document.get("docid") or document.get("id") or "")
    text = _document_text(document)
    return {"docid": docid, "text": text}


def _document_text(document: Mapping[str, Any]) -> str:
    for key in ("text", "contents", "content", "snippet", "body"):
        value = document.get(key)
        if isinstance(value, str):
            return value
    return json.dumps(dict(document), ensure_ascii=False, separators=(",", ":"))


def _record_metadata(record: FunctionCallTaskRecord | Mapping[str, Any]) -> dict[str, Any]:
    metadata = record.metadata if isinstance(record, FunctionCallTaskRecord) else record.get("metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _record_env(record: FunctionCallTaskRecord | Mapping[str, Any]) -> dict[str, Any]:
    env = record.env if isinstance(record, FunctionCallTaskRecord) else record.get("env")
    return dict(env) if isinstance(env, Mapping) else {}


def _record_task_id(record: FunctionCallTaskRecord | Mapping[str, Any]) -> str:
    return record.task_id if isinstance(record, FunctionCallTaskRecord) else str(record.get("task_id") or "")


def _record_instruction(record: FunctionCallTaskRecord | Mapping[str, Any]) -> str:
    return record.instruction if isinstance(record, FunctionCallTaskRecord) else str(record.get("instruction") or "")


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _official_row_documents(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("gold_docs", "evidence_docs", "negative_docs"):
        for document in _list_of_dicts(item.get(key)):
            docid = str(document.get("docid") or document.get("id") or "")
            if docid and docid in seen:
                continue
            if docid:
                seen.add(docid)
            documents.append(document)
    return documents


def _load_documents_for_query(source_path: Path, query_id: str) -> list[dict[str, Any]]:
    if not source_path.exists():
        return []
    with source_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, Mapping) and str(item.get("query_id") or item.get("id") or "") == str(query_id):
                return _official_row_documents(item)
    return []


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


class _PyseriniBM25Searcher:
    def __init__(self, index_path: Path) -> None:
        self.index_path = require_browsecomp_plus_bm25_index(index_path)
        try:
            _ensure_pyserini_java_home()
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as exc:
            raise RuntimeError(
                "BrowseComp-Plus BM25 search requires pyserini. Install the official "
                "BrowseComp-Plus retrieval dependency, e.g. `uv add pyserini>=1.2.0`."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "BrowseComp-Plus BM25 search failed to initialize pyserini. Ensure Java 21+ "
                "is installed and set RWKV_BROWSECOMP_PLUS_JAVA_HOME or JAVA_HOME if needed."
            ) from exc
        try:
            self.searcher = LuceneSearcher(str(self.index_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to initialize BrowseComp-Plus BM25 index: {self.index_path}") from exc

    def search(self, query: str, k: int) -> list[tuple[dict[str, Any], float]]:
        hits = self.searcher.search(query, k)
        results: list[tuple[dict[str, Any], float]] = []
        for hit in hits:
            raw = hit.lucene_document.get("raw")
            contents = _raw_lucene_contents(raw)
            results.append(({"docid": str(hit.docid), "text": contents}, float(hit.score)))
        return results

    def get_document(self, docid: str) -> dict[str, Any] | None:
        document = self.searcher.doc(docid)
        if document is None:
            return None
        return {"docid": str(docid), "text": _raw_lucene_contents(document.raw())}


_BM25_SEARCHER_CACHE: dict[Path, _PyseriniBM25Searcher] = {}


def _get_pyserini_bm25_searcher(index_path: Path) -> _PyseriniBM25Searcher:
    resolved = require_browsecomp_plus_bm25_index(index_path)
    searcher = _BM25_SEARCHER_CACHE.get(resolved)
    if searcher is None:
        searcher = _PyseriniBM25Searcher(resolved)
        _BM25_SEARCHER_CACHE[resolved] = searcher
    return searcher


def _ensure_pyserini_java_home() -> None:
    override = os.environ.get("RWKV_BROWSECOMP_PLUS_JAVA_HOME") or os.environ.get("BROWSECOMP_PLUS_JAVA_HOME")
    if override:
        os.environ["JAVA_HOME"] = str(Path(override).expanduser().resolve())
        return
    current = os.environ.get("JAVA_HOME")
    if current and (_java_home_major(Path(current)) or 0) >= 21:
        return
    for candidate in _java_home_candidates():
        if (_java_home_major(candidate) or 0) >= 21 and (candidate / "lib" / "server" / "libjvm.so").exists():
            os.environ["JAVA_HOME"] = str(candidate)
            return


def _java_home_candidates() -> tuple[Path, ...]:
    return tuple(
        Path(f"/usr/lib/jvm/java-{version}-openjdk-amd64")
        for version in range(25, 20, -1)
    )


def _java_home_major(path: Path) -> int | None:
    release_path = path.expanduser() / "release"
    if not release_path.exists():
        return None
    try:
        content = release_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'JAVA_VERSION="(\d+)', content)
    return int(match.group(1)) if match else None


def _raw_lucene_contents(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, Mapping):
        value = parsed.get("contents") or parsed.get("text")
        if isinstance(value, str):
            return value
    return raw


__all__ = [
    "BROWSECOMP_PLUS_FINAL_SCHEMA",
    "BROWSECOMP_PLUS_GET_DOCUMENT_SCHEMA",
    "BROWSECOMP_PLUS_SEARCH_SCHEMA",
    "BROWSECOMP_PLUS_TOOL_SCHEMAS",
    "BrowseCompPlusAdapterConfig",
    "BrowseCompPlusEnv",
    "OFFICIAL_BROWSECOMP_PLUS_SOURCE",
    "browsecomp_plus_index_path",
    "browsecomp_plus_run_from_agent_details",
    "create_browsecomp_plus_env",
    "load_browsecomp_plus_rows_from_decrypted_jsonl",
    "require_browsecomp_plus_assets",
    "require_browsecomp_plus_bm25_index",
]
