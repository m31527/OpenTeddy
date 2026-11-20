"""
OpenTeddy knowledge-base search (knowledge.py).

The entries that match the person's words are already in the task's
prompt; this is for looking further — another wording, a detail of a
product, a second question in the same request. Only a task that has
datasets sees it (the executor hides it otherwise).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import knowledge as K
from tool_registry import RiskLevel, make_result


async def knowledge_search(query: str, dataset: str = "", k: int = 5) -> Dict[str, Any]:
    t0 = time.monotonic()
    ms = lambda: int((time.monotonic() - t0) * 1000)  # noqa: E731
    ids = list(K.current())
    if not ids:
        return make_result(False, error="This task has no knowledge-base datasets.", duration_ms=ms())
    if (dataset or "").strip():
        d = await K.get_dataset(dataset)
        if not d or d["id"] not in ids:
            names = [x["name"] for x in await K.list_datasets() if x["id"] in ids]
            return make_result(False, error=f"No dataset '{dataset}' in this task; it has: {', '.join(names)}",
                               duration_ms=ms())
        ids = [d["id"]]
    try:
        hits = await K.search(query or "", ids, k=max(1, min(int(k or 5), 10)), loose=True)
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=f"{type(exc).__name__}: {exc}", duration_ms=ms())
    if not hits:
        return make_result(True, result=f"Nothing in the knowledge base matches 「{query}」. Try other words once if "
                                        "the question could be worded differently; otherwise say the knowledge base "
                                        "doesn't cover it — don't answer the owner's policies from general practice.",
                           duration_ms=ms())
    return make_result(True, result=("Knowledge-base entries (the owner's own reference — answer from them, "
                                     "name the entry):\n" + K.format_hits(hits, limit=6000) + "\n" + K.RULE),
                       duration_ms=ms())


_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "knowledge_search",
        "description": (
            "Search the owner's knowledge base (FAQ, product notes, policies uploaded to OpenTeddy) for "
            "entries about a question. Answers about the owner's own business come from here, not from "
            "the web or your own knowledge. Entries matching the request are already in the prompt; use "
            "this for other wordings or further details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or keywords, in the person's language."},
                "dataset": {"type": "string", "description": "Only this dataset (name or id). Empty = all of the task's."},
                "k": {"type": "integer", "description": "How many entries (1–10). Default 5."},
            },
            "required": ["query"],
        },
    },
}

KNOWLEDGE_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (knowledge_search, _SCHEMA, "low"),
]
