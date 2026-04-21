"""
OpenTeddy GCP Tool
Google Cloud operations: Storage, BigQuery, Pub/Sub.
Credentials from GOOGLE_APPLICATION_CREDENTIALS env var.
All google-cloud packages are optional — clear ImportError returned if missing.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _import_error(package: str) -> Dict[str, Any]:
    return make_result(
        False,
        error=(
            f"Required package '{package}' is not installed. "
            f"Run: pip install {package}"
        ),
    )


async def _run_in_thread(fn, *args, **kwargs):
    """Run a blocking GCP SDK call in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# ── Storage ───────────────────────────────────────────────────────────────────

async def gcp_storage_list(bucket: str) -> Dict[str, Any]:
    """List blobs in a GCS bucket. LOW risk."""
    start = time.monotonic()
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        return _import_error("google-cloud-storage")
    try:
        def _list():
            client = storage.Client()
            blobs = list(client.list_blobs(bucket))
            return [{"name": b.name, "size": b.size, "updated": str(b.updated)}
                    for b in blobs]
        blobs = await _run_in_thread(_list)
        return make_result(True, result={"bucket": bucket, "blobs": blobs},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def gcp_storage_read(bucket: str, blob: str) -> Dict[str, Any]:
    """Download a blob from GCS as text. LOW risk."""
    start = time.monotonic()
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        return _import_error("google-cloud-storage")
    try:
        def _read():
            client = storage.Client()
            b = client.bucket(bucket).blob(blob)
            return b.download_as_text()
        content = await _run_in_thread(_read)
        return make_result(True, result={"bucket": bucket, "blob": blob,
                                          "content": content},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def gcp_storage_write(bucket: str, blob: str, content: str) -> Dict[str, Any]:
    """Upload text content to a GCS blob. HIGH risk — requires approval."""
    start = time.monotonic()
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        return _import_error("google-cloud-storage")
    try:
        def _write():
            client = storage.Client()
            client.bucket(bucket).blob(blob).upload_from_string(
                content, content_type="text/plain"
            )
        await _run_in_thread(_write)
        return make_result(True, result={"bucket": bucket, "blob": blob,
                                          "bytes_written": len(content.encode())},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=str(exc), duration_ms=_ms(start))


# ── BigQuery ──────────────────────────────────────────────────────────────────

async def gcp_bigquery_query(sql: str) -> Dict[str, Any]:
    """Run a BigQuery SELECT query. LOW risk."""
    start = time.monotonic()
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError:
        return _import_error("google-cloud-bigquery")
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
        return make_result(False,
                           error="gcp_bigquery_query only allows SELECT/WITH queries. "
                                 "Use gcp_bigquery_execute for writes.",
                           duration_ms=_ms(start))
    try:
        def _query():
            client = bigquery.Client()
            job = client.query(sql)
            rows = list(job.result())
            cols = [f.name for f in job.result().schema] if rows else []
            return [dict(zip(cols, row.values())) for row in rows], cols
        rows, cols = await _run_in_thread(_query)
        return make_result(True, result={"columns": cols, "rows": rows,
                                          "row_count": len(rows)},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def gcp_bigquery_execute(sql: str) -> Dict[str, Any]:
    """Execute a BigQuery INSERT/UPDATE/DELETE/DDL. HIGH risk — requires approval."""
    start = time.monotonic()
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError:
        return _import_error("google-cloud-bigquery")
    try:
        def _exec():
            client = bigquery.Client()
            job = client.query(sql)
            job.result()  # wait for completion
            return job.num_dml_affected_rows or 0
        rows_affected = await _run_in_thread(_exec)
        return make_result(True, result={"rows_affected": rows_affected},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=str(exc), duration_ms=_ms(start))


# ── Pub/Sub ───────────────────────────────────────────────────────────────────

async def gcp_pubsub_publish(topic: str, message: str) -> Dict[str, Any]:
    """Publish a message to a Pub/Sub topic. HIGH risk — requires approval."""
    start = time.monotonic()
    try:
        from google.cloud import pubsub_v1  # type: ignore
    except ImportError:
        return _import_error("google-cloud-pubsub")
    try:
        def _publish():
            publisher = pubsub_v1.PublisherClient()
            future = publisher.publish(topic, message.encode("utf-8"))
            return future.result()  # message ID
        message_id = await _run_in_thread(_publish)
        return make_result(True, result={"topic": topic, "message_id": message_id},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=str(exc), duration_ms=_ms(start))


# ── Schemas ───────────────────────────────────────────────────────────────────

_SCHEMA_GCS_LIST: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "gcp_storage_list",
        "description": "List blobs in a Google Cloud Storage bucket.",
        "parameters": {
            "type": "object",
            "properties": {"bucket": {"type": "string"}},
            "required": ["bucket"],
        },
    },
}

_SCHEMA_GCS_READ: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "gcp_storage_read",
        "description": "Download a blob from GCS as text.",
        "parameters": {
            "type": "object",
            "properties": {
                "bucket": {"type": "string"},
                "blob": {"type": "string", "description": "Blob/object path."},
            },
            "required": ["bucket", "blob"],
        },
    },
}

_SCHEMA_GCS_WRITE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "gcp_storage_write",
        "description": "Upload text content to a GCS blob. Requires approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "bucket": {"type": "string"},
                "blob": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["bucket", "blob", "content"],
        },
    },
}

_SCHEMA_BQ_QUERY: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "gcp_bigquery_query",
        "description": "Run a BigQuery SELECT or WITH query.",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
}

_SCHEMA_BQ_EXEC: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "gcp_bigquery_execute",
        "description": "Execute a BigQuery INSERT/UPDATE/DELETE/DDL. Requires approval.",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
}

_SCHEMA_PUBSUB: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "gcp_pubsub_publish",
        "description": "Publish a message to a Pub/Sub topic. Requires approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string",
                          "description": "Full topic path: projects/{proj}/topics/{name}"},
                "message": {"type": "string"},
            },
            "required": ["topic", "message"],
        },
    },
}

# ── Export ─────────────────────────────────────────────────────────────────────

GCP_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (gcp_storage_list,      _SCHEMA_GCS_LIST,  "low"),
    (gcp_storage_read,      _SCHEMA_GCS_READ,  "low"),
    (gcp_storage_write,     _SCHEMA_GCS_WRITE, "high"),
    (gcp_bigquery_query,    _SCHEMA_BQ_QUERY,  "low"),
    (gcp_bigquery_execute,  _SCHEMA_BQ_EXEC,   "high"),
    (gcp_pubsub_publish,    _SCHEMA_PUBSUB,    "high"),
]
