"""Local JSON-lines audit trail. Never include tokens, passwords, or chunks."""

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

_logger = logging.getLogger("rag.audit")
_logger.setLevel(logging.INFO)
_logger.propagate = False
_lock = Lock()


def document_id(result):
    payload = result.payload if isinstance(result.payload, dict) else {}
    return {
        "point_id": str(result.id),
        "source": payload.get("source"),
        "chunk_id": payload.get("chunk_id"),
    }


def log_event(*, decision, reason, identity=None, query=None, results=(), **timings):
    identity = identity or {}
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": identity.get("user_id"),
        "tenant_id": identity.get("tenant_id"),
        "role": identity.get("role"),
        "query": query,
        "retrieved_document_ids": [document_id(result) for result in results],
        "authorization_decision": decision,
        "reason": reason,
        **timings,
    }
    with _lock:
        if not _logger.handlers:
            log_dir = Path(__file__).resolve().parent / "logs"
            log_dir.mkdir(exist_ok=True, mode=0o700)
            handler = RotatingFileHandler(
                log_dir / "audit.jsonl", maxBytes=5_000_000, backupCount=3,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            _logger.addHandler(handler)
    _logger.info(json.dumps(event, ensure_ascii=True))
