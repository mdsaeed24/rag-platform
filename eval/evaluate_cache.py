"""Offline cache integration checks using real embeddings and persisted vectors.

Only the completion endpoint is mocked. This measures routing and avoided model
calls, not live answer correctness or real DeepSeek latency.
"""

import copy
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import auth
import llm
import main
from acl_config import DOCUMENT_ACLS
from semantic_cache import SemanticAnswerCache


CASES = [
    ("bob_initial", "bob", "What does the CEO earn?", "miss", None),
    ("bob_repeat", "bob", "What does the CEO earn?", "exact_hit", None),
    ("bob_paraphrase", "bob", "How much does the CEO earn?", "semantic_hit", None),
    ("carol_isolated", "carol", "What does the CEO earn?", "miss", None),
    ("carol_repeat", "carol", "What does the CEO earn?", "exact_hit", None),
    ("alice_isolated", "alice", "What does the CEO earn?", "bypass", None),
    ("alice_unknown_not_cached", "alice", "What does the CEO earn?", "bypass", None),
    ("bob_foreign_company", "bob", "What does the Globex CEO earn?", "bypass", None),
    ("bob_different_job", "bob", "What does the engineering manager earn?", "miss", None),
    ("bob_negation", "bob", "What does the CEO not earn?", "miss", None),
    ("bob_monthly", "bob", "What does the CEO earn monthly?", "miss", None),
    ("bob_low_similarity", "bob", "What is the CEO salary?", "miss", None),
    ("bob_document_revoked", "bob", "What does the CEO earn?", None, "document"),
    ("bob_user_revoked", "bob", "What does the CEO earn?", None, "user"),
    ("bob_new_token_version", "bob", "What does the CEO earn?", "miss", "version"),
]


def mock_completion(*args, **kwargs):
    prompt = kwargs["messages"][1]["content"]
    context = prompt.split("\nContext:\n", 1)[1].split("\nQuestion:\n", 1)[0]
    question = prompt.split("\nQuestion:\n", 1)[1].casefold()
    source = "data/acme/salaries.txt" if "data/acme/salaries.txt" in context else "data/globex/salaries.txt"
    unknown = "salaries.txt" not in context or " not " in question
    unknown |= "globex" in question and "data/globex/" not in context
    unknown |= "acme" in question and "data/acme/" not in context
    answer = "I don't know based on the provided documents." if unknown else f"[Mocked salary answer] Source: {source}, Chunk: 0"
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])


def evaluate():
    rows = []
    cache = SemanticAnswerCache()
    with TestClient(main.app) as client, patch.object(main, "answer_cache", cache), patch.object(llm.client.chat.completions, "create", side_effect=mock_completion) as completion:
        tokens = {}
        for username in ("bob", "alice", "carol"):
            response = client.post("/login", json={"username": username, "password": username + "123"})
            if response.status_code != 200:
                raise RuntimeError("Demo login failed")
            tokens[username] = response.json()["access_token"]
        for case_id, username, question, expected_cache, change in CASES:
            with ExitStack() as stack:
                token = tokens[username]
                if change == "document":
                    revised = copy.deepcopy(DOCUMENT_ACLS)
                    del revised["data/acme/salaries.txt"]
                    stack.enter_context(patch.dict(DOCUMENT_ACLS, revised, clear=True))
                elif change in {"user", "version"}:
                    revised = copy.deepcopy(auth.USERS)
                    if change == "user":
                        revised[username]["active"] = False
                    else:
                        revised[username]["token_version"] += 1
                    stack.enter_context(patch.dict(auth.USERS, revised, clear=True))
                    if change == "version":
                        user = auth.USERS[username]
                        token = auth.create_access_token(username, user["tenant_id"], user["role"])
                calls_before = completion.call_count
                start = perf_counter()
                response = client.post("/ask", headers={"Authorization": "Bearer " + token}, json={"question": question})
                elapsed = perf_counter() - start
                data = response.json()
                expected_status = 403 if change == "document" else 401 if change == "user" else 200
                model_calls = completion.call_count - calls_before
                expected_calls = int(expected_status == 200 and expected_cache == "miss")
                passed = response.status_code == expected_status and data.get("cache_status") == expected_cache and model_calls == expected_calls
                row = {"id": case_id, "user_id": username, "question": question, "http_status": response.status_code, "cache_status": data.get("cache_status"), "cache_similarity": data.get("cache_similarity"), "completion_calls": model_calls, "elapsed_seconds": round(elapsed, 6), "passed": passed}
                if response.status_code == 200:
                    row["sources"] = data["sources"]
                rows.append(row)
                print(json.dumps(row), flush=True)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "real MiniLM embeddings and persisted Qdrant; mocked completion endpoint",
        "live_answer_correctness_evaluated": False,
        "passed": sum(row["passed"] for row in rows), "total": len(rows),
        "cache_hits": sum(row["cache_status"] in {"exact_hit", "semantic_hit"} for row in rows),
        "completion_calls": sum(row["completion_calls"] for row in rows),
        "completion_calls_avoided_by_cache": sum(row["cache_status"] in {"exact_hit", "semantic_hit"} and row["completion_calls"] == 0 for row in rows),
        "latency_scope": "In-process /ask with mocked completions; excludes login/model startup. No live speedup claim.",
        "checks": rows,
    }


if __name__ == "__main__":
    report = evaluate()
    output = Path(__file__).with_name("cache_results.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "total": report["total"], "report": str(output)}))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
