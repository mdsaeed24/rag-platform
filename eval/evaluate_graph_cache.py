"""Offline graph cache checks with real retrieval and mocked completions only."""

import copy
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import auth
import graph_rag
import llm
import main
from acl_config import DOCUMENT_ACLS
from authorization import authorize_results
from graph_rag import graph_context, plan_graph_query, traverse_graph
from query_router import QueryRouter
from semantic_cache import SemanticAnswerCache

QUESTION = "Compare CEO and software engineer salaries."
CASES = [
    ("bob_initial", "bob", QUESTION, "miss", None),
    ("bob_repeat", "bob", QUESTION, "exact_hit", None),
    ("bob_paraphrase", "bob", "Compare the CEO and software engineer salaries.", "semantic_hit", None),
    ("carol_isolated", "carol", QUESTION, "miss", None),
    ("carol_repeat", "carol", QUESTION, "exact_hit", None),
    ("employee_denied", "alice", QUESTION, "bypass", None),
    ("foreign_tenant_denied", "bob", "Compare Acme and Globex CEO and software engineer salaries.", "bypass", None),
    ("different_job", "bob", "Compare CEO and engineering manager salaries.", "miss", None),
    ("added_policy", "bob", QUESTION + " Include annual leave.", "miss", None),
    ("document_revoked", "bob", QUESTION, None, "document"),
    ("user_revoked", "bob", QUESTION, None, "user"),
    ("new_token_version", "bob", QUESTION, "miss", "version"),
    ("new_graph_version", "bob", QUESTION, "miss", "graph_version"),
    ("graph_cache_disabled", "bob", QUESTION, "bypass", "disabled"),
]


def evaluate():
    rows = []
    original_expand = main.load_graph_documents
    with TestClient(main.app) as client, patch.object(main, "answer_cache", SemanticAnswerCache()), patch.object(main, "graph_rag_enabled", True), patch.object(main, "graph_answer_cache_enabled", True), patch.object(main, "query_router", QueryRouter()):
        tokens = {}
        for username in ("bob", "carol", "alice"):
            response = client.post("/login", json={"username": username, "password": username + "123"})
            response.raise_for_status()
            tokens[username] = response.json()["access_token"]
        for case_id, username, question, expected_cache, change in CASES:
            expanded = []
            token = tokens[username]
            with ExitStack() as stack:
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
                elif change == "graph_version":
                    stack.enter_context(patch.object(graph_rag, "GRAPH_ANSWER_VERSION", "graph-answer:evaluation-v2"))
                elif change == "disabled":
                    stack.enter_context(patch.object(main, "graph_answer_cache_enabled", False))

                def capture_expansion(tenant_id, role):
                    points = original_expand(tenant_id, role)
                    authorize_results(points, tenant_id, role)
                    expanded.extend(points)
                    return points

                def guarded_mock_completion(**kwargs):
                    identity = auth.verify_access_token(token)
                    if identity is None:
                        raise AssertionError("Completion for invalid identity")
                    authorize_results(expanded, identity["tenant_id"], identity["role"])
                    prompt = kwargs["messages"][1]["content"]
                    context = prompt.split("\nContext:\n", 1)[1].split("\n\nQuestion:\n", 1)[0]
                    expected = graph_context(plan_graph_query(question), expanded, tenant_id=identity["tenant_id"], role=identity["role"])
                    if context != expected:
                        raise AssertionError("Outgoing graph context differs from authorized evidence")
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[Mocked graph answer; correctness not evaluated]"))])

                completion = stack.enter_context(patch.object(llm.client.chat.completions, "create", side_effect=guarded_mock_completion))
                stack.enter_context(patch.object(main, "load_graph_documents", side_effect=capture_expansion))
                start = perf_counter()
                response = client.post("/ask", json={"question": question}, headers={"Authorization": "Bearer " + token})
                data = response.json()
                expected_status = 403 if change == "document" else 401 if change == "user" else 200
                abstain = case_id in {"employee_denied", "foreign_tenant_denied"}
                expected_route = None if expected_status != 200 else "abstain" if abstain else "graph_cache" if expected_cache in {"exact_hit", "semantic_hit"} else "graph"
                expected_calls = int(expected_route == "graph")
                checks = {"expected_status": response.status_code == expected_status,
                          "expected_cache_status": data.get("cache_status") == expected_cache,
                          "expected_route": data.get("route") == expected_route,
                          "expected_completion_count": completion.call_count == expected_calls}
                if expected_route in {"graph", "graph_cache"}:
                    identity = auth.verify_access_token(token)
                    evidence = traverse_graph(plan_graph_query(question), expanded, tenant_id=identity["tenant_id"], role=identity["role"])
                    checks["fresh_authorized_graph_evidence"] = (len(expanded) > 0 and evidence.complete
                        and data.get("graph_evidence") == [asdict(edge) for edge in evidence.edges])
                elif expected_route == "abstain":
                    checks["no_evidence_returned"] = data.get("graph_evidence") == [] and data.get("sources") == []
                row = {"id": case_id, "user_id": username, "http_status": response.status_code,
                       "route": data.get("route"), "cache_status": data.get("cache_status"), "cache_similarity": data.get("cache_similarity"),
                       "expanded_chunk_count": len(expanded), "completion_calls": completion.call_count,
                       "elapsed_seconds": round(perf_counter() - start, 6), "checks": checks, "passed": all(checks.values())}
                rows.append(row)
                print(json.dumps(row), flush=True)
    return {"timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "real MiniLM embeddings and persisted Qdrant; mocked completions only",
            "live_answer_correctness_evaluated": False,
            "passed": sum(row["passed"] for row in rows), "total": len(rows),
            "completion_calls": sum(row["completion_calls"] for row in rows),
            "completion_calls_avoided_by_cache": sum(row["route"] == "graph_cache" and row["completion_calls"] == 0 for row in rows),
            "limitations": "Four-chunk fixture corpus. Verifies cache isolation and call avoidance, not live answer correctness or production speedup.",
            "checks": rows}


if __name__ == "__main__":
    report = evaluate()
    output = Path(__file__).with_name("graph_cache_results.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "total": report["total"], "report": str(output)}))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
