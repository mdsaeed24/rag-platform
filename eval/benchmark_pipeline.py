"""Sequential in-process latency checks; real retrieval, mocked completions only."""

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import secrets
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# id, user, question, optional cache seed, cache enabled, route, cache status,
# HTTP status, generation calls, retrieval calls, graph expansion calls
CASES = (
    ("rag_fresh", "bob", "What does the CEO earn?", None, False, "rag", "bypass", 200, 1, 1, 0),
    ("rag_exact", "bob", "What does the CEO earn?", "What does the CEO earn?", True, "cache", "exact_hit", 200, 0, 1, 0),
    ("rag_semantic", "bob", "How much does the CEO earn?", "What does the CEO earn?", True, "cache", "semantic_hit", 200, 0, 1, 0),
    ("graph_fresh", "bob", "Compare CEO and software engineer salaries.", None, False, "graph", "bypass", 200, 1, 1, 1),
    ("graph_exact", "bob", "Compare CEO and software engineer salaries.", "Compare CEO and software engineer salaries.", True, "graph_cache", "exact_hit", 200, 0, 1, 1),
    ("graph_semantic", "bob", "Compare the CEO and software engineer salaries.", "Compare CEO and software engineer salaries.", True, "graph_cache", "semantic_hit", 200, 0, 1, 1),
    ("employee_abstain", "alice", "What does the CEO earn?", None, True, "abstain", "bypass", 200, 0, 1, 0),
    ("foreign_tenant_abstain", "bob", "What does the Globex CEO earn?", None, True, "abstain", "bypass", 200, 0, 1, 0),
    ("incomplete_graph_abstain", "carol", "CEO salary and remote availability hours?", None, True, "abstain", "bypass", 200, 0, 1, 1),
    ("invalid_token", None, "What does the CEO earn?", None, True, None, None, 401, 0, 0, 0),
)


def latency_summary(samples):
    if not samples or any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("Latency samples must be nonempty, finite, and nonnegative")
    ordered = sorted(samples)
    return {"samples": len(samples), "median_ms": round(statistics.median(ordered), 6),
            "p95_ms": round(ordered[math.ceil(len(ordered) * .95) - 1], 6)}


def benchmark(iterations=30, warmup=5):
    if iterations < 1 or warmup < 0:
        raise ValueError("Invalid benchmark iteration count")
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient
    import auth
    import llm
    import main
    import search
    from acl_config import DOCUMENT_ACLS
    from authorization import authorize_results
    from graph_rag import graph_context, plan_graph_query
    from query_router import QueryRouter
    from semantic_cache import SemanticAnswerCache

    client_storage = QdrantClient(path=str(search.QDRANT_PATH))
    try:
        point_count = client_storage.get_collection(search.COLLECTION_NAME).points_count
    finally:
        client_storage.close()
    original_retrieve, original_expand = main.search_documents, main.load_graph_documents
    active = {}

    def retrieve(*args, **kwargs):
        active["retrieval_calls"] += 1
        points = original_retrieve(*args, **kwargs)
        active["retrieved"] = points
        return points

    def expand(*args, **kwargs):
        active["graph_expansion_calls"] += 1
        points = original_expand(*args, **kwargs)
        active["expanded"] = points
        return points

    def complete(**kwargs):
        active["completion_calls"] += 1
        active["outbound_context_authorized"] = False
        identity = auth.verify_access_token(active["token"])
        if identity is None:
            raise AssertionError("Completion for an invalid identity")
        tenant, role = identity["tenant_id"], identity["role"]
        points = active["expanded"] if active["graph_expansion_calls"] else active["retrieved"]
        authorize_results(points, tenant, role)
        context = kwargs["messages"][1]["content"].split("\nContext:\n", 1)[1].split("\n\nQuestion:\n", 1)[0]
        if active["graph_expansion_calls"]:
            expected = graph_context(plan_graph_query(active["question"]), points, tenant_id=tenant, role=role)
        else:
            expected = "\n\n".join(f"[Source: {point.payload['source']}, Chunk: {point.payload['chunk_id']}]\n{point.payload['text']}" for point in points)
        if context != expected:
            raise AssertionError("Completion context differs from authorized evidence")
        active["outbound_context_authorized"] = True
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content="Offline mocked benchmark answer"))])

    states = [{"case": case, "cache": SemanticAnswerCache(enabled=case[4]), "rows": []} for case in CASES]
    with ExitStack() as stack:
        client = stack.enter_context(TestClient(main.app))
        stack.enter_context(patch.object(main, "query_router", QueryRouter()))
        stack.enter_context(patch.object(main, "graph_rag_enabled", True))
        stack.enter_context(patch.object(main, "graph_answer_cache_enabled", True))
        stack.enter_context(patch.object(main, "search_documents", side_effect=retrieve))
        stack.enter_context(patch.object(main, "load_graph_documents", side_effect=expand))
        stack.enter_context(patch.object(main, "log_event"))
        stack.enter_context(patch.object(llm.client.chat.completions, "create", side_effect=complete))
        tokens = {}
        for username in ("alice", "bob", "carol"):
            response = client.post("/login", json={"username": username, "password": username + "123"})
            response.raise_for_status()
            tokens[username] = response.json()["access_token"]

        def request(state, *, seed=False):
            case_id, username, question, seed_question, _, route, cache_status, status, calls, searches, expansions = state["case"]
            token = tokens[username] if username else "invalid"
            active.clear()
            active.update(token=token, question=seed_question if seed else question, retrieved=[], expanded=[],
                          completion_calls=0, retrieval_calls=0, graph_expansion_calls=0,
                          outbound_context_authorized=None)
            with patch.object(main, "answer_cache", state["cache"]):
                start = perf_counter()
                response = client.post("/ask", json={"question": active["question"]},
                                       headers={"Authorization": "Bearer " + token})
                elapsed_ms = (perf_counter() - start) * 1000
            data = response.json()
            identity = auth.verify_access_token(token)
            sources_valid = all(
                identity is not None and source.get("source") in DOCUMENT_ACLS
                and DOCUMENT_ACLS[source["source"]]["tenant_id"] == identity["tenant_id"]
                and identity["role"] in DOCUMENT_ACLS[source["source"]]["allowed_roles"]
                for source in data.get("sources", [])
            )
            checks = {"expected_status": response.status_code == (200 if seed else status),
                      "expected_route": data.get("route") == (("graph" if expansions else "rag") if seed else route),
                      "expected_cache_status": data.get("cache_status") == ("miss" if seed else cache_status),
                      "expected_completion_calls": active["completion_calls"] == (1 if seed else calls),
                      "retrieval_still_runs": active["retrieval_calls"] == searches,
                      "graph_expansion_still_runs": active["graph_expansion_calls"] == expansions,
                      "response_sources_authorized": sources_valid,
                      "outbound_context_safe": active["outbound_context_authorized"] is not False}
            return {"elapsed_ms": round(elapsed_ms, 6), "status": response.status_code,
                    "route": data.get("route"), "cache_status": data.get("cache_status"),
                    "mock_completion_calls": active["completion_calls"],
                    "retrieval_calls": active["retrieval_calls"], "graph_expansion_calls": active["graph_expansion_calls"],
                    "context_guard_exercised": active["outbound_context_authorized"] is not None,
                    "checks": checks, "passed": all(checks.values())}

        for state in states:
            if state["case"][3]:
                if not request(state, seed=True)["passed"]:
                    raise AssertionError("Benchmark cache seed failed")
        # Alternate forward/reverse round-robin order. Each case has a separate
        # cache; seeds and warmup requests are excluded from measured statistics.
        for trial in range(warmup + iterations):
            for state in states if trial % 2 else reversed(states):
                row = request(state)
                if not row["passed"]:
                    raise AssertionError("Benchmark request checks failed")
                if trial >= warmup:
                    state["rows"].append(row)
    scenarios = [{"id": state["case"][0], "passed": all(row["passed"] for row in state["rows"]),
                  **latency_summary([row["elapsed_ms"] for row in state["rows"]]),
                  "mock_completion_calls": sum(row["mock_completion_calls"] for row in state["rows"]),
                  "retrieval_calls": sum(row["retrieval_calls"] for row in state["rows"]),
                  "graph_expansion_calls": sum(row["graph_expansion_calls"] for row in state["rows"]),
                  "measurements": state["rows"]} for state in states]
    return {"mode": "offline; real cached MiniLM and persisted Qdrant; mocked completions",
            "point_count": point_count, "iterations_per_scenario": iterations, "warmup_per_scenario": warmup,
            "passed": sum(case["passed"] for case in scenarios), "total": len(scenarios),
            "measured_requests": len(scenarios) * iterations, "real_provider_calls": 0,
            "live_answer_correctness_evaluated": False,
            "latency_scope": "Sequential in-process /ask including embedding, storage opening, authorization, routing, graph and cache work. Excludes login, model startup, HTTP transport, seeds and warmup. Completions and audit writes are mocked; not live latency or a load test.",
            "scenarios": scenarios}


def offline_environment():
    return {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "DEEPSEEK_API_KEY": "offline-validation-no-provider-access", "JWT_SECRET_KEY": secrets.token_urlsafe(48),
            "METRICS_TOKEN": "", "ANSWER_CACHE_ENABLED": "true", "GRAPH_RAG_ENABLED": "true",
            "GRAPH_ANSWER_CACHE_ENABLED": "true", "QUERY_ROUTING_ENABLED": "true", "ROUTING_MIN_SCORE": "0.25",
            "MAX_INFLIGHT_ASK": "4", "MAX_INFLIGHT_LOGIN": "4", "MAX_REQUEST_BODY_BYTES": "16384", "LLM_TIMEOUT_SECONDS": "30"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "pipeline_benchmark.json")
    args = parser.parse_args()
    with patch.dict(os.environ, offline_environment()), patch("socket.socket.connect", side_effect=AssertionError("Offline network guard")), patch("socket.socket.connect_ex", side_effect=AssertionError("Offline network guard")):
        report = benchmark(args.iterations, args.warmup)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "total", "measured_requests", "real_provider_calls")}))
