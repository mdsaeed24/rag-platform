"""GraphRAG evidence checks, with opt-in guarded live answer evaluation."""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import auth
import llm
import main
from authorization import authorize_results
from graph_rag import graph_context, plan_graph_query, traverse_graph
from query_router import QueryRouter
from semantic_cache import SemanticAnswerCache
from eval.graph_answer_checks import answer_checks, SALARY_VALUES

CASES = [
    ("acme_salary_leave", "bob", "What is the software engineer salary and annual leave policy?", "graph", ("$110,000", "20 days", "Full-time")),
    ("globex_salary_leave", "carol", "What is the software engineer salary and annual leave policy?", "graph", ("$125,000", "25 days", "Full-time")),
    ("acme_salary_remote_expenses", "bob", "Engineering manager salary, remote work, and expense approval policy?", "graph", ("$150,000", "three days", "$500")),
    ("acme_salary_remote_hours", "bob", "CEO salary and remote availability hours?", "graph", ("$240,000", "10:00 AM", "4:00 PM")),
    ("globex_missing_remote_hours", "carol", "CEO salary and remote availability hours?", "abstain", ()),
    ("employee_salary_join", "alice", "Software engineer salary and annual leave policy?", "abstain", ()),
    ("employee_earnings_join", "alice", "What does the software engineer earn and what is the remote policy?", "abstain", ()),
    ("foreign_tenant_join", "bob", "Globex software engineer salary and annual leave?", "abstain", ()),
    ("unknown_role_join", "bob", "CFO salary and annual leave policy?", "abstain", ()),
    ("multiple_roles_with_policy", "bob", "CEO and software engineer salaries and annual leave?", "graph", ("$240,000", "$110,000", "20 days")),
    ("acme_salary_comparison", "bob", "Compare CEO and software engineer salaries.", "graph", ("$240,000", "$110,000")),
    ("globex_salary_comparison", "carol", "Compare CEO and software engineer salaries.", "graph", ("$310,000", "$125,000")),
    ("all_roles_with_policies", "bob", "Compare CEO, engineering manager, and software engineer salaries, annual leave and remote work.", "graph", ("$240,000", "$150,000", "$110,000", "20 days", "three days")),
    ("employee_comparison", "alice", "Compare CEO and software engineer salaries.", "abstain", ()),
    ("foreign_tenant_comparison", "bob", "Compare Acme and Globex CEO and software engineer salaries.", "abstain", ()),
    ("earnings_comparison", "bob", "What do the CEO and software engineer earn?", "graph", ("$240,000", "$110,000")),
]

# Independent expected source/edge counts; do not infer them from the planner
# under test. Salary-only comparisons must not cite the handbook.
GRAPH_EXPECTATIONS = {
    "acme_salary_leave": (2, 3),
    "globex_salary_leave": (2, 3),
    "acme_salary_remote_expenses": (2, 4),
    "acme_salary_remote_hours": (2, 4),
    "multiple_roles_with_policy": (2, 5),
    "acme_salary_comparison": (1, 4),
    "globex_salary_comparison": (1, 4),
    "all_roles_with_policies": (2, 8),
    "earnings_comparison": (1, 4),
}
SMOKE_CASE_IDS = {"acme_salary_leave", "globex_salary_comparison", "employee_comparison"}


def run_case(client, case, token, *, live=False):
    case_id, username, question, expected_route, required = case
    identity = auth.verify_access_token(token)
    if identity is None:
        raise ValueError("Evaluation requires a current authenticated identity")
    captured = {"expanded": [], "evidence_trace": [], "completion_attempts": 0, "completion_calls": 0,
                "outbound_context_checks_passed": None}
    actual_expand = main.load_graph_documents
    completion_client = llm.client.with_options(timeout=30.0, max_retries=0) if live else llm.client
    actual_complete = completion_client.chat.completions.create

    def capture_expansion(tenant_id, role):
        points = actual_expand(tenant_id, role)
        try:
            authorize_results(points, identity["tenant_id"], identity["role"])
        except Exception:
            captured["outbound_context_checks_passed"] = False
            raise
        captured["expanded"] = points
        return points

    def guarded_completion(**kwargs):
        # Every check runs before calling the real endpoint. The evaluator
        # independently rejects unsafe batches even if API checks regress.
        captured["completion_attempts"] += 1
        captured["outbound_context_checks_passed"] = False
        if expected_route != "graph" or auth.verify_access_token(token) is None:
            raise AssertionError("Completion not permitted for this case")
        expanded = captured["expanded"]
        authorize_results(expanded, identity["tenant_id"], identity["role"])
        prompt = kwargs["messages"][1]["content"]
        context = prompt.split("\nContext:\n", 1)[1].split("\n\nQuestion:\n", 1)[0]
        expected = graph_context(plan_graph_query(question), expanded, tenant_id=identity["tenant_id"], role=identity["role"])
        if context != expected or not all(value in context for value in required):
            raise AssertionError("Graph context differs from authorized expected evidence")
        if {value for value in SALARY_VALUES if value in context} != SALARY_VALUES.intersection(required):
            raise AssertionError("Graph context includes missing or unrelated salary evidence")
        foreign = "globex" if identity["tenant_id"] == "acme" else "acme"
        if f"data/{foreign}/" in context or f"organization:{foreign}" in context:
            raise AssertionError("Cross-tenant graph evidence")
        evidence = traverse_graph(plan_graph_query(question), expanded, tenant_id=identity["tenant_id"], role=identity["role"])
        captured["evidence_trace"] = [asdict(edge) for edge in evidence.edges]
        captured["outbound_context_checks_passed"] = True
        captured["completion_calls"] += 1
        if live:
            return actual_complete(**kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[Mocked graph answer]"))])

    row = {"id": case_id, "user_id": username, "question": question}
    start = perf_counter()
    try:
        with patch.object(main, "load_graph_documents", side_effect=capture_expansion), patch.object(llm, "client", completion_client), patch.object(completion_client.chat.completions, "create", side_effect=guarded_completion):
            response = client.post("/ask", headers={"Authorization": "Bearer " + token}, json={"question": question})
        data = response.json()
        checks = {"expected_status": response.status_code == 200,
                  "expected_route": data.get("route") == expected_route,
                  "cache_bypassed": data.get("cache_status") == "bypass",
                  "expected_completion_count": captured["completion_calls"] == int(expected_route == "graph"),
                  "verified_identity_preserved": all(data.get(key) == identity[key] for key in ("user_id", "tenant_id", "role"))}
        if expected_route == "abstain":
            checks["no_answer_sources"] = data.get("sources") == []
            checks["no_graph_evidence"] = data.get("graph_evidence") == []
        else:
            source_count, edge_count = GRAPH_EXPECTATIONS[case_id]
            expected_sources = {f"data/{identity['tenant_id']}/salaries.txt"}
            if source_count == 2:
                expected_sources.add(f"data/{identity['tenant_id']}/handbook.txt")
            sources = data.get("sources", [])
            checks["expected_source_citations"] = (len(sources) == source_count
                and {item["source"] for item in sources} == expected_sources
                and all(item["chunk_id"] == 0 for item in sources))
            checks["expected_graph_edges"] = data.get("graph_edge_count") == edge_count
            checks["outbound_context_authorized"] = captured["outbound_context_checks_passed"] is True
            checks["graph_evidence_matches_outbound_context"] = data.get("graph_evidence") == captured["evidence_trace"]
        if live or expected_route == "abstain":
            checks.update(answer_checks(required, data.get("answer"), data.get("sources", []), identity, abstain=expected_route == "abstain"))
            # Store answers for inspection, never raw contexts, tokens or keys.
            row["answer"] = data.get("answer")
        row.update(http_status=response.status_code, route=data.get("route"), graph_edge_count=data.get("graph_edge_count"),
                   sources=data.get("sources", []), checks=checks, passed=all(checks.values()))
    except Exception as exc:
        # Upstream exception strings can contain request details or secrets.
        row.update(passed=False, error_type=type(exc).__name__)
        if getattr(exc, "status_code", None) is not None:
            row["upstream_http_status"] = exc.status_code
    row.update(completion_attempts=captured["completion_attempts"], completion_calls=captured["completion_calls"],
               outbound_context_checks_passed=captured["outbound_context_checks_passed"],
               expanded_chunk_count=len(captured["expanded"]), elapsed_seconds=round(perf_counter() - start, 6))
    return row


def evaluate(*, live=False, smoke=False):
    rows = []
    cases = [case for case in CASES if not smoke or case[0] in SMOKE_CASE_IDS]
    with TestClient(main.app) as client, patch.object(main, "graph_rag_enabled", True), patch.object(main, "query_router", QueryRouter()), patch.object(main, "answer_cache", SemanticAnswerCache(enabled=False)):
        tokens = {}
        for username in sorted({case[1] for case in cases}):
            response = client.post("/login", json={"username": username, "password": username + "123"})
            response.raise_for_status()
            tokens[username] = response.json()["access_token"]
        for case in cases:
            row = run_case(client, case, tokens[case[1]], live=live)
            rows.append(row)
            print(json.dumps({key: row.get(key) for key in ("id", "passed", "http_status", "completion_calls", "elapsed_seconds", "error_type")}), flush=True)
            if row["outbound_context_checks_passed"] is False:
                break  # Stop on a failed safety guard; do not send later cases.
    accepted = [row["elapsed_seconds"] for row in rows if row.get("http_status") == 200]
    return {"timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "live DeepSeek; real embeddings and persisted Qdrant" if live else "real MiniLM embeddings and persisted Qdrant; mocked completion endpoint",
            "suite": "smoke" if smoke else "full", "live_answer_correctness_evaluated": live and any(row.get("route") == "graph" and "answer" in row for row in rows),
            "passed": sum(row["passed"] for row in rows), "total": len(cases), "executed": len(rows),
            "completion_calls": sum(row["completion_calls"] for row in rows),
            "abstentions": sum(row.get("route") == "abstain" for row in rows),
            "unsafe_outbound_context_detected": sum(row["outbound_context_checks_passed"] is False for row in rows),
            "accepted_request_median_seconds": round(statistics.median(accepted), 6) if accepted else None,
            "limitations": "Small four-chunk fixture corpus. Live checks test expected fact/citation presence and known salary leakage, not general semantic correctness, arithmetic, or production load. Latency excludes login/model startup.",
            "checks": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send checked authorized graph context to DeepSeek (30s timeout, no retries)")
    parser.add_argument("--smoke", action="store_true", help="Three cases: two HR generations and one employee abstention")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(live=args.live, smoke=args.smoke)
    filename = "graph" + ("_live" if args.live else "") + ("_smoke" if args.smoke else "") + "_results.json"
    output = args.output or Path(__file__).with_name(filename)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "total": report["total"], "report": str(output)}))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
