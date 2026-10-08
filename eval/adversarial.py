"""Authenticated authorization checks against existing vectors; offline by default.

--live makes real DeepSeek calls with checked, authorized context. User and ACL
revocation is simulated in this process only; configuration files are untouched.
"""

import argparse
import copy
import json
import re
import statistics
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from jose import jwt

import auth
import llm
import main
from acl_config import DOCUMENT_ACLS
from authorization import authorize_results
from semantic_cache import SemanticAnswerCache

CASES_PATH = Path(__file__).with_name("adversarial_cases.json")


def contains_amount(text, amount):
    """Recognize the fixture's salary in digits, including common k notation.

    This is a fixture-specific answer check, not a general disclosure detector.
    Spelled-out numbers or other encodings require additional evaluation.
    """
    digits = re.sub(r"\D", "", amount)
    numeric = r"[ ,.]?".join(digits)
    patterns = [numeric]
    if digits.endswith("000"):
        patterns.append(re.escape(digits[:-3]) + r"\s*(?:k\b|thousand\b)")
    return bool(re.search(r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)", text, re.IGNORECASE))


def restricted_salary_amounts(identity):
    amounts = set()
    for source, acl in DOCUMENT_ACLS.items():
        if acl["classification"] != "confidential":
            continue
        if acl["tenant_id"] != identity["tenant_id"] or identity["role"] not in acl["allowed_roles"]:
            amounts.update(re.findall(r"\$[\d,]+", (PROJECT_ROOT / source).read_text(encoding="utf-8")))
    return sorted(amounts)


def answer_checks(case, answer, restricted_amounts):
    checks = {"restricted_salary_absent": not any(contains_amount(answer, amount) for amount in restricted_amounts)}
    if case.get("expected_answer"):
        checks["expected_answer_present"] = contains_amount(answer, case["expected_answer"])
        checks["salary_citation_present"] = "salaries.txt" in answer and bool(re.search(r"chunk\s*:?\s*0\b", answer, re.IGNORECASE))
    if case.get("expect_unknown"):
        checks["unknown_answer_present"] = "i don't know based on the provided documents" in answer.lower()
    return checks


def token_for_case(case, token):
    mode = case.get("token_mode", "valid")
    if mode == "missing":
        return None
    if mode == "malformed":
        return "not-a-jwt"
    if mode == "valid":
        return token
    claims = auth.verify_access_token(token).copy()
    key = auth.SECRET_KEY
    if mode == "forged":
        key = "attacker-controlled-key"
    elif mode == "expired":
        claims["exp"] = 1
    elif mode == "role_mismatch":
        claims["role"] = "hr"
    else:
        raise ValueError("Unknown token mode")
    return jwt.encode(claims, key, algorithm=auth.ALGORITHM)


def apply_revocation(stack, case):
    mode = case.get("revocation")
    if mode == "document":
        revised = copy.deepcopy(DOCUMENT_ACLS)
        del revised["data/acme/salaries.txt"]
        stack.enter_context(patch.dict(DOCUMENT_ACLS, revised, clear=True))
    elif mode:
        revised = copy.deepcopy(auth.USERS)
        user = revised[case["username"]]
        if mode == "disabled":
            user["active"] = False
        elif mode == "version":
            user["token_version"] += 1
        elif mode == "role":
            user["role"] = "employee"
        elif mode == "tenant":
            user["tenant_id"] = "globex"
        else:
            raise ValueError("Unknown revocation mode")
        stack.enter_context(patch.dict(auth.USERS, revised, clear=True))


def run_case(client, case, token, *, live=False):
    identity = auth.verify_access_token(token)
    if identity is None:
        raise ValueError("Test identity is invalid")
    restricted_amounts = restricted_salary_amounts(identity)
    request_token = token_for_case(case, token)
    captured = {"results": [], "retrieval_calls": 0, "llm_calls": 0, "context_checks_passed": None}
    original_search = main.search_documents
    row = {"id": case["id"], "user_id": case["username"], "tenant_id": identity["tenant_id"], "role": identity["role"], "question": case["question"], "expected_status": case["expected_status"]}

    def capture_search(*args, **kwargs):
        captured["retrieval_calls"] += 1
        results = original_search(*args, **kwargs)
        captured["results"] = results
        return results

    # Disable retries so errors remain visible and each case has bounded cost.
    completion_client = llm.client.with_options(timeout=30.0, max_retries=0) if live else llm.client
    original_create = completion_client.chat.completions.create

    def checked_completion(*args, **kwargs):
        captured["llm_calls"] += 1
        captured["context_checks_passed"] = False
        results = captured["results"]
        authorize_results(results, identity["tenant_id"], identity["role"])
        expected_context = "\n\n".join(
            f"[Source: {point.payload['source']}, Chunk: {point.payload['chunk_id']}]\n{point.payload['text']}"
            for point in results
        )
        prompt = kwargs["messages"][1]["content"]
        context = prompt.split("\nContext:\n", 1)[1].split("\nQuestion:\n", 1)[0].strip()
        if context != expected_context.strip():
            raise AssertionError("Outbound context differs from checked retrieval")
        if any(contains_amount(context, amount) for amount in restricted_amounts):
            raise AssertionError("Restricted salary appeared in outbound context")
        captured["context_checks_passed"] = True
        if live:
            return original_create(*args, **kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[Mocked completion; live answer not evaluated]"))])

    start = perf_counter()
    try:
        with ExitStack() as stack:
            apply_revocation(stack, case)
            # Exercise fresh generation for every adversarial case. Cache
            # authorization and reuse are covered by their own regression suite.
            stack.enter_context(patch.object(main, "answer_cache", SemanticAnswerCache(enabled=False)))
            stack.enter_context(patch.object(main, "search_documents", side_effect=capture_search))
            stack.enter_context(patch.object(llm, "client", completion_client))
            stack.enter_context(patch.object(completion_client.chat.completions, "create", side_effect=checked_completion))
            headers = dict(case.get("headers", {}))
            if request_token is not None:
                headers["Authorization"] = "Bearer " + request_token
            query_string = case.get("query_string", "")
            response = client.post("/ask" + ("?" + query_string if query_string else ""), headers=headers, json={"question": case["question"], **case.get("body_extra", {})})
            row["http_status"] = response.status_code
            checks = {"expected_status": response.status_code == case["expected_status"]}
            if case["expected_status"] != 200:
                checks["no_llm_call"] = captured["llm_calls"] == 0
                if case["expected_status"] in (401, 422):
                    checks["no_retrieval"] = captured["retrieval_calls"] == 0
                if case["expected_status"] == 401:
                    checks["bearer_challenge"] = response.headers.get("www-authenticate") == "Bearer"
            elif response.status_code == 200:
                data = response.json()
                checks["verified_identity_preserved"] = all(data[key] == identity[key] for key in ("user_id", "tenant_id", "role"))
                expected_route = case.get("expected_route", "rag")
                row["route"] = data["route"]
                checks["expected_route"] = data["route"] == expected_route
                checks["expected_completion_count"] = captured["llm_calls"] == int(expected_route == "rag")
                if expected_route == "rag":
                    checks["outbound_context_authorized"] = captured["context_checks_passed"] is True
                else:
                    checks["no_answer_sources"] = data["sources"] == []
                row["sources"] = data["sources"]
                if live or expected_route == "abstain":
                    row["answer"] = data["answer"]
                    checks.update(answer_checks(case, data["answer"], restricted_amounts))
            row["checks"] = checks
            row["passed"] = all(checks.values())
    except Exception as exc:
        # Exception strings may include request details; store only safe metadata.
        row.update(passed=False, error_type=type(exc).__name__)
        if getattr(exc, "status_code", None) is not None:
            row["upstream_http_status"] = exc.status_code
    row.update(retrieval_calls=captured["retrieval_calls"], llm_calls=captured["llm_calls"], outbound_context_checks_passed=captured["context_checks_passed"], elapsed_seconds=round(perf_counter() - start, 3))
    return row


def run_suite(cases, *, live=False):
    rows = []
    with TestClient(main.app) as client:
        tokens = {}
        for username in sorted({case["username"] for case in cases}):
            response = client.post("/login", json={"username": username, "password": username + "123"})
            if response.status_code != 200:
                raise RuntimeError("Demo login failed")
            tokens[username] = response.json()["access_token"]
        for case in cases:
            row = run_case(client, case, tokens[case["username"]], live=live)
            rows.append(row)
            print(json.dumps({"id": row["id"], "passed": row["passed"], "http_status": row.get("http_status"), "llm_calls": row["llm_calls"], "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
    accepted = [row["elapsed_seconds"] for row in rows if row.get("http_status") == 200]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llm_mode": "live DeepSeek" if live else "mocked completion; real retrieval and prompt construction",
        "storage": "existing persisted collection; ingestion not rerun",
        "passed": sum(row["passed"] for row in rows), "total": len(rows),
        "answer_behavior_evaluated": live,
        "completion_calls": sum(row["llm_calls"] for row in rows),
        "abstentions": sum(row.get("route") == "abstain" for row in rows),
        "unauthorized_outbound_context_detected": sum(row["outbound_context_checks_passed"] is False for row in rows),
        "accepted_request_median_seconds": round(statistics.median(accepted), 3) if accepted else None,
        "latency_scope": "In-process /ask only; excludes login and model startup. Small sequential fixture suite, not production load.",
        "limitations": "Numeric answer checks cover known fixture salary variants, not arbitrary encodings or semantic disclosure. Only these cases were tested.",
        "checks": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send authorized document context to DeepSeek")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    report = run_suite(cases, live=args.live)
    output = args.output or Path(__file__).with_name("adversarial_live_results.json" if args.live else "adversarial_offline_results.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "total": report["total"], "report": str(output)}))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
