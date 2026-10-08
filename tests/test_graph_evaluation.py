"""Live evaluation uses mocked endpoints in tests; no network is permitted."""

import contextlib
import copy
import io
import json
import os
import secrets
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("DEEPSEEK_API_KEY", "test-only-no-network")
os.environ.setdefault("JWT_SECRET_KEY", secrets.token_urlsafe(48))

from fastapi.testclient import TestClient

import auth
import graph_rag
import llm
import main
from acl_config import DOCUMENT_ACLS
from eval import evaluate_graph as evaluator
from eval.graph_answer_checks import answer_checks, required_fact_present
from query_router import QueryRouter
from semantic_cache import SemanticAnswerCache

ROOT = Path(__file__).resolve().parent.parent
IDENTITY = {"user_id": "bob", "tenant_id": "acme", "role": "hr"}
SOURCES = [{"source": "data/acme/salaries.txt", "chunk_id": 0}, {"source": "data/acme/handbook.txt", "chunk_id": 0}]
ANSWER = ("The software engineer salary is $110,000 per year [Source: data/acme/salaries.txt, Chunk: 0]. "
          "Full-time Acme employees receive 20 days of annual leave [Source: data/acme/handbook.txt, Chunk: 0].")


def documents(tenant="acme"):
    return [SimpleNamespace(id=index, score=0.8, payload={**copy.deepcopy(DOCUMENT_ACLS[source]),
            "source": source, "chunk_id": 0, "text": (ROOT / source).read_text()})
            for index, source in enumerate((f"data/{tenant}/handbook.txt", f"data/{tenant}/salaries.txt"))]


class GraphAnswerCheckTests(unittest.TestCase):
    def test_expected_facts_and_citations_pass(self):
        self.assertTrue(all(answer_checks(("$110,000", "20 days", "Full-time"), ANSWER, SOURCES, IDENTITY).values()))

    def test_common_numeric_and_time_formats_are_accepted(self):
        for answer, fact in (("110 thousand dollars", "$110,000"), ("three days per week", "three days"),
                             ("3 days per week", "three days"), ("twenty paid annual leave days", "20 days"),
                             ("twenty-five days", "25 days"), ("10 AM", "10:00 AM"), ("16:00", "4:00 PM")):
            with self.subTest(answer=answer):
                self.assertTrue(required_fact_present(answer, fact))

    def test_wrong_or_missing_facts_fail(self):
        for answer, fact in (("$125,000", "$110,000"), ("120 days", "20 days"), ("4 AM", "4:00 PM"), ("20 days", "Full-time")):
            with self.subTest(answer=answer):
                self.assertFalse(required_fact_present(answer, fact))

    def test_missing_and_wrong_chunk_citations_fail(self):
        for answer in (ANSWER.replace("handbook.txt", "policy.txt"), ANSWER.replace("Chunk: 0", "Chunk: 7")):
            checks = answer_checks(("$110,000", "20 days"), answer, SOURCES, IDENTITY)
            self.assertFalse(checks["required_source_chunk_citations"])

    def test_unrelated_salary_and_foreign_citations_fail(self):
        answer = ANSWER + " The manager earns $150,000, see data/globex/salaries.txt."
        checks = answer_checks(("$110,000", "20 days"), answer, SOURCES, IDENTITY)
        self.assertFalse(checks["unrelated_fixture_salaries_absent"])
        self.assertFalse(checks["invented_source_citations_absent"])
        self.assertFalse(checks["foreign_tenant_absent"])

    def test_swapped_salary_roles_fail_despite_correct_numbers_and_citations(self):
        answer = "CEO: $110,000; Software engineer: $240,000 [Source: data/acme/salaries.txt, Chunk: 0]"
        checks = answer_checks(("$240,000", "$110,000"), answer, SOURCES[:1], IDENTITY)
        self.assertTrue(checks["required_facts_present"])
        self.assertFalse(checks["salary_role_associations"])

    def test_salary_role_check_accepts_markdown_table(self):
        answer = "| Role | Annual salary |\n| CEO | $240,000 |\n| Software engineers | $110,000 |\nSource: salaries.txt, Chunk: 0"
        checks = answer_checks(("$240,000", "$110,000"), answer, SOURCES[:1], IDENTITY)
        self.assertTrue(all(checks.values()), checks)

    def test_abstention_requires_exact_unknown_and_no_salary(self):
        self.assertTrue(all(answer_checks((), "I don't know based on the provided documents.", [], IDENTITY, abstain=True).values()))
        self.assertFalse(all(answer_checks((), "Unknown, maybe 240k.", [], IDENTITY, abstain=True).values()))


class GraphEvaluationGuardTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)
        self.enterContext(patch.object(main, "graph_rag_enabled", True))
        self.enterContext(patch.object(main, "query_router", QueryRouter()))
        self.enterContext(patch.object(main, "answer_cache", SemanticAnswerCache(enabled=False)))
        self.enterContext(patch.object(main, "embed_query", return_value=[1.0] + [0.0] * 383))
        self.enterContext(patch.object(main, "search_documents", return_value=documents()))
        self.expand = self.enterContext(patch.object(main, "load_graph_documents", return_value=documents()))
        self.enterContext(patch.object(main, "log_event"))
        self.options = self.enterContext(patch.object(llm.client, "with_options", return_value=llm.client))
        self.completion = self.enterContext(patch.object(llm.client.chat.completions, "create", return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=ANSWER))])))
        self.token = auth.create_access_token("bob", "acme", "hr")
        self.case = evaluator.CASES[0]

    def test_live_path_checks_answer_and_bounds_endpoint_options(self):
        row = evaluator.run_case(self.client, self.case, self.token, live=True)
        self.assertTrue(row["passed"], row)
        self.assertEqual(row["answer"], ANSWER)
        self.assertTrue(row["outbound_context_checks_passed"])
        self.assertEqual(row["completion_calls"], 1)
        self.completion.assert_called_once()
        self.options.assert_called_once_with(timeout=30.0, max_retries=0)

    def test_tampered_context_is_stopped_before_real_endpoint(self):
        with patch.object(llm, "graph_context", return_value="Unchecked salary: $310,000"):
            row = evaluator.run_case(self.client, self.case, self.token, live=True)
        self.assertFalse(row["passed"])
        self.assertFalse(row["outbound_context_checks_passed"])
        self.assertEqual(row["completion_calls"], 0)
        self.completion.assert_not_called()

    def test_foreign_batch_is_stopped_even_when_production_acl_checks_are_bypassed(self):
        self.expand.return_value += documents("globex")
        with patch.object(main, "authorize_results"), patch.object(graph_rag, "authorize_results"):
            row = evaluator.run_case(self.client, self.case, self.token, live=True)
        self.assertFalse(row["passed"])
        self.assertFalse(row["outbound_context_checks_passed"])
        self.assertEqual(row["completion_calls"], 0)
        self.completion.assert_not_called()

    def test_wrong_live_answer_fails_even_with_authorized_context(self):
        self.completion.return_value.choices[0].message.content = "$125,000 [Source: data/globex/salaries.txt, Chunk: 0]"
        row = evaluator.run_case(self.client, self.case, self.token, live=True)
        self.assertFalse(row["passed"])
        self.assertTrue(row["outbound_context_checks_passed"])
        self.assertFalse(row["checks"]["required_facts_present"])

    def test_returned_trace_must_match_the_checked_outbound_evidence(self):
        def corrupt_trace(edge):
            return {**asdict(edge), "source": "data/globex/salaries.txt"}
        with patch.object(main, "asdict", side_effect=corrupt_trace):
            row = evaluator.run_case(self.client, self.case, self.token, live=True)
        self.assertFalse(row["passed"])
        self.assertTrue(row["outbound_context_checks_passed"])
        self.assertFalse(row["checks"]["graph_evidence_matches_outbound_context"])

    def test_exception_details_are_not_written_to_report(self):
        self.completion.side_effect = RuntimeError("DO_NOT_LOG_REQUEST_OR_SECRET")
        row = evaluator.run_case(self.client, self.case, self.token, live=True)
        self.assertFalse(row["passed"])
        self.assertEqual(row["error_type"], "RuntimeError")
        self.assertNotIn("DO_NOT_LOG", json.dumps(row))

    def test_offline_mode_never_calls_endpoint_or_claims_live_answer_quality(self):
        row = evaluator.run_case(self.client, self.case, self.token)
        self.assertTrue(row["passed"])
        self.assertNotIn("answer", row)
        self.assertNotIn("required_facts_present", row["checks"])
        self.completion.assert_not_called()
        self.options.assert_not_called()

    def test_suite_stops_on_failed_outbound_guard_without_hiding_unexecuted_cases(self):
        unsafe = {"id": "guard_failure", "passed": False, "http_status": None, "completion_calls": 0,
                  "outbound_context_checks_passed": False, "elapsed_seconds": 0}
        with patch.object(evaluator, "run_case", return_value=unsafe) as run, contextlib.redirect_stdout(io.StringIO()):
            report = evaluator.evaluate(live=True)
        run.assert_called_once()
        self.assertEqual(report["executed"], 1)
        self.assertEqual(report["total"], 16)
        self.assertEqual(report["passed"], 0)
        self.assertFalse(report["live_answer_correctness_evaluated"])


if __name__ == "__main__":
    unittest.main()
