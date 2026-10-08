"""Routing is a cost optimization, never an authorization boundary."""

import copy
import os
import secrets
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("DEEPSEEK_API_KEY", "test-only-no-network")
os.environ.setdefault("JWT_SECRET_KEY", secrets.token_urlsafe(48))

from fastapi.testclient import TestClient

import auth
import llm
import main
from acl_config import DOCUMENT_ACLS
from query_router import QueryRouter, UNKNOWN_ANSWER
from semantic_cache import SemanticAnswerCache


def hit(source="data/acme/handbook.txt", text="Employees receive 20 days of paid annual leave.", score=0.8):
    return SimpleNamespace(id=1, score=score, payload={**copy.deepcopy(DOCUMENT_ACLS[source]), "source": source, "text": text, "chunk_id": 0})


class RoutingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter()

    def test_empty_context_always_abstains(self):
        for enabled in (True, False):
            decision = QueryRouter(enabled=enabled).decide("Annual leave?", "acme", [])
            self.assertEqual(decision.route, "abstain")
            self.assertEqual(decision.reason, "no_context")

    def test_low_score_and_nonfinite_evidence_abstain(self):
        for score in (0.249, -0.5, float("nan"), float("inf"), None):
            with self.subTest(score=score):
                self.assertEqual(self.router.decide("Annual leave?", "acme", [hit(score=score)]).route, "abstain")
        self.assertEqual(self.router.decide("Annual leave?", "acme", [hit(score=0.25)]).route, "rag")

    def test_foreign_tenant_mentions_abstain_without_specific_explanation(self):
        for question in ("Globex annual leave?", "What is GLOBEX's leave policy?", "Compare Acme and Globex leave."):
            with self.subTest(question=question):
                decision = self.router.decide(question, "acme", [hit()])
                self.assertEqual(decision.route, "abstain")
                self.assertEqual(decision.reason, "insufficient_context")
        self.assertEqual(self.router.decide("Acme annual leave?", "acme", [hit()]).route, "rag")

    def test_salary_requires_evidence_in_current_context(self):
        for question in ("What does the CEO earn?", "What is the CEO paid?", "List salaries.", "Engineering manager compensation?"):
            with self.subTest(question=question):
                self.assertEqual(self.router.decide(question, "acme", [hit()]).route, "abstain")
        salary = hit("data/acme/salaries.txt", "The CEO earns $240,000 per year.")
        self.assertEqual(self.router.decide("What does the CEO earn?", "acme", [salary]).route, "rag")

    def test_paid_leave_and_expense_questions_do_not_become_salary_questions(self):
        for question in ("How many days of paid annual leave do employees receive?", "How many paid leave days do engineering managers receive?", "How much annual leave do employees earn?", "When do employees pay expenses?"):
            with self.subTest(question=question):
                self.assertEqual(self.router.decide(question, "acme", [hit()]).route, "rag")

    def test_disabled_router_preserves_nonempty_rag_fallback(self):
        router = QueryRouter(enabled=False)
        self.assertEqual(router.decide("Globex CEO salary?", "acme", [hit(score=-0.5)]).route, "rag")


class RoutingAPITests(unittest.TestCase):
    def setUp(self):
        self.users = copy.deepcopy(auth.USERS)
        self.router = QueryRouter()
        self.cache = SemanticAnswerCache()
        self.enterContext(patch.object(main, "query_router", self.router))
        self.enterContext(patch.object(main, "answer_cache", self.cache))
        self.enterContext(patch.object(main, "embed_query", return_value=[1.0] + [0.0] * 383))
        self.retrieve = self.enterContext(patch.object(main, "search_documents", return_value=[hit()]))
        self.generate = self.enterContext(patch.object(main, "generate_answer", return_value="20 days [Source: data/acme/handbook.txt, Chunk: 0]"))
        self.audit = self.enterContext(patch.object(main, "log_event"))
        self.enterContext(patch.object(llm.client.chat.completions, "create", side_effect=AssertionError("No network allowed")))
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def tearDown(self):
        auth.USERS.clear()
        auth.USERS.update(self.users)

    def ask(self, question="Annual leave?", **body_extra):
        user = auth.USERS["alice"]
        token = auth.create_access_token("alice", user["tenant_id"], user["role"])
        return self.client.post("/ask", headers={"Authorization": "Bearer " + token}, json={"question": question, **body_extra})

    def test_supported_question_uses_rag_then_cache(self):
        first, second = self.ask().json(), self.ask().json()
        self.assertEqual(first["route"], "rag")
        self.assertEqual(first["routing_reason"], "supported_context")
        self.assertEqual(second["route"], "cache")
        self.assertEqual(second["routing_reason"], "cached_answer")
        self.assertEqual(self.retrieve.call_count, 2)
        self.generate.assert_called_once()

    def test_salary_without_evidence_abstains_and_never_reads_or_writes_cache(self):
        with patch.object(self.cache, "lookup", wraps=self.cache.lookup) as lookup, patch.object(self.cache, "store", wraps=self.cache.store) as store:
            response = self.ask("What does the CEO earn?")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], UNKNOWN_ANSWER)
        self.assertEqual(response.json()["route"], "abstain")
        self.assertEqual(response.json()["cache_status"], "bypass")
        self.assertEqual(response.json()["sources"], [])
        lookup.assert_not_called()
        store.assert_not_called()
        self.generate.assert_not_called()
        self.assertEqual(self.audit.call_args.kwargs["route"], "abstain")

    def test_low_evidence_cannot_use_warmed_cache(self):
        self.ask()
        self.retrieve.return_value[0].score = 0.01
        with patch.object(self.cache, "lookup", wraps=self.cache.lookup) as lookup:
            self.assertEqual(self.ask().json()["route"], "abstain")
        lookup.assert_not_called()
        self.generate.assert_called_once()

    def test_unauthorized_chunk_is_rejected_before_router_can_abstain(self):
        self.retrieve.return_value = [hit("data/globex/handbook.txt", score=0.01)]
        with patch.object(self.router, "decide", wraps=self.router.decide) as decide:
            self.assertEqual(self.ask("Quantum mechanics?").status_code, 403)
        decide.assert_not_called()
        self.generate.assert_not_called()

    def test_missing_authentication_and_body_route_override_cannot_bypass_checks(self):
        self.assertEqual(self.client.post("/ask", json={"question": "Annual leave?"}).status_code, 401)
        self.assertEqual(self.ask(route="rag").status_code, 422)
        self.assertEqual(self.ask(tenant_id="globex").status_code, 422)
        self.retrieve.assert_not_called()
        self.generate.assert_not_called()

    def test_router_disabled_keeps_acl_checks_and_rag_fallback(self):
        self.router.enabled = False
        self.retrieve.return_value[0].score = -0.5
        self.assertEqual(self.ask().json()["route"], "rag")
        self.retrieve.return_value = [hit("data/globex/handbook.txt")]
        self.assertEqual(self.ask().status_code, 403)
        self.generate.assert_called_once()

    def test_revocation_during_abstention_still_blocks_response(self):
        decide = self.router.decide
        def revoke(*args, **kwargs):
            decision = decide(*args, **kwargs)
            auth.USERS["alice"]["active"] = False
            return decision
        with patch.object(self.router, "decide", side_effect=revoke):
            self.assertEqual(self.ask("What does the CEO earn?").status_code, 401)
        self.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
