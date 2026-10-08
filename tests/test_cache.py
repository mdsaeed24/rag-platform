"""Cache correctness and authorization tests; all DeepSeek calls are mocked."""

import copy
import os
import secrets
import unittest
from concurrent.futures import ThreadPoolExecutor
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
from semantic_cache import SemanticAnswerCache


VECTOR = [1.0] + [0.0] * 383
IDENTITY = {"user_id": "bob", "tenant_id": "acme", "role": "hr", "token_version": 0}
QUESTION = "What does the CEO earn?"


def hit(source="data/acme/salaries.txt", text="The Acme CEO earns $240,000 per year."):
    return SimpleNamespace(id=1, score=0.9, payload={**copy.deepcopy(DOCUMENT_ACLS[source]), "source": source, "text": text, "chunk_id": 0})


class CachePolicyTests(unittest.TestCase):
    def setUp(self):
        self.cache = SemanticAnswerCache()
        self.results = [hit()]

    def store(self, question=QUESTION, **kwargs):
        self.cache.store(question, VECTOR, kwargs.get("identity", IDENTITY), kwargs.get("results", self.results), kwargs.get("answer", "$240,000"))

    def lookup(self, question=QUESTION, **kwargs):
        return self.cache.lookup(question, kwargs.get("vector", VECTOR), kwargs.get("identity", IDENTITY), kwargs.get("results", self.results))

    def test_exact_and_conservative_semantic_reuse(self):
        self.store()
        self.assertEqual(self.lookup().status, "exact_hit")
        self.assertEqual(self.lookup("How much does the CEO earn?").status, "semantic_hit")

    def test_nearby_queries_cannot_change_entity_negation_units_or_question_form(self):
        self.store()
        for question in (
            "What does the Globex CEO earn?", "What does the Acme CEO earn?",
            "What does the engineering manager earn?", "What does the CEO not earn?",
            "What does the CEO earn monthly?", "What does the CEO earn in 2025?",
            "Does the CEO earn?", "How does the CEO earn?", "What does the CEO earn least?",
        ):
            with self.subTest(question=question):
                # Even identical vectors must not defeat the content checks.
                self.assertIsNone(self.lookup(question))

    def test_semantic_threshold_rejects_low_similarity(self):
        self.store()
        self.assertIsNone(self.lookup("How much does the CEO earn?", vector=[0.0, 1.0] + [0.0] * 382))

    def test_all_identity_dimensions_are_isolated(self):
        self.store()
        for changes in ({"user_id": "other_acme_hr"}, {"tenant_id": "globex"}, {"role": "employee"}, {"token_version": 1}):
            with self.subTest(changes=changes):
                self.assertIsNone(self.lookup(identity={**IDENTITY, **changes}))

    def test_namespaces_isolate_exact_and_semantic_reuse(self):
        self.store()
        for question in (QUESTION, "How much does the CEO earn?"):
            self.assertIsNone(self.cache.lookup(question, VECTOR, IDENTITY, self.results, namespace="graph:v1"))
        self.cache.store(QUESTION, VECTOR, IDENTITY, self.results, "Graph answer", namespace="graph:v1")
        self.assertEqual(self.lookup().answer, "$240,000")
        self.assertEqual(self.cache.lookup(QUESTION, VECTOR, IDENTITY, self.results, namespace="graph:v1").answer, "Graph answer")
        self.assertIsNone(self.cache.lookup(QUESTION, VECTOR, IDENTITY, self.results, namespace="graph:v2"))

    def test_namespaces_share_the_existing_capacity_budget(self):
        self.cache = SemanticAnswerCache(max_entries=2)
        for namespace in ("rag:v1", "graph:v1", "graph:v2"):
            self.cache.store(QUESTION, VECTOR, IDENTITY, self.results, namespace, namespace=namespace)
        self.assertIsNone(self.lookup())
        self.assertIsNotNone(self.cache.lookup(QUESTION, VECTOR, IDENTITY, self.results, namespace="graph:v1"))

    def test_changed_document_content_acl_or_citation_cannot_hit(self):
        self.store()
        for key, value in (("text", "The CEO now earns $250,000"), ("allowed_roles", ["employee"]), ("classification", "internal"), ("chunk_id", 9)):
            with self.subTest(key=key):
                results = copy.deepcopy(self.results)
                results[0].payload[key] = value
                self.assertIsNone(self.lookup(results=results))
        extra = self.results + [hit("data/acme/handbook.txt", "20 days of leave")]
        self.assertIsNone(self.lookup(results=extra))

    def test_current_scores_and_order_do_not_invalidate_same_context(self):
        first, second = hit(), hit("data/acme/handbook.txt", "20 days")
        self.store(results=[first, second])
        second.score = 0.1
        self.assertEqual(self.lookup(results=[second, first]).status, "exact_hit")

    def test_expiration_uses_fixed_lifetime(self):
        now = [0.0]
        self.cache = SemanticAnswerCache(ttl_seconds=10, clock=lambda: now[0])
        self.store()
        now[0] = 9
        self.assertIsNotNone(self.lookup())
        now[0] = 10
        self.assertIsNone(self.lookup())

    def test_capacity_evicts_least_recently_used_answer(self):
        self.cache = SemanticAnswerCache(max_entries=2)
        self.store("CEO salary 1")
        self.store("CEO salary 2")
        self.assertIsNotNone(self.lookup("CEO salary 1"))
        self.store("CEO salary 3")
        self.assertIsNone(self.lookup("CEO salary 2"))
        self.assertIsNotNone(self.lookup("CEO salary 1"))

    def test_unknown_empty_and_no_context_answers_are_not_cached(self):
        for answer in ("", None, "I don't know based on the provided documents."):
            with self.subTest(answer=answer):
                self.store(answer=answer)
                self.assertIsNone(self.lookup())
        self.store(results=[])
        self.assertIsNone(self.lookup())

    def test_disabled_cache_and_parallel_access(self):
        self.cache = SemanticAnswerCache(enabled=False)
        self.store()
        self.assertIsNone(self.lookup())
        self.cache = SemanticAnswerCache()
        def use_cache(_):
            self.store()
            return self.lookup().answer
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(use_cache, range(20))), ["$240,000"] * 20)


class CachedAPITests(unittest.TestCase):
    def setUp(self):
        self.original_users = copy.deepcopy(auth.USERS)
        self.cache = SemanticAnswerCache()
        self.enterContext(patch.object(main, "answer_cache", self.cache))
        self.enterContext(patch.object(main, "embed_query", return_value=VECTOR))
        self.results = [hit()]
        self.retrieve = self.enterContext(patch.object(main, "search_documents", return_value=self.results))
        self.generate = self.enterContext(patch.object(main, "generate_answer", return_value="$240,000 [Source: data/acme/salaries.txt, Chunk: 0]"))
        self.audit = self.enterContext(patch.object(main, "log_event"))
        self.enterContext(patch.object(llm.client.chat.completions, "create", side_effect=AssertionError("No network allowed")))
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def tearDown(self):
        auth.USERS.clear()
        auth.USERS.update(self.original_users)

    def token(self, username="bob"):
        user = auth.USERS[username]
        return auth.create_access_token(username, user["tenant_id"], user["role"])

    def ask(self, question=QUESTION, username="bob", token=None):
        return self.client.post("/ask", headers={"Authorization": "Bearer " + (token or self.token(username))}, json={"question": question})

    def test_hits_skip_only_llm_and_return_current_question_and_sources(self):
        first = self.ask().json()
        exact = self.ask().json()
        self.results[0].score = 0.75
        semantic = self.ask("How much does the CEO earn?").json()
        self.assertEqual([first["cache_status"], exact["cache_status"], semantic["cache_status"]], ["miss", "exact_hit", "semantic_hit"])
        self.assertEqual(semantic["route"], "cache")
        self.assertEqual(semantic["question"], "How much does the CEO earn?")
        self.assertEqual(semantic["sources"][0]["score"], 0.75)
        self.assertEqual(self.retrieve.call_count, 3)
        self.generate.assert_called_once()
        self.assertEqual(self.audit.call_args.kwargs["cache_status"], "semantic_hit")

    def test_bob_salary_cache_cannot_serve_alice_or_carol(self):
        self.ask()
        self.retrieve.return_value = [hit("data/acme/handbook.txt", "Annual leave is 20 days")]
        self.generate.return_value = "I don't know based on the provided documents."
        alice = self.ask(username="alice")
        self.assertEqual(alice.status_code, 200)
        self.assertNotIn("240,000", alice.json()["answer"])
        self.assertEqual(alice.json()["cache_status"], "bypass")
        self.assertEqual(alice.json()["route"], "abstain")
        self.retrieve.return_value = [hit("data/globex/salaries.txt", "Globex CEO earns $310,000")]
        self.generate.return_value = "$310,000"
        carol = self.ask(username="carol")
        self.assertEqual(carol.json()["cache_status"], "miss")
        self.assertEqual(carol.json()["answer"], "$310,000")
        self.assertEqual(self.generate.call_count, 2)

    def test_invalid_token_never_reads_cache(self):
        self.ask()
        with patch.object(self.cache, "lookup", wraps=self.cache.lookup) as lookup:
            self.assertEqual(self.ask(token="forged").status_code, 401)
        lookup.assert_not_called()

    def test_revocation_blocks_warmed_cache_and_new_version_misses(self):
        old_token = self.token()
        self.ask(token=old_token)
        auth.USERS["bob"]["token_version"] += 1
        self.assertEqual(self.ask(token=old_token).status_code, 401)
        self.assertEqual(self.ask().json()["cache_status"], "miss")
        self.assertEqual(self.generate.call_count, 2)

    def test_acl_revocation_blocks_warmed_cache_before_lookup(self):
        self.ask()
        revised = copy.deepcopy(DOCUMENT_ACLS)
        del revised["data/acme/salaries.txt"]
        with patch.dict(DOCUMENT_ACLS, revised, clear=True), patch.object(self.cache, "lookup", wraps=self.cache.lookup) as lookup:
            response = self.ask()
        self.assertEqual(response.status_code, 403)
        lookup.assert_not_called()
        self.generate.assert_called_once()

    def test_changed_context_forces_generation(self):
        self.ask()
        self.results[0].payload["text"] = "Acme CEO now earns $250,000"
        self.generate.return_value = "$250,000"
        response = self.ask().json()
        self.assertEqual(response["cache_status"], "miss")
        self.assertEqual(response["answer"], "$250,000")
        self.assertEqual(self.generate.call_count, 2)

    def test_unsafe_current_retrieval_blocks_warmed_cache(self):
        self.ask()
        self.retrieve.return_value = [hit("data/globex/salaries.txt")]
        self.assertEqual(self.ask().status_code, 403)
        self.generate.assert_called_once()

    def test_revocation_during_hit_lookup_blocks_response(self):
        self.ask()
        lookup = self.cache.lookup
        def revoke(*args, **kwargs):
            match = lookup(*args, **kwargs)
            auth.USERS["bob"]["active"] = False
            return match
        with patch.object(self.cache, "lookup", side_effect=revoke):
            self.assertEqual(self.ask().status_code, 401)
        self.generate.assert_called_once()

    def test_revocation_during_generation_does_not_store_answer(self):
        def revoke(**kwargs):
            auth.USERS["bob"]["active"] = False
            return "$240,000"
        self.generate.side_effect = revoke
        self.assertEqual(self.ask().status_code, 401)
        auth.USERS["bob"]["active"] = True
        self.generate.side_effect = None
        self.assertEqual(self.ask().json()["cache_status"], "miss")
        self.assertEqual(self.generate.call_count, 2)

    def test_document_revoked_during_generation_does_not_store_answer(self):
        revised = copy.deepcopy(DOCUMENT_ACLS)
        def revoke(**kwargs):
            del DOCUMENT_ACLS["data/acme/salaries.txt"]
            return "$240,000"
        with patch.dict(DOCUMENT_ACLS, revised, clear=True):
            self.generate.side_effect = revoke
            self.assertEqual(self.ask().status_code, 403)
        self.generate.side_effect = None
        self.assertEqual(self.ask().json()["cache_status"], "miss")

    def test_generation_failure_is_not_cached(self):
        self.generate.side_effect = RuntimeError("Mocked upstream failure")
        with self.assertRaises(RuntimeError):
            self.ask()
        self.generate.side_effect = None
        self.assertEqual(self.ask().json()["cache_status"], "miss")

    def test_disabled_cache_keeps_rag_behavior(self):
        self.cache.enabled = False
        self.assertEqual(self.ask().json()["cache_status"], "bypass")
        self.assertEqual(self.ask().json()["route"], "rag")
        self.assertEqual(self.generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
