"""Provider failure handling with mocked completions and an in-memory transport."""

import copy
import json
import os
from pathlib import Path
import secrets
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("DEEPSEEK_API_KEY", "test-only-no-network")
os.environ.setdefault("JWT_SECRET_KEY", secrets.token_urlsafe(48))

import httpx2
from fastapi.testclient import TestClient
from openai import APITimeoutError, APIConnectionError, APIStatusError, APIResponseValidationError

import auth
import llm
import main
from acl_config import DOCUMENT_ACLS
from query_router import QueryRouter
from semantic_cache import SemanticAnswerCache

ROOT = Path(__file__).resolve().parents[1]
VECTOR = [1.0] + [0.0] * 383
PRIVATE = "DO_NOT_EXPOSE_PROVIDER_BODY_OR_KEY"
REQUEST = httpx2.Request("POST", "https://provider.invalid/chat/completions",
                         headers={"Authorization": "Bearer " + PRIVATE})
QUESTIONS = {"rag": "What does the CEO earn?",
             "graph": "What do software engineers earn and how much annual leave is provided?"}


def corpus():
    return [SimpleNamespace(id=index, score=0.9, payload={
        **copy.deepcopy(DOCUMENT_ACLS[source]), "source": source, "chunk_id": 0,
        "text": (ROOT / source).read_text(),
    }) for index, source in enumerate(("data/acme/handbook.txt", "data/acme/salaries.txt"))]


def completion(content="Authorized answer", finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason)])


def provider_error(status):
    response = httpx2.Response(status, request=REQUEST, json={"error": PRIVATE})
    return APIStatusError(PRIVATE, response=response, body={"error": PRIVATE})


class CompletionConfigurationTests(unittest.TestCase):
    def test_timeout_defaults_and_finite_positive_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(llm.completion_timeout(), 30)
        for value in ("0", "-1", "nan", "inf", "-inf", "invalid", ""):
            with self.subTest(value=value), patch.dict(os.environ, {"LLM_TIMEOUT_SECONDS": value}):
                with self.assertRaises(ValueError):
                    llm.completion_timeout()
        with patch.dict(os.environ, {"LLM_TIMEOUT_SECONDS": "12.5"}):
            self.assertEqual(llm.completion_timeout(), 12.5)

    def test_sdk_timeout_and_no_retry_reach_the_transport(self):
        calls = []

        def fail(request):
            calls.append(request)
            raise httpx2.ReadTimeout(PRIVATE, request=request)

        with httpx2.Client(transport=httpx2.MockTransport(fail)) as transport:
            provider = llm.client.with_options(http_client=transport)
            with patch.object(llm, "client", provider):
                with self.assertRaises(llm.GenerationError) as captured:
                    llm._complete_answer("Question", "Authorized context")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].extensions["timeout"]["read"], llm.completion_timeout())
        self.assertEqual(provider.max_retries, 0)
        self.assertEqual(captured.exception.status_code, 504)
        self.assertNotIn(PRIVATE, str(captured.exception))


class GenerationAPITests(unittest.TestCase):
    def setUp(self):
        self.documents = corpus()
        self.cache = SemanticAnswerCache()
        self.enterContext(patch.object(main, "answer_cache", self.cache))
        self.enterContext(patch.object(main, "query_router", QueryRouter()))
        self.enterContext(patch.object(main, "graph_rag_enabled", True))
        self.enterContext(patch.object(main, "graph_answer_cache_enabled", True))
        self.enterContext(patch.object(main, "embed_query", return_value=VECTOR))
        self.enterContext(patch.object(main, "search_documents", return_value=self.documents))
        self.enterContext(patch.object(main, "load_graph_documents", return_value=self.documents))
        self.audit = self.enterContext(patch.object(main, "log_event"))
        self.complete = self.enterContext(patch.object(llm.client.chat.completions, "create",
                                                    return_value=completion()))
        self.store = self.enterContext(patch.object(self.cache, "store", wraps=self.cache.store))
        self.token = auth.create_access_token("bob", "acme", "hr")
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def ask(self, route):
        return self.client.post("/ask", json={"question": QUESTIONS[route]},
                                headers={"Authorization": "Bearer " + self.token})

    def check_failure_and_recovery(self, route, error, status, reason):
        self.cache.clear()
        self.store.reset_mock()
        self.audit.reset_mock()
        self.complete.reset_mock()
        self.complete.side_effect = error
        response = self.ask(route)
        self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(set(response.json()), {"detail"})
        self.assertNotIn(PRIVATE, response.text)
        self.assertNotIn("www-authenticate", response.headers)
        self.assertNotIn("answer", response.json())
        self.store.assert_not_called()
        self.complete.assert_called_once()
        event = self.audit.call_args.kwargs
        self.assertEqual((event["decision"], event["reason"], event["route"]),
                         ("error", reason, route))
        self.assertGreaterEqual(event["generation_ms"], 0)
        self.assertNotIn(PRIVATE, str(self.audit.call_args_list))
        self.assertNotIn("$110,000", json.dumps({key: value for key, value in event.items()
                                                if key != "results"}))
        self.complete.side_effect = None
        recovered = self.ask(route)
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertEqual(recovered.json()["cache_status"], "miss")
        cached = self.ask(route)
        self.assertEqual(cached.json()["route"], "graph_cache" if route == "graph" else "cache")
        self.assertEqual(self.complete.call_count, 2)

    def test_timeouts_return_504_and_recover_without_failed_cache_entries(self):
        for route in QUESTIONS:
            with self.subTest(route=route):
                self.check_failure_and_recovery(route, APITimeoutError(REQUEST), 504,
                                                "generation_timeout")

    def test_connection_failures_return_503(self):
        for route in QUESTIONS:
            with self.subTest(route=route):
                self.check_failure_and_recovery(route, APIConnectionError(request=REQUEST, message=PRIVATE),
                                                503, "generation_unavailable")

    def test_rate_limits_and_server_failures_return_503(self):
        for route in QUESTIONS:
            for status in (429, 500, 502, 503, 504):
                with self.subTest(route=route, status=status):
                    self.check_failure_and_recovery(route, provider_error(status), 503,
                                                    "generation_unavailable")

    def test_provider_rejections_return_502_including_provider_auth_failures(self):
        for route in QUESTIONS:
            for status in (400, 401, 403, 404, 422):
                with self.subTest(route=route, status=status):
                    self.check_failure_and_recovery(route, provider_error(status), 502,
                                                    "generation_rejected")

    def test_schema_validation_failures_return_502(self):
        response = httpx2.Response(200, request=REQUEST)
        for route in QUESTIONS:
            with self.subTest(route=route):
                self.check_failure_and_recovery(route,
                    APIResponseValidationError(response, {"secret": PRIVATE}),
                    502, "generation_invalid_response")

    def test_empty_malformed_and_incomplete_answers_never_enter_cache(self):
        invalid = [None, SimpleNamespace(choices=[]), SimpleNamespace(choices=1),
                   SimpleNamespace(choices=[None]), completion(None), completion(" "),
                   completion([PRIVATE]), completion(PRIVATE, "length"),
                   completion(PRIVATE, "content_filter"), completion(PRIVATE, "tool_calls"),
                   completion(PRIVATE, None)]
        for route in QUESTIONS:
            for index, result in enumerate(invalid):
                with self.subTest(route=route, invalid=index):
                    self.cache.clear()
                    self.store.reset_mock()
                    self.complete.return_value = result
                    response = self.ask(route)
                    self.assertEqual(response.status_code, 502, response.text)
                    self.assertNotIn(PRIVATE, response.text)
                    self.store.assert_not_called()

    def test_cache_hits_and_abstentions_skip_failing_provider(self):
        for route in QUESTIONS:
            with self.subTest(route=route):
                self.complete.side_effect = None
                self.assertEqual(self.ask(route).status_code, 200)
                self.complete.reset_mock()
                self.complete.side_effect = APITimeoutError(REQUEST)
                self.assertEqual(self.ask(route).status_code, 200)
                self.complete.assert_not_called()
        response = self.client.post("/ask", json={"question": "What does the Globex CEO earn?"},
                                    headers={"Authorization": "Bearer " + self.token})
        self.assertEqual(response.json()["route"], "abstain")
        self.complete.assert_not_called()

    def test_invalid_token_skips_provider_even_during_outage(self):
        self.complete.side_effect = APITimeoutError(REQUEST)
        response = self.client.post("/ask", json={"question": QUESTIONS["graph"]},
                                    headers={"Authorization": "Bearer invalid"})
        self.assertEqual(response.status_code, 401)
        self.complete.assert_not_called()

    def test_acl_change_at_generation_boundary_still_returns_403(self):
        for route, name in (("rag", "generate_answer"), ("graph", "generate_graph_answer")):
            original = getattr(main, name)

            def revoke_then_generate(*args, **kwargs):
                with patch.dict(DOCUMENT_ACLS, {}, clear=True):
                    return original(*args, **kwargs)

            with self.subTest(route=route), patch.object(main, name, side_effect=revoke_then_generate):
                response = self.ask(route)
                self.assertEqual(response.status_code, 403, response.text)
                self.complete.assert_not_called()
                self.store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
