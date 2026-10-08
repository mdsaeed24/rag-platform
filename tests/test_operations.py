"""Offline admission, health, and content-free operational metrics checks."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import secrets
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("DEEPSEEK_API_KEY", "test-only-no-network")
os.environ.setdefault("JWT_SECRET_KEY", secrets.token_urlsafe(48))

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

import auth
import llm
import main
import search
from acl_config import DOCUMENT_ACLS
from operations import OperationalMetrics, RequestControlsMiddleware, positive_integer_setting, metrics_token_setting
from query_router import QueryRouter
from semantic_cache import SemanticAnswerCache

TOKEN = "metrics-only-" + "x" * 32


class OperationalAPITests(unittest.TestCase):
    def setUp(self):
        self.metrics = OperationalMetrics(ask_limit=1, login_limit=1)
        self.enterContext(patch.object(main, "operational_metrics", self.metrics))
        self.enterContext(patch.object(main, "metrics_token", TOKEN))
        self.enterContext(patch.object(main, "answer_cache", SemanticAnswerCache()))
        self.enterContext(patch.object(main, "query_router", QueryRouter()))
        self.enterContext(patch.object(main, "embed_query", return_value=[1.0] + [0.0] * 383))
        source = "data/acme/handbook.txt"
        hit = SimpleNamespace(id=1, score=0.9, payload={**copy.deepcopy(DOCUMENT_ACLS[source]),
            "source": source, "chunk_id": 0, "text": "Employees receive 20 days of paid annual leave."})
        self.retrieve = self.enterContext(patch.object(main, "search_documents", return_value=[hit]))
        self.generate = self.enterContext(patch.object(main, "generate_answer", return_value="20 days"))
        self.enterContext(patch.object(main, "log_event"))
        self.network = self.enterContext(patch.object(llm.client.chat.completions, "create",
                                                     side_effect=AssertionError("No network allowed")))
        self.client = TestClient(RequestControlsMiddleware(main.app, metrics=self.metrics, max_body_bytes=16384))
        self.addCleanup(self.client.close)
        self.token = auth.create_access_token("bob", "acme", "hr")

    def ask(self, question="Annual leave?"):
        return self.client.post("/ask", json={"question": question},
                                headers={"Authorization": "Bearer " + self.token})

    def test_health_is_public_and_does_not_retrieve_or_call_provider(self):
        self.assertEqual(self.client.get("/health/live").json(), {"status": "alive"})
        for ready, status in ((True, 200), (False, 503)):
            with self.subTest(ready=ready), patch.object(main, "local_dependencies_ready", return_value=ready):
                response = self.client.get("/health/ready")
                self.assertEqual(response.status_code, status)
                self.assertEqual(set(response.json()), {"status"})
        self.retrieve.assert_not_called()
        self.generate.assert_not_called()
        self.network.assert_not_called()

    def test_readiness_errors_are_sanitized(self):
        with patch.object(main, "local_dependencies_ready", side_effect=RuntimeError("PRIVATE_STORAGE_DETAILS")):
            response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "not_ready"})

    def test_metrics_disabled_without_configuration(self):
        with patch.object(main, "metrics_token", ""):
            self.assertEqual(self.client.get("/metrics").status_code, 404)

    def test_metrics_require_independent_credentials(self):
        for header in (None, "Bearer wrong", "Basic " + TOKEN, "Bearer " + self.token, "Bearer a b"):
            with self.subTest(header=header):
                response = self.client.get("/metrics", headers={"Authorization": header} if header else {})
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.headers["www-authenticate"], "Bearer")
                self.assertNotIn("rag_http_requests", response.text)
        self.assertEqual(self.client.post("/ask", json={"question": "Annual leave?"},
                         headers={"Authorization": "Bearer " + TOKEN}).status_code, 401)

    def test_metrics_count_routes_status_and_latency_without_content_or_identity(self):
        self.assertEqual(self.ask().json()["route"], "rag")
        self.assertEqual(self.ask().json()["route"], "cache")
        self.assertEqual(self.ask("Globex leave?").json()["route"], "abstain")
        self.assertEqual(self.client.post("/ask", json={"question": "PRIVATE_QUERY"}).status_code, 401)
        self.client.get("/PRIVATE_UNKNOWN_PATH?token=PRIVATE_QUERY")
        response = self.client.get("/metrics", headers={"Authorization": "Bearer " + TOKEN})
        self.assertEqual(response.status_code, 200)
        self.assertIn("version=0.0.4", response.headers["content-type"])
        for route in ("rag", "cache", "abstain"):
            self.assertIn(f'rag_answers_total{{route="{route}"}} 1', response.text)
        self.assertIn('rag_http_requests_total{endpoint="/ask",status="200"} 3', response.text)
        self.assertIn('rag_http_requests_total{endpoint="/ask",status="401"} 1', response.text)
        self.assertIn('rag_http_request_duration_seconds_sum{endpoint="/ask",status="200"}', response.text)
        self.assertIn('rag_http_requests_total{endpoint="other",status="404"} 1', response.text)
        for private in ("PRIVATE", "bob", "acme", "handbook", "20 days", self.token, TOKEN):
            self.assertNotIn(private, response.text)

    def test_large_body_rejected_before_auth_retrieval_or_generation(self):
        for endpoint in ("/ask", "/login"):
            with self.subTest(endpoint=endpoint):
                response = self.client.post(endpoint, content=b"x" * 16385)
                self.assertEqual(response.status_code, 413)
        self.retrieve.assert_not_called()
        self.generate.assert_not_called()
        self.assertEqual(self.ask().status_code, 200)

    def test_graph_and_graph_cache_answers_are_counted_after_authorization(self):
        root = Path(__file__).resolve().parents[1]
        documents = [SimpleNamespace(id=index, score=0.9, payload={
            **copy.deepcopy(DOCUMENT_ACLS[source]), "source": source, "chunk_id": 0,
            "text": (root / source).read_text(),
        }) for index, source in enumerate(("data/acme/handbook.txt", "data/acme/salaries.txt"))]
        self.retrieve.return_value = documents
        with patch.object(main, "graph_rag_enabled", True), patch.object(main, "graph_answer_cache_enabled", True), patch.object(main, "load_graph_documents", return_value=documents), patch.object(main, "generate_graph_answer", return_value="Graph answer"):
            self.assertEqual(self.ask("Compare CEO and software engineer salaries.").json()["route"], "graph")
            self.assertEqual(self.ask("Compare CEO and software engineer salaries.").json()["route"], "graph_cache")
        self.assertEqual(self.metrics.answers["graph"], 1)
        self.assertEqual(self.metrics.answers["graph_cache"], 1)

    def test_provider_errors_count_as_failed_requests_without_counting_answers(self):
        self.generate.side_effect = llm.GenerationError("generation_timeout", 504, "Answer provider timed out")
        self.assertEqual(self.ask().status_code, 504)
        self.assertEqual(self.metrics.requests["/ask", 504], 1)
        self.assertEqual(sum(self.metrics.answers.values()), 0)
        self.assertEqual(self.metrics.active["/ask"], 0)

    def test_ask_capacity_rejects_without_queue_and_health_remains_available(self):
        entered, release = Event(), Event()

        def slow_answer(**kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError("Timed out waiting for test release")
            return "20 days"

        self.generate.side_effect = slow_answer
        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(self.ask)
            try:
                self.assertTrue(entered.wait(5))
                rejected = self.ask()
                self.assertEqual(rejected.status_code, 503)
                self.assertEqual(rejected.headers["retry-after"], "1")
                self.assertEqual(self.client.get("/health/live").status_code, 200)
                self.assertEqual(self.generate.call_count, 1)
                self.assertEqual(self.retrieve.call_count, 1)
            finally:
                release.set()
            self.assertEqual(running.result(timeout=5).status_code, 200)
        self.assertEqual(self.metrics.active["/ask"], 0)
        self.assertEqual(self.ask().status_code, 200)

    def test_login_has_separate_capacity_and_slots_recover_on_validation_errors(self):
        self.assertTrue(self.metrics.enter("/login"))
        try:
            response = self.client.post("/login", json={"username": "bob", "password": "bob123"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(self.ask().status_code, 200)
        finally:
            self.metrics.leave("/login")
        self.assertEqual(self.client.post("/login", json={}).status_code, 422)
        self.assertEqual(self.client.post("/login", json={"username": "bob", "password": "bob123"}).status_code, 200)
        self.assertEqual(self.metrics.active["/login"], 0)

    def test_slots_release_after_unexpected_errors(self):
        self.generate.side_effect = RuntimeError("test failure")
        with self.assertRaises(RuntimeError):
            self.ask()
        self.assertEqual(self.metrics.active["/ask"], 0)
        self.generate.side_effect = None
        self.assertEqual(self.ask().status_code, 200)


class StreamAndConfigurationTests(unittest.TestCase):
    def test_chunked_and_dishonest_content_lengths_cannot_bypass_body_limit(self):
        async def run(headers):
            called, messages = [], []

            async def app(scope, receive, send):
                called.append(True)

            chunks = iter([{"type": "http.request", "body": b"1234", "more_body": True},
                           {"type": "http.request", "body": b"5678", "more_body": False}])

            async def receive():
                return next(chunks)

            async def send(message):
                messages.append(message)

            metrics = OperationalMetrics(ask_limit=1)
            await RequestControlsMiddleware(app, metrics=metrics, max_body_bytes=7)(
                {"type": "http", "method": "POST", "path": "/ask", "headers": headers}, receive, send)
            self.assertEqual(messages[0]["status"], 413)
            self.assertEqual(called, [])
            self.assertEqual(metrics.active["/ask"], 0)

        for headers in ([], [(b"content-length", b"1")], [(b"content-length", b"invalid")]):
            asyncio.run(run(headers))

    def test_disconnect_releases_admission_slot(self):
        async def run():
            metrics = OperationalMetrics(ask_limit=1)

            async def receive():
                return {"type": "http.disconnect"}

            async def fail(*args):
                raise AssertionError("Application and response must be skipped")

            await RequestControlsMiddleware(fail, metrics=metrics, max_body_bytes=7)(
                {"type": "http", "method": "POST", "path": "/ask"}, receive, fail)
            self.assertEqual(metrics.active["/ask"], 0)
            self.assertEqual(metrics.requests["/ask", 499], 1)

        asyncio.run(run())

    def test_limits_and_metrics_credentials_fail_closed_on_invalid_configuration(self):
        for value in ("0", "-1", "1.5", "nan", "invalid", ""):
            with self.subTest(value=value), patch.dict(os.environ, {"LIMIT_TEST": value}):
                with self.assertRaises(ValueError):
                    positive_integer_setting("LIMIT_TEST", 4)
        with patch.dict(os.environ, {"LIMIT_TEST": "2"}):
            self.assertEqual(positive_integer_setting("LIMIT_TEST", 4), 2)
        for token in ("short", " " * 32, " " + TOKEN):
            with self.subTest(token=token), patch.dict(os.environ, {"METRICS_TOKEN": token}):
                with self.assertRaises(ValueError):
                    metrics_token_setting()
        with patch.dict(os.environ, {"METRICS_TOKEN": ""}):
            self.assertEqual(metrics_token_setting(), "")

    def test_parallel_metrics_updates_are_not_lost_and_unknown_paths_are_bounded(self):
        metrics = OperationalMetrics()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: metrics.record_request(f"/unknown/{i}", 404, 0.1), range(200)))
        self.assertEqual(metrics.requests["other", 404], 200)
        self.assertEqual(len(metrics.requests), 1)


class LocalReadinessTests(unittest.TestCase):
    def test_checks_existing_collection_configuration_and_releases_storage(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(search, "QDRANT_PATH", Path(directory)):
            self.assertFalse(search.local_dependencies_ready())
            client = QdrantClient(path=directory)
            client.create_collection(search.COLLECTION_NAME,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
            client.close()
            self.assertFalse(search.local_dependencies_ready())
            client = QdrantClient(path=directory)
            client.upsert(search.COLLECTION_NAME, points=[models.PointStruct(id=1, vector=[1.0] + [0.0] * 383)])
            client.close()
            self.assertTrue(search.local_dependencies_ready())
            with patch.object(search.model, "get_embedding_dimension", return_value=12):
                self.assertFalse(search.local_dependencies_ready())
            self.assertTrue(search.local_dependencies_ready())
            with patch.object(search, "COLLECTION_NAME", "missing_collection"):
                self.assertFalse(search.local_dependencies_ready())
            self.assertTrue(search.local_dependencies_ready())

    def test_incompatible_collection_dimensions_fail_readiness(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(search, "QDRANT_PATH", Path(directory)):
            client = QdrantClient(path=directory)
            client.create_collection(search.COLLECTION_NAME,
                vectors_config=models.VectorParams(size=12, distance=models.Distance.COSINE))
            client.upsert(search.COLLECTION_NAME, points=[models.PointStruct(id=1, vector=[1.0] * 12)])
            client.close()
            self.assertFalse(search.local_dependencies_ready())

    def test_storage_probe_error_closes_client_and_releases_lock(self):
        with patch.object(Path, "is_file", return_value=True), patch.object(search, "QdrantClient") as factory:
            factory.return_value.collection_exists.side_effect = RuntimeError("Storage unavailable")
            with self.assertRaises(RuntimeError):
                search.local_dependencies_ready()
            factory.return_value.close.assert_called_once()
            self.assertTrue(search._storage_lock.acquire(blocking=False))
            search._storage_lock.release()


if __name__ == "__main__":
    unittest.main()
