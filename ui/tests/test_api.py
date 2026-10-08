"""HTTP contract and credential handling; no sockets or backend imports."""

import copy
import json
import unittest

import httpx

from ui.api import ApiError, RagApi, backend_url

QUESTION = "What does the CEO earn?"
ANSWER = {
    "question": QUESTION, "user_id": "bob", "tenant_id": "acme", "role": "hr",
    "answer": "The Acme CEO earns $240,000 per year.", "route": "rag",
    "routing_reason": "authorized_context", "cache_status": "miss",
    "sources": [{"source": "data/acme/salaries.txt", "chunk_id": 0, "score": 0.8}],
    "graph_evidence": [],
}


class ApiTests(unittest.TestCase):
    def api(self, handler):
        return RagApi("http://127.0.0.1:8001", transport=httpx.MockTransport(handler))

    def test_only_login_credentials_and_question_are_sent(self):
        requests = []
        def handle(request):
            requests.append(request)
            return httpx.Response(200, json={"access_token": "SESSION_TOKEN", "token_type": "bearer"}
                                  if request.url.path == "/login" else ANSWER)
        api = self.api(handle)
        token = api.login("bob", "private-password")
        self.assertEqual(api.ask(QUESTION, token), ANSWER)
        self.assertEqual(json.loads(requests[0].content), {"username": "bob", "password": "private-password"})
        self.assertNotIn("authorization", requests[0].headers)
        self.assertEqual(json.loads(requests[1].content), {"question": QUESTION})
        self.assertEqual(requests[1].headers["authorization"], "Bearer SESSION_TOKEN")
        self.assertEqual(requests[1].url.query, b"")

    def test_operator_url_requires_local_http_or_https_without_embedded_credentials(self):
        for url in ("http://127.0.0.1:8001/", "http://localhost:8001", "http://[::1]:8001", "https://api.example.com/rag/"):
            self.assertEqual(backend_url(url), url.rstrip("/"))
        for url in ("http://public.example.com", "ftp://localhost", "https://u:p@example.com",
                    "https://example.com?token=secret", "https://example.com#fragment", "", None,
                    "https://example.com:bad", "https://example.com:0", "https://example.com\\evil", "https://ex ample.com"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                backend_url(url)

    def test_redirects_are_never_followed_with_credentials(self):
        requests = []
        def redirect(request):
            requests.append(request)
            return httpx.Response(307, headers={"location": "https://other.example.com/ask"})
        with self.assertRaises(ApiError):
            self.api(redirect).ask(QUESTION, "SESSION_TOKEN")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.host, "127.0.0.1")

    def test_http_errors_do_not_display_response_bodies(self):
        for status in (401, 403, 409, 413, 422, 429, 500, 502, 503, 504):
            with self.subTest(status=status), self.assertRaises(ApiError) as error:
                self.api(lambda request: httpx.Response(status, text="PRIVATE_PROVIDER_BODY SESSION_TOKEN")).ask(QUESTION, "SESSION_TOKEN")
            self.assertEqual(error.exception.status, status)
            self.assertNotIn("PRIVATE", str(error.exception))
            self.assertNotIn("SESSION_TOKEN", str(error.exception))

    def test_timeout_and_connection_failures_are_sanitized(self):
        for kind in (httpx.ReadTimeout, httpx.ConnectError):
            def fail(request):
                raise kind("PRIVATE_REQUEST_DETAILS", request=request)
            with self.subTest(kind=kind), self.assertRaises(ApiError) as error:
                self.api(fail).ask(QUESTION, "SESSION_TOKEN")
            self.assertNotIn("PRIVATE", str(error.exception))

    def test_invalid_login_and_non_json_responses_are_rejected(self):
        for body in ({}, {"access_token": "bad\ntoken", "token_type": "bearer"},
                     {"access_token": "TOKEN", "token_type": "other"}, []):
            with self.subTest(body=body), self.assertRaises(ApiError):
                self.api(lambda request: httpx.Response(200, json=body)).login("bob", "password")
        with self.assertRaises(ApiError):
            self.api(lambda request: httpx.Response(200, text="not JSON")).ask(QUESTION, "TOKEN")

    def test_invalid_answer_citations_and_mismatched_questions_fail_closed(self):
        patches = ({"route": []}, {"sources": None}, {"question": "another user's question"},
                   {"sources": [{"source": "x", "chunk_id": True}]}, {"sources": []},
                   {"route": "abstain"}, {"graph_evidence": [{"subject": "x"}]},
                   {"route": "graph", "graph_evidence": []})
        for changes in patches:
            body = {**copy.deepcopy(ANSWER), **changes}
            with self.subTest(changes=changes), self.assertRaises(ApiError):
                self.api(lambda request: httpx.Response(200, json=body)).ask(QUESTION, "TOKEN")

    def test_readiness_does_not_send_a_bearer_token(self):
        def ready(request):
            self.assertEqual(request.url.path, "/health/ready")
            self.assertNotIn("authorization", request.headers)
            return httpx.Response(200, json={"status": "ready"})
        self.assertTrue(self.api(ready).ready())
        self.assertFalse(self.api(lambda request: httpx.Response(503)).ready())


if __name__ == "__main__":
    unittest.main()
