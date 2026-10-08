"""Small HTTP client. Identity and document authorization belong to FastAPI."""

import math
from urllib.parse import urlsplit

import httpx

ROUTES = {"rag", "cache", "graph", "graph_cache", "abstain"}


class ApiError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def backend_url(value):
    """Only operator configuration can select the credential destination."""
    if not isinstance(value, str) or any(char.isspace() for char in value):
        raise ValueError("Invalid RAG_API_URL")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        raise ValueError("Invalid RAG_API_URL") from None
    if (parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment or "\\" in value
            or (port is not None and not 1 <= port <= 65535)
            or (parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1", "::1"})):
        raise ValueError("RAG_API_URL requires HTTPS, or HTTP on localhost")
    return value.rstrip("/")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _chunk(value):
    return type(value) is int and value >= 0


def _score(value):
    return value is None or (type(value) in (int, float) and math.isfinite(value))


def valid_answer(data, question):
    if (not isinstance(data, dict) or data.get("question") != question
            or any(not _text(data.get(key)) for key in ("answer", "user_id", "tenant_id", "role", "cache_status", "routing_reason"))
            or not isinstance(data.get("route"), str) or data["route"] not in ROUTES):
        return False
    sources, edges = data.get("sources"), data.get("graph_evidence")
    if not isinstance(sources, list) or not isinstance(edges, list):
        return False
    if any(not isinstance(row, dict) or not _text(row.get("source"))
           or not _chunk(row.get("chunk_id")) or not _score(row.get("score")) for row in sources):
        return False
    citations = {(row["source"], row["chunk_id"]) for row in sources}
    if any(not isinstance(row, dict)
           or any(not _text(row.get(key)) for key in ("subject", "relation", "target", "source", "evidence"))
           or not _chunk(row.get("chunk_id"))
           or (row["source"], row["chunk_id"]) not in citations for row in edges):
        return False
    if data["route"] == "abstain":
        return not sources and not edges
    if not sources:
        return False
    return bool(edges) if data["route"] in {"graph", "graph_cache"} else not edges


class RagApi:
    def __init__(self, url, *, transport=None):
        self.url = backend_url(url)
        self.transport = transport

    def _request(self, method, path, *, token=None, body=None):
        headers = {"Authorization": "Bearer " + token} if token else {}
        try:
            with httpx.Client(timeout=httpx.Timeout(60, connect=5), follow_redirects=False,
                              trust_env=False, transport=self.transport) as client:
                response = client.request(method, self.url + path, json=body, headers=headers)
        except httpx.TimeoutException:
            raise ApiError("The request timed out. Please try again.") from None
        except httpx.RequestError:
            raise ApiError("Cannot connect to the API. Check that the backend is running.") from None
        if response.status_code != 200:
            messages = {
                401: "Incorrect username or password." if path == "/login" else "Your session expired or access was revoked. Sign in again.",
                403: "This request could not be authorized.",
                409: "The evidence changed during this request. Please try again.",
                413: "The request is too large. Please shorten it.",
                422: "Check your input and try again.",
                429: "Too many requests. Please wait before trying again.",
                502: "The answer service could not complete this request.",
                503: "The service is busy or unavailable. Please try again shortly.",
                504: "The answer service timed out. Please try again.",
            }
            raise ApiError(messages.get(response.status_code, "The API returned an unexpected response."), response.status_code)
        try:
            return response.json()
        except ValueError:
            raise ApiError("The API returned an invalid response.") from None

    def login(self, username, password):
        data = self._request("POST", "/login", body={"username": username, "password": password})
        if (not isinstance(data, dict) or data.get("token_type") != "bearer"
                or not _text(data.get("access_token"))
                or any(char.isspace() for char in data["access_token"])):
            raise ApiError("The API returned an invalid login response.")
        return data["access_token"]

    def ask(self, question, token):
        data = self._request("POST", "/ask", token=token, body={"question": question})
        if not valid_answer(data, question):
            raise ApiError("The API returned an invalid answer response.")
        return data

    def ready(self):
        try:
            data = self._request("GET", "/health/ready")
        except ApiError as exc:
            if exc.status == 503:
                return False
            raise
        if not isinstance(data, dict) or data.get("status") != "ready":
            raise ApiError("The API returned an invalid readiness response.")
        return True
