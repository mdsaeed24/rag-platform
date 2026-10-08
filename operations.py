"""Process-local admission controls and bounded, content-free metrics."""

from collections import Counter
from threading import Lock
from time import perf_counter
import os

from starlette.responses import JSONResponse

ENDPOINTS = {"/", "/login", "/ask", "/health/live", "/health/ready", "/metrics"}
ANSWER_ROUTES = {"rag", "graph", "cache", "graph_cache", "abstain"}


def positive_integer_setting(name, default):
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def metrics_token_setting():
    token = os.getenv("METRICS_TOKEN", "")
    if token and (len(token) < 32 or token != token.strip()):
        raise ValueError("METRICS_TOKEN must contain at least 32 characters without surrounding whitespace")
    return token


class OperationalMetrics:
    def __init__(self, *, ask_limit=4, login_limit=4):
        self._lock = Lock()
        self.limits = {"/ask": ask_limit, "/login": login_limit}
        self.active = Counter()
        self.requests = Counter()
        self.seconds = Counter()
        self.answers = Counter()

    def enter(self, endpoint):
        with self._lock:
            if self.active[endpoint] >= self.limits[endpoint]:
                return False
            self.active[endpoint] += 1
            return True

    def leave(self, endpoint):
        with self._lock:
            self.active[endpoint] -= 1

    def record_request(self, path, status, seconds):
        endpoint = path if path in ENDPOINTS else "other"
        status = status if 100 <= status <= 599 else 500
        with self._lock:
            self.requests[endpoint, status] += 1
            self.seconds[endpoint, status] += seconds

    def record_answer(self, route):
        if route not in ANSWER_ROUTES:
            raise ValueError("Unknown answer route")
        with self._lock:
            self.answers[route] += 1

    def render(self):
        with self._lock:
            lines = ["# TYPE rag_http_requests_total counter",
                     "# TYPE rag_http_request_duration_seconds summary"]
            for (endpoint, status), count in sorted(self.requests.items()):
                labels = f'endpoint="{endpoint}",status="{status}"'
                lines.extend([
                    f"rag_http_requests_total{{{labels}}} {count}",
                    f"rag_http_request_duration_seconds_count{{{labels}}} {count}",
                    f"rag_http_request_duration_seconds_sum{{{labels}}} {self.seconds[endpoint, status]:.6f}",
                ])
            lines.extend(["# TYPE rag_inflight_requests gauge", "# TYPE rag_inflight_limit gauge"])
            for endpoint, limit in sorted(self.limits.items()):
                lines.extend([f'rag_inflight_requests{{endpoint="{endpoint}"}} {self.active[endpoint]}',
                              f'rag_inflight_limit{{endpoint="{endpoint}"}} {limit}'])
            lines.append("# TYPE rag_answers_total counter")
            lines.extend(f'rag_answers_total{{route="{route}"}} {self.answers[route]}'
                         for route in sorted(ANSWER_ROUTES))
        return "\n".join(lines) + "\n"


class RequestControlsMiddleware:
    def __init__(self, app, *, metrics, max_body_bytes):
        self.app = app
        self.metrics = metrics
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        start = perf_counter()
        status = 500
        admitted = False

        async def observed_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            if scope["method"] == "POST" and path in self.metrics.limits:
                admitted = self.metrics.enter(path)
                if not admitted:
                    await JSONResponse({"detail": "Request capacity exhausted; retry later"},
                                       status_code=503, headers={"Retry-After": "1"})(scope, receive, observed_send)
                    return
            # Count actual streamed bytes rather than trusting Content-Length.
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    status = 499
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > self.max_body_bytes:
                    await JSONResponse({"detail": "Request body too large"}, status_code=413)(scope, receive, observed_send)
                    return
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
            delivered = False

            async def bounded_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, bounded_receive, observed_send)
        finally:
            if admitted:
                self.metrics.leave(path)
            self.metrics.record_request(path, status, perf_counter() - start)
