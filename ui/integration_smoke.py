"""Exercise the UI HTTP client against FastAPI offline; run in the backend venv."""

import copy
import json
import os
from pathlib import Path
import secrets
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def run_smoke():
    settings = {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "DEEPSEEK_API_KEY": "offline-ui-no-provider-access", "JWT_SECRET_KEY": secrets.token_urlsafe(48),
                "METRICS_TOKEN": "", "ANSWER_CACHE_ENABLED": "true", "GRAPH_RAG_ENABLED": "true",
                "GRAPH_ANSWER_CACHE_ENABLED": "true", "QUERY_ROUTING_ENABLED": "true", "ROUTING_MIN_SCORE": "0.25"}
    with patch.dict(os.environ, settings), patch("socket.socket.connect", side_effect=AssertionError("Offline UI smoke")), patch("socket.socket.connect_ex", side_effect=AssertionError("Offline UI smoke")):
        import httpx
        from fastapi.testclient import TestClient
        from qdrant_client import QdrantClient, models
        from acl_config import DOCUMENT_ACLS
        from deploy.smoke import FIXTURES
        import llm
        import main
        import search
        from semantic_cache import SemanticAnswerCache
        from ui.api import ApiError, RagApi

        rows = []
        with tempfile.TemporaryDirectory(prefix="rag-ui-smoke-") as directory, patch.object(search, "QDRANT_PATH", Path(directory) / "vectors"), patch.object(main, "answer_cache", SemanticAnswerCache()), patch.object(main, "log_event"), patch.object(llm.client.chat.completions, "create", return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Offline integration answer"), finish_reason="stop")])) as completion:
            vectors = search.model.encode(list(FIXTURES.values()))
            storage = QdrantClient(path=str(search.QDRANT_PATH))
            try:
                storage.create_collection(search.COLLECTION_NAME, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
                storage.upsert(search.COLLECTION_NAME, points=[models.PointStruct(id=index, vector=vector.tolist(),
                    payload={**copy.deepcopy(DOCUMENT_ACLS[source]), "source": source, "chunk_id": 0, "text": text})
                    for index, ((source, text), vector) in enumerate(zip(FIXTURES.items(), vectors))])
            finally:
                storage.close()
            with TestClient(main.app) as app:
                def forward(request):
                    upstream = app.request(request.method, request.url.path, content=request.content, headers=dict(request.headers))
                    # Starlette's TestClient uses httpx2 in the pinned backend;
                    # translate its response into the UI client's httpx type.
                    return httpx.Response(upstream.status_code, headers=dict(upstream.headers), content=upstream.content)
                api = RagApi("http://127.0.0.1:8001", transport=httpx.MockTransport(forward))
                tokens = {name: api.login(name, name + "123") for name in ("bob", "alice", "carol")}
                cases = (("bob_rag", "bob", "What does the CEO earn?", "rag", 1),
                         ("bob_cache", "bob", "What does the CEO earn?", "cache", 0),
                         ("alice_abstains", "alice", "What does the CEO earn?", "abstain", 0),
                         ("carol_rag", "carol", "What does the CEO earn?", "rag", 1),
                         ("bob_graph", "bob", "Compare CEO and software engineer salaries.", "graph", 1),
                         ("bob_graph_cache", "bob", "Compare CEO and software engineer salaries.", "graph_cache", 0),
                         ("foreign_tenant_abstains", "bob", "What does the Globex CEO earn?", "abstain", 0))
                for case_id, name, question, route, calls in cases:
                    completion.reset_mock()
                    answer = api.ask(question, tokens[name])
                    foreign = "acme" if name == "carol" else "globex"
                    context_safe = all(foreign not in json.dumps(call.kwargs["messages"]).casefold()
                                       for call in completion.call_args_list)
                    rows.append({"id": case_id, "passed": answer["route"] == route and completion.call_count == calls and context_safe,
                                 "route": answer["route"], "mock_completion_calls": completion.call_count,
                                 "outbound_tenant_check_passed": context_safe})
                try:
                    api.ask("What does the CEO earn?", "invalid-token")
                except ApiError as exc:
                    denied = exc.status == 401
                else:
                    denied = False
                rows.extend([{"id": "invalid_token", "passed": denied}, {"id": "readiness", "passed": api.ready()}])
        return {"mode": "ui_http_client_to_fastapi_offline", "real_provider_calls": 0,
                "passed": sum(row["passed"] for row in rows), "total": len(rows), "cases": rows}


if __name__ == "__main__":
    report = run_smoke()
    output = Path(__file__).resolve().parents[1] / "dist/ui-integration-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
