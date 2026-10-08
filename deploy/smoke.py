"""Offline deployment smoke using isolated fictional fixtures and mocked answers."""

import argparse
import copy
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = {
    "data/acme/handbook.txt": "Full-time Acme employees receive 20 days of paid annual leave per calendar year.",
    "data/acme/salaries.txt": "The Acme CEO earns $240,000 per year.\nAcme software engineers earn $110,000 per year.",
    "data/globex/handbook.txt": "Full-time Globex employees receive 25 days of paid annual leave per calendar year.",
    "data/globex/salaries.txt": "The Globex CEO earns $310,000 per year.\nGlobex software engineers earn $125,000 per year.",
}
CASES = (
    ("bob_rag", "bob", "What does the CEO earn?", 200, "rag", 1),
    ("bob_cache", "bob", "What does the CEO earn?", 200, "cache", 0),
    ("bob_graph", "bob", "Compare CEO and software engineer salaries.", 200, "graph", 1),
    ("bob_graph_cache", "bob", "Compare CEO and software engineer salaries.", 200, "graph_cache", 0),
    ("alice_abstains", "alice", "What does the CEO earn?", 200, "abstain", 0),
    ("carol_rag", "carol", "What does the CEO earn?", 200, "rag", 1),
    ("foreign_tenant_abstains", "bob", "What does the Globex CEO earn?", 200, "abstain", 0),
    ("missing_token", None, "What does the CEO earn?", 401, None, 0),
)


def run_smoke():
    # Override credentials before application import and block socket connections
    # as a second barrier against accidental network calls during the smoke.
    settings = {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "DEEPSEEK_API_KEY": "offline-smoke-no-provider-access",
                       "JWT_SECRET_KEY": secrets.token_urlsafe(48), "METRICS_TOKEN": "",
                       "ANSWER_CACHE_ENABLED": "true", "GRAPH_RAG_ENABLED": "true",
                       "GRAPH_ANSWER_CACHE_ENABLED": "true", "QUERY_ROUTING_ENABLED": "true",
                       "ROUTING_MIN_SCORE": "0.25", "MAX_REQUEST_BODY_BYTES": "16384",
                       "MAX_INFLIGHT_ASK": "4", "MAX_INFLIGHT_LOGIN": "4", "LLM_TIMEOUT_SECONDS": "30"}
    with patch.dict(os.environ, settings):
        return _isolated_smoke()


def _isolated_smoke():
    with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden in offline smoke")), patch("socket.socket.connect_ex", side_effect=AssertionError("Network forbidden in offline smoke")):
        from fastapi.testclient import TestClient
        from qdrant_client import QdrantClient, models
        import llm
        import main
        import search
        from acl_config import DOCUMENT_ACLS
        from semantic_cache import SemanticAnswerCache

        rows = []
        with tempfile.TemporaryDirectory(prefix="rag-deploy-smoke-") as directory, patch.object(search, "QDRANT_PATH", Path(directory) / "qdrant_data"), patch.object(main, "answer_cache", SemanticAnswerCache()), patch.object(main, "log_event"), patch.object(main, "metrics_token", ""), patch.object(llm.client.chat.completions, "create", return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Offline mocked answer"), finish_reason="stop")])) as completion:
            vectors = search.model.encode(list(FIXTURES.values()))
            storage = QdrantClient(path=str(search.QDRANT_PATH))
            try:
                storage.create_collection(search.COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
                storage.upsert(search.COLLECTION_NAME, points=[models.PointStruct(id=index,
                    vector=vector.tolist(), payload={**copy.deepcopy(DOCUMENT_ACLS[source]),
                        "source": source, "chunk_id": 0, "text": text})
                    for index, ((source, text), vector) in enumerate(zip(FIXTURES.items(), vectors))])
            finally:
                storage.close()
            with TestClient(main.app) as client:
                tokens = {}
                for username in ("alice", "bob", "carol"):
                    response = client.post("/login", json={"username": username, "password": username + "123"})
                    response.raise_for_status()
                    tokens[username] = response.json()["access_token"]
                for case_id, username, question, expected_status, expected_route, expected_calls in CASES:
                    completion.reset_mock()
                    response = client.post("/ask", json={"question": question},
                        headers={"Authorization": "Bearer " + tokens[username]} if username else {})
                    route = response.json().get("route")
                    rows.append({"id": case_id, "status": response.status_code, "route": route,
                        "mock_completion_calls": completion.call_count,
                        "passed": response.status_code == expected_status and route == expected_route
                                  and completion.call_count == expected_calls})
                for path, expected_status in (("/health/live", 200), ("/health/ready", 200), ("/metrics", 404)):
                    response = client.get(path)
                    rows.append({"id": path, "status": response.status_code,
                                 "passed": response.status_code == expected_status})
        return {"mode": "offline_isolated_mocked", "real_provider_calls": 0,
                "passed": sum(row["passed"] for row in rows), "total": len(rows), "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "deployment-smoke.json")
    args = parser.parse_args()
    report = run_smoke()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
