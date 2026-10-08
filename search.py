from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client import models
from pathlib import Path
from threading import Lock


COLLECTION_NAME = "company_documents"
QDRANT_PATH = Path(__file__).resolve().parent / "qdrant_data"
_storage_lock = Lock()

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")


def build_acl_filter(tenant_id, role):
    if (
        not isinstance(tenant_id, str) or not tenant_id.strip()
        or not isinstance(role, str) or role not in {"employee", "hr"}
    ):
        raise ValueError("A valid tenant and role are required for retrieval")
    # SECURITY:
    # A document must belong to the user's tenant
    # AND allow the user's role.
    return models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(
                    value=tenant_id
                ),
            ),
            models.FieldCondition(
                key="allowed_roles",
                match=models.MatchValue(
                    value=role
                ),
            ),
        ]
    )

def embed_query(query):
    return model.encode(query)


def local_dependencies_ready():
    """Inspect existing collection metadata only; never embed or call a provider."""
    if model.get_embedding_dimension() != 384 or not (QDRANT_PATH / "meta.json").is_file():
        return False
    if not _storage_lock.acquire(timeout=1):
        return False
    try:
        client = QdrantClient(path=str(QDRANT_PATH))
        try:
            if not client.collection_exists(COLLECTION_NAME):
                return False
            info = client.get_collection(COLLECTION_NAME)
            vectors = info.config.params.vectors
            return (isinstance(vectors, models.VectorParams) and vectors.size == 384
                    and vectors.distance == models.Distance.COSINE and (info.points_count or 0) > 0)
        finally:
            client.close()
    finally:
        _storage_lock.release()


def search_documents(query, tenant_id, role, limit=3, *, query_embedding=None):
    acl_filter = build_acl_filter(tenant_id, role)
    if query_embedding is None:
        query_embedding = embed_query(query)
    # Local Qdrant allows one owner of its storage directory at a time.
    # Serialize opens within this process and always release the lock on errors.
    with _storage_lock:
        client = QdrantClient(path=str(QDRANT_PATH))
        try:
            return client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding.tolist(),
                query_filter=acl_filter,
                limit=limit,
            ).points
        finally:
            client.close()


class GraphExpansionLimitError(Exception):
    pass


def load_graph_documents(tenant_id, role, *, max_chunks=64):
    """Bounded graph expansion with the same mandatory database ACL filter.

    The first graph increment scans only this identity's permitted chunks.
    Refuse partial graphs when the corpus exceeds the request-local budget.
    """
    acl_filter = build_acl_filter(tenant_id, role)
    with _storage_lock:
        client = QdrantClient(path=str(QDRANT_PATH))
        try:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=acl_filter,
                limit=max_chunks + 1,
                with_payload=True,
                with_vectors=False,
            )
            if len(points) > max_chunks or next_offset is not None:
                raise GraphExpansionLimitError("Graph expansion budget exceeded")
            return points
        finally:
            client.close()


if __name__ == "__main__":
    question = "What does the Globex CEO earn?"

    results = search_documents(
        query=question,
        tenant_id="acme",
        role="hr",
        limit=3,
    )

    print("\nQuestion:")
    print(question)

    print("\nUser:")
    print("Tenant: acme")
    print("Role: hr")

    print("\nSearch results:")

    for i, result in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.payload['source']}")
        print(f"Tenant: {result.payload['tenant_id']}")
        print(f"Roles: {result.payload['allowed_roles']}")
        print(f"Classification: {result.payload['classification']}")
        print()
        print(result.payload["text"])
