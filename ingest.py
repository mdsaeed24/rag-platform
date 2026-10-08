from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from acl_config import DOCUMENT_ACLS
from authorization import validate_document_acl


COLLECTION_NAME = "company_documents"


def load_document(file_path):
    return Path(file_path).read_text(encoding="utf-8")


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    # Validate every ACL and input file before modifying the existing collection.
    for source, acl in DOCUMENT_ACLS.items():
        validate_document_acl(source, acl)
        if not Path(source).is_file():
            raise ValueError(f"Document {source} does not exist")
    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding model loaded.")

    client = QdrantClient(path="qdrant_data")

    points = []
    point_id = 0

    # Process every document defined in our ACL config
    for file_path, acl in DOCUMENT_ACLS.items():

        # Security check:
        # Never ingest a document without a tenant.
        if not acl.get("tenant_id"):
            raise ValueError(
                f"Document {file_path} has no tenant_id"
            )

        print(f"\nProcessing: {file_path}")

        document = load_document(file_path)

        chunks = chunk_text(document)

        print(f"Created {len(chunks)} chunks.")

        embeddings = model.encode(chunks)

        for chunk_index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "source": file_path,
                    "chunk_id": chunk_index,

                    # ACL metadata
                    "tenant_id": acl["tenant_id"],
                    "allowed_roles": acl["allowed_roles"],
                    "classification": acl["classification"],
                },
            )

            points.append(point)

            point_id += 1

    # Prepare the new chunks successfully before replacing the old collection.
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print("\n================================")
    print("Ingestion complete!")
    print(f"Stored {len(points)} chunks.")
    print("================================")

    client.close()
