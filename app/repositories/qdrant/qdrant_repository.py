from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from langchain_qdrant import QdrantVectorStore

CONTENT_KEY = "page_content"
METADATA_KEY = "metadata"
VECTOR_SIZE = 1024  # bge-m3


class QdrantRepository:
    """langchain-qdrant >=1.x integration.

    The LC store reads point payloads via `metadata`/`page_content` keys
    (`_document_from_point` does payload.get(metadata_payload_key)), so every
    point is stored nested: {"page_content": "", "metadata": <flat payload>}.
    Document.metadata then equals our lean payload. All writes/patches go
    through the same raw client, so the convention is enforced in one place.
    """

    def __init__(self, client, collection_name: str, embeddings_loader):
        self.client = client
        self.collection_name = collection_name
        self._embeddings_loader = embeddings_loader
        self._store = None

    def _ensure_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        self.client.create_payload_index(self.collection_name, "metadata.is_latest", PayloadSchemaType.BOOL)
        self.client.create_payload_index(self.collection_name, "metadata.version_year", PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection_name, "metadata.standard_code", PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection_name, "metadata.ref_number", PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(
            self.collection_name, "metadata.category_table_number", PayloadSchemaType.KEYWORD
        )

    def _store_instance(self) -> QdrantVectorStore:
        if self._store is None:
            self._ensure_collection()
            self._store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self._embeddings_loader(),
            )
        return self._store

    # --- writes (raw client, deterministic ids, no re-embedding) ---

    def upsert_point(self, point_id: str, vector: list[float], payload: dict):
        self._ensure_collection()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=point_id,
                vector=vector,
                payload={CONTENT_KEY: "", METADATA_KEY: payload},
            )],
        )

    def patch_metadata(self, point_id: str, updates: dict):
        """Read-modify-write on the nested metadata dict (versioning patches)."""
        self._ensure_collection()
        points = self.client.retrieve(
            collection_name=self.collection_name, ids=[point_id], with_payload=True
        )
        if not points:
            return
        metadata = dict(points[0].payload.get(METADATA_KEY, {}))
        metadata.update(updates)
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={METADATA_KEY: metadata},
            points=[point_id],
        )

    def point_exists(self, point_id: str) -> bool:
        self._ensure_collection()
        return bool(
            self.client.retrieve(
                collection_name=self.collection_name, ids=[point_id]
            )
        )

    def get_payload(self, point_id: str) -> dict | None:
        """Returns the nested metadata dict of a point (or the raw/legacy flat
        payload if present) — None when the point doesn't exist."""
        self._ensure_collection()
        points = self.client.retrieve(
            collection_name=self.collection_name, ids=[point_id], with_payload=True
        )
        if not points:
            return None
        payload = points[0].payload
        meta = payload.get(METADATA_KEY)
        if isinstance(meta, dict):
            return dict(meta)
        return dict(payload)

    def count(self) -> int:
        self._ensure_collection()
        return self.client.count(collection_name=self.collection_name).count

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

    # --- reads (LC store) ---

    def search(self, embedding: list[float], qdrant_filter, top_k: int):
        return self._store_instance().similarity_search_with_score_by_vector(
            embedding=embedding, k=top_k, filter=qdrant_filter
        )