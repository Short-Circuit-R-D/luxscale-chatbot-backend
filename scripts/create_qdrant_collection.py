"""Idempotent Qdrant collection creation. Usage:
    python scripts/create_qdrant_collection.py [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from app.config import Config

SIZE = 1024  # bge-m3


def create():
    force = "--force" in sys.argv
    client = QdrantClient(url=Config.QDRANT_URL)
    name = Config.QDRANT_COLLECTION

    existing = {c.name for c in client.get_collections().collections}
    if name in existing and not force:
        print(f"collection '{name}' exists (use --force to recreate)")
        return

    if name in existing:
        client.delete_collection(name)
        print(f"dropped '{name}'")

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=SIZE, distance=Distance.COSINE),
    )
    client.create_payload_index(name, "metadata.is_latest", PayloadSchemaType.BOOL)
    client.create_payload_index(name, "metadata.version_year", PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "metadata.standard_code", PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "metadata.ref_number", PayloadSchemaType.KEYWORD)
    print(f"collection '{name}' created (size={SIZE}, cosine)")


if __name__ == "__main__":
    create()