from qdrant_client import QdrantClient

from app.config import Config


def create_qdrant_client() -> QdrantClient:
    """Connect via QDRANT_URL; fall back to QDRANT_URL_LOCALHOST on failure."""
    candidates = [Config.QDRANT_URL]
    fallback = Config.QDRANT_URL_LOCALHOST
    if fallback and fallback != Config.QDRANT_URL:
        candidates.append(fallback)

    errors: list[str] = []
    for url in candidates:
        try:
            client = QdrantClient(url=url, check_compatibility=False)
            client.get_collections()
            return client
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise ConnectionError(
        "Could not connect to Qdrant. Tried:\n" + "\n".join(errors)
    )
