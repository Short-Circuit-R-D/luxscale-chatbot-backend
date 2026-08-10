from app.api.schemas.ingest_schema import ClauseDocument


class StoredDocument(ClauseDocument):
    """Same full clause shape for the raw bulk-store endpoint.

    `model_dump(by_alias=True)` reproduces the wire document with `_id` for
    storage in Mongo.
    """