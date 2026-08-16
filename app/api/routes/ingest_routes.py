from flask import Blueprint, request, jsonify

from app.api.schemas.ingest_schema import ClauseDocument, IngestResultItem, IngestResponse
from app.api.schemas.store_schema import StoredDocument
from app.extensions import ingestion_service

ingest_bp = Blueprint("ingest", __name__, url_prefix="/api/v1")


@ingest_bp.route("/ingest", methods=["POST"])
def ingest():
    body = request.get_json(force=True)
    if isinstance(body, dict) and "documents" in body:
        docs = body["documents"]
    elif isinstance(body, list):
        docs = body
    else:
        docs = [body]

    results = []
    for raw_doc in docs:
        try:
            validated = ClauseDocument.model_validate(raw_doc)
            result = ingestion_service.ingest_document(validated.model_dump())
            results.append(IngestResultItem(**result))
            print(f"Successfully ingested document with mongo_id: {result.get('mongo_id')} and status: {result.get('status')}")
        except Exception as e:
            mongo_id = raw_doc.get("_id", "unknown") if isinstance(raw_doc, dict) else "unknown"
            results.append(IngestResultItem(mongo_id=mongo_id, status="failed", error=str(e)))
            print(f"Failed to ingest document with mongo_id: {mongo_id}. Error: {str(e)}")

    return jsonify(IngestResponse(results=results).model_dump()), 200


@ingest_bp.route("/documents", methods=["POST"])
def store_documents():
    """Bulk raw store of full clause documents. Idempotent by `_id`;
    identical re-submissions are `skipped_unchanged` (no writes). No
    embedding, no Qdrant writes."""
    body = request.get_json(force=True)
    if isinstance(body, dict) and "documents" in body:
        docs = body["documents"]
    elif isinstance(body, list):
        docs = body
    else:
        docs = [body]

    results = []
    for raw_doc in docs:
        try:
            validated = StoredDocument.model_validate(raw_doc)
            result = ingestion_service.store_document(validated.model_dump(by_alias=True))
            results.append(IngestResultItem(**result))
        except Exception as e:
            results.append(IngestResultItem(
                mongo_id=raw_doc.get("_id", "unknown"), status="failed", error=str(e)
            ))

    return jsonify(IngestResponse(results=results).model_dump()), 200