from datetime import datetime, timezone

from app.utils.hashing import content_hash, doc_json_hash
from app.utils.ids import derive_point_id
from app.utils.version import parse_year


class IngestionService:
    def __init__(self, standards_repo, qdrant_repo, embedding_service):
        self.standards_repo = standards_repo
        self.qdrant_repo = qdrant_repo
        self.embedder = embedding_service

    def ingest_document(self, doc: dict) -> dict:
        mongo_id = doc["id"]
        new_hash = content_hash(doc["searchable_text"])
        existing = self.standards_repo.find_by_id(mongo_id)

        if existing and existing.get("content_hash") == new_hash:
            return self._check_vector_state(doc, mongo_id, new_hash, existing)

        is_latest = self._declared_latest(doc)
        doc["standard_metadata"] = {**doc["standard_metadata"], "is_latest": is_latest}

        now = datetime.now(timezone.utc).isoformat()
        mongo_doc = {
            **doc,
            "_id": mongo_id,
            "content_hash": new_hash,
            "updated_at": now,
            "created_at": existing["created_at"] if existing else now,
        }
        self.standards_repo.upsert(mongo_doc)

        self._write_vector(doc, mongo_id, new_hash)

        status = "updated" if existing else "inserted"
        return {"mongo_id": mongo_id, "status": status}

    def _check_vector_state(self, doc: dict, mongo_id: str, new_hash: str,
                            existing: dict) -> dict:
        """Mongo already matches. Point missing or stale in Qdrant (dropped
        collection, renamed collection, deleted point, legacy flat payload)
        -> re-embed and backfill the point WITHOUT touching Mongo."""
        point_id = derive_point_id(mongo_id)
        stored = self.qdrant_repo.get_payload(point_id)
        if stored is not None and stored.get("content_hash") == new_hash:
            return {"mongo_id": mongo_id, "status": "skipped_unchanged"}

        doc["standard_metadata"] = {
            **doc["standard_metadata"],
            "is_latest": existing["standard_metadata"]["is_latest"],
        }
        self._write_vector(doc, mongo_id, new_hash)
        return {"mongo_id": mongo_id, "status": "reindexed"}

    def _write_vector(self, doc: dict, mongo_id: str, new_hash: str):
        vector = self.embedder.embed(doc["searchable_text"])
        point_id = derive_point_id(mongo_id)
        self.qdrant_repo.upsert_point(point_id, vector, self._payload(doc, mongo_id, new_hash))
        if doc["standard_metadata"]["is_latest"]:
            self._supersede_older_versions(doc)

    def _declared_latest(self, doc: dict) -> bool:
        incoming_year = parse_year(doc["standard_metadata"]["version_year"])
        latest_year = self.standards_repo.latest_version_year(
            doc["standard_metadata"]["standard_code"],
            doc["hierarchy"]["ref_number"],
        )
        if incoming_year is not None and latest_year is not None and incoming_year < latest_year:
            return False
        return doc["standard_metadata"]["is_latest"]

    def _supersede_older_versions(self, doc: dict):
        siblings = self.standards_repo.find_siblings_for_versioning(
            doc["standard_metadata"]["standard_code"],
            doc["hierarchy"]["ref_number"],
            doc["id"],
        )
        for sibling in siblings:
            if sibling.get("standard_metadata", {}).get("is_latest"):
                self.standards_repo.set_is_latest(sibling["_id"], False)
                self.qdrant_repo.patch_metadata(
                    derive_point_id(sibling["_id"]), {"is_latest": False}
                )

    def _payload(self, doc: dict, mongo_id: str, new_hash: str) -> dict:
        return {
            "mongo_id": mongo_id,
            "standard_code": doc["standard_metadata"]["standard_code"],
            "version_year": doc["standard_metadata"]["version_year"],
            "is_latest": doc["standard_metadata"]["is_latest"],
            "category_table_number": doc["hierarchy"]["category_table_number"],
            "ref_number": doc["hierarchy"]["ref_number"],
            "activity": doc["activity"],
            "page": doc["hierarchy"]["page"],
            "content_hash": new_hash,
        }

    def store_document(self, doc: dict) -> dict:
        mongo_id = doc.get("_id")
        if not mongo_id:
            raise ValueError("_id is required")

        new_hash = doc_json_hash(doc)
        existing = self.standards_repo.find_by_id(mongo_id)
        now = datetime.now(timezone.utc).isoformat()

        if existing:
            if existing.get("content_hash") == new_hash:
                return {"mongo_id": mongo_id, "status": "skipped_unchanged"}
            self.standards_repo.upsert({
                **doc,
                "content_hash": new_hash,
                "created_at": existing["created_at"],
                "updated_at": now,
            })
            return {"mongo_id": mongo_id, "status": "updated"}

        self.standards_repo.upsert({
            **doc,
            "content_hash": new_hash,
            "created_at": now,
            "updated_at": now,
        })
        return {"mongo_id": mongo_id, "status": "inserted"}