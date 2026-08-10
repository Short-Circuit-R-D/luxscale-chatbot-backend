from app.utils.version import parse_year


class StandardsRepository:
    def __init__(self, db):
        self.collection = db["standards_clauses"]

    def find_by_id(self, mongo_id: str):
        return self.collection.find_one({"_id": mongo_id})

    def find_many(self, mongo_ids: list[str]):
        return list(self.collection.find({"_id": {"$in": list(mongo_ids)}}))

    def upsert(self, doc: dict):
        self.collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def find_siblings_for_versioning(self, standard_code: str, ref_number: str, exclude_id: str):
        return list(self.collection.find({
            "standard_metadata.standard_code": standard_code,
            "hierarchy.ref_number": ref_number,
            "_id": {"$ne": exclude_id},
        }))

    def latest_version_year(self, standard_code: str, ref_number: str) -> int | None:
        doc = self.collection.find_one(
            {
                "standard_metadata.standard_code": standard_code,
                "hierarchy.ref_number": ref_number,
                "standard_metadata.is_latest": True,
            },
            {"standard_metadata.version_year": 1},
        )
        if not doc:
            return None
        return parse_year(doc["standard_metadata"]["version_year"])

    def set_is_latest(self, mongo_id: str, value: bool):
        self.collection.update_one(
            {"_id": mongo_id}, {"$set": {"standard_metadata.is_latest": value}}
        )