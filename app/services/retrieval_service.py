from qdrant_client.models import FieldCondition, Filter, MatchValue


class RetrievedClause:
    def __init__(self, mongo_id: str, score: float, payload: dict, document: dict | None):
        self.mongo_id = mongo_id
        self.score = score
        self.payload = payload
        self.document = document

    def citation(self) -> str:
        if self.document:
            text = self.document.get("searchable_text", "")
        else:
            text = self.payload.get("activity", "")
        return (
            f"[{self.payload.get('standard_code', '?')} | "
            f"{self.payload.get('version_year', '?')} | ref "
            f"{self.payload.get('ref_number', '?')} | "
            f"{self.payload.get('category_table_number', '?')} | page "
            f"{self.payload.get('page', '?')}]: {text}"
        )


class RetrievalService:
    def __init__(self, embedding_service, qdrant_repo, standards_repo, top_k: int = 5):
        self.embedder = embedding_service
        self.qdrant_repo = qdrant_repo
        self.standards_repo = standards_repo
        self.top_k = top_k

    def search(
        self,
        query: str,
        version_year: str | None = None,
        standard_code: str | None = None,
        ref_number: str | None = None,
        category_table_number: str | None = None,
        top_k: int | None = None,
        force_latest: bool = True,
    ) -> list[RetrievedClause]:
        embedding = self.embedder.embed(query)
        qfilter = self._build_filter(
            version_year,
            standard_code,
            ref_number,
            category_table_number,
            force_latest,
        )
        hits = self.qdrant_repo.search(embedding, qfilter, top_k or self.top_k)

        mongo_ids = [h[0].metadata.get("mongo_id") for h in hits]
        by_id = {d["_id"]: d for d in self.standards_repo.find_many(mongo_ids)}

        clauses = []
        for document, score in hits:
            mongo_id = document.metadata.get("mongo_id", "")
            clauses.append(
                RetrievedClause(
                    mongo_id=mongo_id,
                    score=score,
                    payload=document.metadata,
                    document=by_id.get(mongo_id),
                )
            )
        return clauses

    def build_citations(self, clauses: list[RetrievedClause]) -> str:
        return "\n\n".join(c.citation() for c in clauses)

    def _build_filter(
        self,
        version_year,
        standard_code,
        ref_number,
        category_table_number,
        force_latest: bool = True,
    ) -> Filter:
        # payload is nested under the "metadata" key (langchain-qdrant 1.x
        # convention) -> filter conditions use dotted paths.
        def fields(key: str) -> str:
            return f"metadata.{key}"

        must: list[FieldCondition] = []
        if version_year is not None:
            must.append(
                FieldCondition(key=fields("version_year"), match=MatchValue(value=version_year))
            )
        elif force_latest:
            must.append(
                FieldCondition(key=fields("is_latest"), match=MatchValue(value=True))
            )
        if standard_code:
            must.append(
                FieldCondition(key=fields("standard_code"), match=MatchValue(value=standard_code))
            )

        # Table numbers live in category_table_number (e.g. "6.2"); clause ids
        # in ref_number (e.g. "6.2.1"). Match either field so "table 6.2" works
        # even if the classifier put it in ref_number.
        locators = {
            v.strip()
            for v in (ref_number, category_table_number)
            if v and str(v).strip()
        }
        should: list[FieldCondition] = []
        for value in locators:
            should.append(
                FieldCondition(key=fields("ref_number"), match=MatchValue(value=value))
            )
            should.append(
                FieldCondition(
                    key=fields("category_table_number"),
                    match=MatchValue(value=value),
                )
            )

        if should:
            # Nested should inside must => at least one locator field must match
            must.append(Filter(should=should))

        return Filter(must=must) if must else Filter(must=[])
