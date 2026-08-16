from app.nlu.handlers.base import BaseHandler, IntentContext, OrchestratorResult, catalog_answer
from app.nlu.intents import Intent


class ComparisonHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        entities = ctx.prediction.entities
        code = entities.standard_code
        ref = entities.ref_number
        table = entities.category_table_number

        if entities.version_year:
            # Edition/year filter present: retrieve current + named year sides
            latest = ctx.retrieval.search(
                query=ctx.query,
                standard_code=code,
                ref_number=ref,
                category_table_number=table,
                top_k=5,
                force_latest=True,
            )
            other = ctx.retrieval.search(
                query=ctx.query,
                version_year=entities.version_year,
                standard_code=code,
                ref_number=ref,
                category_table_number=table,
                top_k=5,
            )
            clauses = latest + other
        else:
            # No edition mentioned: same standard/current edition —
            # compare categories, activities, or tasks within that edition
            clauses = ctx.retrieval.search(
                query=ctx.query,
                standard_code=code,
                ref_number=ref,
                category_table_number=table,
                top_k=10,
                force_latest=True,
            )

        seen: set[str] = set()
        merged = []
        for clause in clauses:
            if clause.mongo_id in seen:
                continue
            seen.add(clause.mongo_id)
            merged.append(clause)

        citations = ctx.retrieval.build_citations(merged)
        return catalog_answer(ctx, Intent.COMPARISON, citation=citations)
