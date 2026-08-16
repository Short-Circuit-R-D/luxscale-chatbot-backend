from app.nlu.handlers.base import BaseHandler, IntentContext, OrchestratorResult, catalog_answer
from app.nlu.intents import Intent


class RefLookupHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        entities = ctx.prediction.entities
        # Tables can contain many activities — fetch more than default top_k
        clauses = ctx.retrieval.search(
            query=ctx.query,
            version_year=entities.version_year,
            standard_code=entities.standard_code,
            ref_number=entities.ref_number,
            category_table_number=entities.category_table_number,
            top_k=15,
        )
        citations = ctx.retrieval.build_citations(clauses)
        return catalog_answer(ctx, Intent.REF_LOOKUP, citation=citations)
