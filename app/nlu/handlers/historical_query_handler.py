from app.nlu.handlers.base import BaseHandler, IntentContext, OrchestratorResult, catalog_answer
from app.nlu.intents import Intent


class HistoricalQueryHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        entities = ctx.prediction.entities
        clauses = ctx.retrieval.search(
            query=ctx.query,
            version_year=entities.version_year,
            standard_code=entities.standard_code,
            ref_number=entities.ref_number,
            category_table_number=entities.category_table_number,
        )
        citations = ctx.retrieval.build_citations(clauses)
        return catalog_answer(ctx, Intent.HISTORICAL_QUERY, citation=citations)
