from app.nlu.handlers.base import BaseHandler, IntentContext
from app.nlu.handlers.base import OrchestratorResult


class StandardQueryHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        entities = ctx.prediction.entities
        clauses = ctx.retrieval.search(
            query=ctx.query,
            standard_code=entities.standard_code,
            ref_number=entities.ref_number,
        )
        citations = ctx.retrieval.build_citations(clauses)
        text = ctx.rag.answer(
            question=ctx.query,
            citations=citations,
            history=ctx.history_messages,
        )
        return OrchestratorResult(text=text, intent="standard_query")