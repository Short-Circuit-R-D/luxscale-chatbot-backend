from app.nlu.handlers.base import BaseHandler, IntentContext
from app.nlu.handlers.base import OrchestratorResult
from app.services.rag_service import CustomChatPromptTemplate


class HistoricalQueryHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        entities = ctx.prediction.entities
        clauses = ctx.retrieval.search(
            query=ctx.query,
            version_year=entities.version_year,
            standard_code=entities.standard_code,
            ref_number=entities.ref_number,
        )
        citations = ctx.retrieval.build_citations(clauses)
        text = ctx.rag.answer(
            CustomChatPromptTemplate(
                question=ctx.query,
                citation=citations,
                history=ctx.history_messages,
                intent_prompt=None
            )
        )
        return OrchestratorResult(text=text, intent="historical_query")