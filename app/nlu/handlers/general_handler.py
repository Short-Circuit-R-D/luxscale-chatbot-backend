from app.nlu.handlers.base import BaseHandler, IntentContext, OrchestratorResult
from app.services.rag_service import CustomChatPromptTemplate


class GeneralHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:

        text = ctx.rag.answer(
            CustomChatPromptTemplate(
                question=ctx.query,
                citation=None,
                history=ctx.history_messages,
                intent_prompt=None
            )
        )

        print("GeneralHandler: ", text)

        return OrchestratorResult(
            text=text,
            intent="general",
        )