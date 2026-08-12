from app.nlu.handlers.base import BaseHandler, IntentContext
from app.nlu.handlers.base import OrchestratorResult
from app.services.rag_service import CustomChatPromptTemplate


class GreetingHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        PROMPT = "Greet the user in a friendly and professional manner, and thank them for using the lighting-standards assistant. Keep it concise and polite."
        print("Handling greeting...")
        text = ctx.rag.answer(
            CustomChatPromptTemplate(
                question=ctx.query,
                citation=None,
                history=ctx.history_messages,
                intent_prompt=PROMPT
            )
        )
        print(f"GreetingHandler response: {text}")
        return OrchestratorResult(
            text=text,
            intent="greeting",
        )