from app.nlu.handlers.base import BaseHandler, IntentContext
from app.nlu.handlers.base import OrchestratorResult


class GreetingHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        return OrchestratorResult(
            text=(
                "Hello! I can answer questions about EN 12464-1 lighting "
                "requirements (maintained illuminance, uniformity, glare limits, "
                "etc.). Ask me about a space, e.g. 'corridor lighting levels'."
            ),
            intent="greeting",
        )