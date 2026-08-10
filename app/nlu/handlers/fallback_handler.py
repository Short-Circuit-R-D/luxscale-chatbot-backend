from app.nlu.handlers.base import BaseHandler, IntentContext
from app.nlu.handlers.base import OrchestratorResult


class FallbackHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        return OrchestratorResult(
            text=(
                "I'm not sure I understood that. I can help with EN 12464-1 "
                "lighting requirements — try asking about a space, e.g. "
                "'what is the maintained illuminance for corridors?'"
            ),
            intent="fallback",
        )