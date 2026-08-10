from app.nlu.handlers.base import BaseHandler, IntentContext
from app.nlu.handlers.base import OrchestratorResult


class ScopeHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        return OrchestratorResult(
            text=(
                "I only answer questions about EN 12464-1 lighting "
                "requirements (illuminance levels, uniformity, glare, UGR, "
                "Ra). Try something like 'what illuminance for offices?' "
                "or 'corridor lighting levels in EN 12464-1'."
            ),
            intent="out_of_scope",
        )