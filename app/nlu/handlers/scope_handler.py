from app.nlu.handlers.base import BaseHandler, IntentContext, OrchestratorResult, catalog_answer
from app.nlu.intents import Intent


class ScopeHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        return catalog_answer(ctx, Intent.OUT_OF_SCOPE, citation=None)
