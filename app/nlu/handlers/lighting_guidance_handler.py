from app.nlu.handlers.base import BaseHandler, IntentContext, OrchestratorResult, catalog_answer
from app.nlu.intents import Intent


class LightingGuidanceHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        return catalog_answer(ctx, Intent.LIGHTING_GUIDANCE, citation=None)
