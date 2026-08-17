from app.nlu.handlers.base import BaseHandler, IntentContext, OrchestratorResult, attach_concept_simulator, catalog_answer
from app.nlu.intents import Intent


class LightingGuidanceHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        result = catalog_answer(ctx, Intent.LIGHTING_GUIDANCE, citation=None)
        return attach_concept_simulator(ctx, result)
