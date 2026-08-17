from app.nlu.handlers.base import BaseHandler, IntentContext, OrchestratorResult, attach_concept_simulator, catalog_answer
from app.nlu.intents import Intent


class GeneralHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        result = catalog_answer(ctx, Intent.GENERAL, citation=None)
        return attach_concept_simulator(ctx, result)
