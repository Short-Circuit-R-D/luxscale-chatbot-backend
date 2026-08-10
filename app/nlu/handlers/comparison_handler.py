from app.nlu.handlers.base import BaseHandler, IntentContext
from app.nlu.handlers.base import OrchestratorResult


class ComparisonHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        return OrchestratorResult(
            text=(
                "Comparing editions isn't supported yet. Ask about one "
                "specific edition instead, e.g. 'EN 12464-1 2019, corridor "
                "light levels' or 'EN 12464-1 2021, classroom illuminance'."
            ),
            intent="comparison",
        )