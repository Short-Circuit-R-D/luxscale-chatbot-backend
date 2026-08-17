from app.nlu.handlers.base import BaseHandler, IntentContext, OrchestratorResult, catalog_answer
from app.nlu.intents import Intent
from app.services.simulator_service import SimulatorSpec


def _clarify_text(candidates: list[SimulatorSpec]) -> str:
    if not candidates:
        return (
            "I don't have a matching lighting simulator for that. "
            "Tell me which concept you want to explore (for example UGR or illuminance)."
        )
    lines = "\n".join(f"- **{spec.title}** (`{spec.id}`)" for spec in candidates)
    return (
        "Which simulator should I open?\n\n"
        f"{lines}"
    )


class GetSimulatorHandler(BaseHandler):
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        payload, candidates = ctx.simulator.resolve_for_direct(
            ctx.query,
            simulator_id=ctx.prediction.entities.simulator_id,
        )
        if payload is None:
            return OrchestratorResult(
                text=_clarify_text(candidates),
                intent=Intent.GET_SIMULATOR.value,
            )
        result = catalog_answer(ctx, Intent.GET_SIMULATOR, citation=None)
        result.simulator = payload.as_dict()
        return result
