from app.nlu.handlers.base import BaseHandler
from app.nlu.handlers.clarify_handler import ClarifyHandler
from app.nlu.handlers.comparison_handler import ComparisonHandler
from app.nlu.handlers.fallback_handler import FallbackHandler
from app.nlu.handlers.general_handler import GeneralHandler
from app.nlu.handlers.get_simulator_handler import GetSimulatorHandler
from app.nlu.handlers.greeting_handler import GreetingHandler
from app.nlu.handlers.historical_query_handler import HistoricalQueryHandler
from app.nlu.handlers.lighting_guidance_handler import LightingGuidanceHandler
from app.nlu.handlers.ref_lookup_handler import RefLookupHandler
from app.nlu.handlers.scope_handler import ScopeHandler
from app.nlu.handlers.standard_query_handler import StandardQueryHandler
from app.nlu.intents import Intent


def build_registry() -> dict[str, BaseHandler]:
    return {
        Intent.STANDARD_QUERY.value: StandardQueryHandler(),
        Intent.HISTORICAL_QUERY.value: HistoricalQueryHandler(),
        Intent.COMPARISON.value: ComparisonHandler(),
        Intent.REF_LOOKUP.value: RefLookupHandler(),
        Intent.GREETING.value: GreetingHandler(),
        Intent.GENERAL.value: GeneralHandler(),
        Intent.LIGHTING_GUIDANCE.value: LightingGuidanceHandler(),
        Intent.CLARIFY.value: ClarifyHandler(),
        Intent.OUT_OF_SCOPE.value: ScopeHandler(),
        Intent.FALLBACK.value: FallbackHandler(),
        Intent.GET_SIMULATOR.value: GetSimulatorHandler(),
    }
