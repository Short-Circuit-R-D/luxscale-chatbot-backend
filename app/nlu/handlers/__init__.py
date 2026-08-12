from app.nlu.handlers.base import BaseHandler
from app.nlu.handlers.comparison_handler import ComparisonHandler
from app.nlu.handlers.fallback_handler import FallbackHandler
from app.nlu.handlers.greeting_handler import GreetingHandler
from app.nlu.handlers.historical_query_handler import HistoricalQueryHandler
from app.nlu.handlers.scope_handler import ScopeHandler
from app.nlu.handlers.standard_query_handler import StandardQueryHandler
from app.nlu.handlers.general_handler import GeneralHandler
from app.nlu.intents import Intent


def build_registry() -> dict[str, BaseHandler]:
    return {
        Intent.GREETING.value: GreetingHandler(),
        Intent.GENERAL.value: GeneralHandler(),
        Intent.STANDARD_QUERY.value: StandardQueryHandler(),
        Intent.HISTORICAL_QUERY.value: HistoricalQueryHandler(),
        Intent.COMPARISON.value: ComparisonHandler(),
        Intent.OUT_OF_SCOPE.value: ScopeHandler(),
        Intent.FALLBACK.value: FallbackHandler(),
    }