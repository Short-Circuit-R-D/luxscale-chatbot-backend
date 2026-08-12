from enum import Enum


class Intent(str, Enum):
    GREETING = "greeting"
    GENERAL = "general"
    STANDARD_QUERY = "standard_query"
    HISTORICAL_QUERY = "historical_query"
    COMPARISON = "comparison"
    OUT_OF_SCOPE = "out_of_scope"
    FALLBACK = "fallback"