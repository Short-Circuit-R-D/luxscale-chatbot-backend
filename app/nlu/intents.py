from enum import Enum


class IntentCategory(str, Enum):
    RAG = "rag"
    LLM = "llm"
    API = "api"


class Intent(str, Enum):
    # RAG — require retrieval
    STANDARD_QUERY = "standard_query"
    HISTORICAL_QUERY = "historical_query"
    COMPARISON = "comparison"
    REF_LOOKUP = "ref_lookup"

    # LLM — no retrieval
    GREETING = "greeting"
    GENERAL = "general"
    LIGHTING_GUIDANCE = "lighting_guidance"
    CLARIFY = "clarify"
    OUT_OF_SCOPE = "out_of_scope"
    FALLBACK = "fallback"

    # API — catalog/tools, no clause retrieval
    GET_SIMULATOR = "get_simulator"
