import re

from langchain_groq import ChatGroq

from app.nlu.schemas import IntentPrediction

# GREETING_WORDS = {
#     "hi", "hello", "hey", "salam", "marhaba", "peace", "yo", "hola",
#     "good", "morning", "evening", "thank", "thanks", "thx", "bye",
# }

# SCOPE_PATTERNS = [
#     re.compile(r"en\s*12464\s*-\s*[2-9]", re.IGNORECASE),
#     re.compile(r"\biso\s*\d", re.IGNORECASE),
#     re.compile(r"\b(calculate|computation)\b", re.IGNORECASE),
#     re.compile(r"\b(number|count|how many)\s+(of\s+)?(luminaire|fixture)s?\b", re.IGNORECASE),
#     re.compile(r"\b(dwg|cad|photometric|ies)\b", re.IGNORECASE),
# ]

# COMPARISON_PATTERNS = [
#     re.compile(r"\bcompare\b", re.IGNORECASE),
#     re.compile(r"\bdifference\b", re.IGNORECASE),
#     re.compile(r"\bsimilarities?\b", re.IGNORECASE),
# ]

YEAR_PATTERN = re.compile(r"^(19\d{2}|20\d{2})$")

PROMPT = (
    "Classify the user message for a lighting-standards assistant.\n\n"
    "Intent contract:\n"
    "- greeting: casual hello/thanks-only messages.\n"
    "- standard_query: asking about lighting levels/requirements of the "
    "current standard edition (default when no year is mentioned). and if the standard is not specified assume its the current edition of EN 12464-1. DONT USE ANY OTHER STANDARD WITHOUT EXPLICITLY SPECIFIED. \n"
    "- general: asking about definitions or any inquiry about lighting definitions and comparison between definitions that can be answered without referring to citations or contexts from us.\n"
    "- historical_query: same, but the user explicitly mentions a specific "
    "version year (e.g. '2019 version').\n"
    "- comparison: asking to compare two editions/values.\n"
    "- out_of_scope: anything not about EN 12464-1 lighting levels "
    "(other standards, calculations, CAD, non-lighting).\n"
    "- fallback: only when nothing else fits.\n\n"
    "Return intent and extract entities: version_year (year only, e.g. 2019), "
    "standard_code (e.g. 'EN 12464-1'), ref_number (e.g. '6.1.1').\n\n"
    "Return a JSON object with the following structure:\n"
    "{\n"
    "  \"intent\": \"string\",\n"
    "  \"entities\": {\n"
    "    \"version_year\": \"string?\",\n"
    "    \"standard_code\": \"string?\",\n"
    "    \"ref_number\": \"string?\"\n"
    "  }\n"
    "}\n"
    "You can return null for any entity that is not present in the user message. \n\n"
    "You can accept any language in the user message, but always return the intent and entities in English."
)


class IntentPredictor:
    def __init__(self, chat: ChatGroq):
        self._llm = chat.with_structured_output(IntentPrediction)

    def predict(self, message: str) -> IntentPrediction:
        lowered = message.strip().lower()

        words = set(re.split(r"\s+", lowered))
        # if words & GREETING_WORDS and len(words) <= 6:
        #     return IntentPrediction(intent="greeting")

        # if any(p.search(message) for p in COMPARISON_PATTERNS):
        #     return IntentPrediction(intent="comparison")

        # if any(p.search(message) for p in SCOPE_PATTERNS):
        #     return IntentPrediction(intent="out_of_scope")

        try:
            prediction = self._llm.invoke(PROMPT + f"\n\nUser message: {message}")

            # print(prediction)
        except Exception as e:
            print("Error predicting intent, returning", e)
            return IntentPrediction(intent="standard_query")

        if prediction.intent == "historical_query":
            print("Historical query detected, validating year:", prediction.entities.version_year)
            year = (prediction.entities.version_year or "").strip()
            if not YEAR_PATTERN.fullmatch(year):
                prediction.entities.version_year = None
                prediction.intent = "standard_query"
        return prediction