import re

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

from app.nlu.intent_catalog import classifier_prompt
from app.nlu.intents import Intent
from app.nlu.schemas import IntentPrediction

YEAR_PATTERN = re.compile(r"^(19\d{2}|20\d{2})$")

CONCEPT_HINT = re.compile(
    r"(?i)("
    r"\bin general\b|"
    r"\bnot according to\b|"
    r"\bconcept of\b|"
    r"\bwhat does\s+\w+\s+mean\b|"
    r"\bwhat\s+\w+\s+means\b|"
    r"\bjust want to know what\b|"
    r"\bdon'?t need requirements\b"
    r")"
)
DEFINITION_TERM = re.compile(
    r"(?i)\b(what\s+(is|are)|what's|explain|define|meaning of)\s+"
    r"(a |an |the )?(cri|c\.?r\.?i\.?|ra|ugr|cct|flicker|glare|"
    r"illuminance|uniformity|colour rendering|color rendering)\b"
)
REQUIREMENTS_HINT = re.compile(
    r"(?i)\b(office|corridor|classroom|hospital|edition|version|"
    r"table|ref\b|en\s*12464|\d{4}|requirements?|lux|levels?)\b"
)
SIMULATOR_HINT = re.compile(
    r"(?i)\b(simulator|open|show me|interactive|iframe)\b"
)


def _format_history(history: list[BaseMessage] | None, limit: int = 6) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-limit:]:
        role = "user" if getattr(msg, "type", None) == "human" else "assistant"
        content = str(getattr(msg, "content", "")).strip().replace("\n", " ")
        if len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def looks_like_concept_question(message: str) -> bool:
    text = message.strip()
    if SIMULATOR_HINT.search(text):
        return False
    if CONCEPT_HINT.search(text):
        return True
    return bool(DEFINITION_TERM.search(text) and not REQUIREMENTS_HINT.search(text))


class IntentPredictor:
    def __init__(self, chat: ChatGroq):
        self._llm = chat.with_structured_output(
            IntentPrediction, method="json_mode"
        )
        self._prompt = classifier_prompt()

    def predict(
        self,
        message: str,
        history: list[BaseMessage] | None = None,
    ) -> IntentPrediction:
        history_block = _format_history(history)
        prompt = self._prompt
        if history_block:
            prompt += f"\n\nRecent conversation:\n{history_block}"
        prompt += f"\n\nUser message: {message}\nRespond in JSON."

        try:
            prediction = self._llm.invoke(prompt)
        except Exception as e:
            print("Error predicting intent, returning fallback:", e)
            if looks_like_concept_question(message):
                return IntentPrediction(intent=Intent.GENERAL)
            return IntentPrediction(intent=Intent.FALLBACK)

        if looks_like_concept_question(message) and prediction.intent in (
            Intent.CLARIFY,
            Intent.STANDARD_QUERY,
            Intent.HISTORICAL_QUERY,
            Intent.FALLBACK,
        ):
            prediction.intent = Intent.GENERAL

        if prediction.intent == Intent.HISTORICAL_QUERY:
            year = (prediction.entities.version_year or "").strip()
            if not YEAR_PATTERN.fullmatch(year):
                prediction.entities.version_year = None
                prediction.intent = Intent.STANDARD_QUERY
        return prediction
