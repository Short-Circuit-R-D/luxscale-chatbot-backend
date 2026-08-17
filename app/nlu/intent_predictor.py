import re

from langchain_groq import ChatGroq

from app.nlu.intent_catalog import classifier_prompt
from app.nlu.intents import Intent
from app.nlu.schemas import IntentPrediction

YEAR_PATTERN = re.compile(r"^(19\d{2}|20\d{2})$")


class IntentPredictor:
    def __init__(self, chat: ChatGroq):
        self._llm = chat.with_structured_output(
            IntentPrediction, method="json_mode"
        )
        self._prompt = classifier_prompt()

    def predict(self, message: str) -> IntentPrediction:
        try:
            prediction = self._llm.invoke(
                self._prompt + f"\n\nUser message: {message}"
            )
        except Exception as e:
            print("Error predicting intent, returning clarify:", e)
            return IntentPrediction(intent=Intent.CLARIFY)

        if prediction.intent == Intent.HISTORICAL_QUERY:
            year = (prediction.entities.version_year or "").strip()
            if not YEAR_PATTERN.fullmatch(year):
                prediction.entities.version_year = None
                prediction.intent = Intent.STANDARD_QUERY
        return prediction
