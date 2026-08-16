from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.nlu.intent_catalog import get_spec
from app.nlu.intents import Intent
from app.nlu.schemas import IntentPrediction
from app.services.rag_service import CustomChatPromptTemplate, RagService
from app.services.retrieval_service import RetrievalService


@dataclass
class OrchestratorResult:
    text: str
    intent: str
    citations: list[str] = field(default_factory=list)


class IntentContext:
    def __init__(self, session_id: str, query: str, prediction: IntentPrediction,
                 retrieval: RetrievalService, rag: RagService, cache, history_messages):
        self.session_id = session_id
        self.query = query
        self.prediction = prediction
        self.retrieval = retrieval
        self.rag = rag
        self.cache = cache
        self.history_messages = history_messages


class BaseHandler(ABC):
    @abstractmethod
    def handle(self, ctx: IntentContext) -> OrchestratorResult:
        raise NotImplementedError


def catalog_answer(
    ctx: IntentContext,
    intent: Intent,
    *,
    citation: str | None = None,
) -> OrchestratorResult:
    """Answer using catalog boundaries + intent_prompt.

    Pass citation=... for RAG intents (use empty string if retrieval empty).
    Pass citation=None for LLM intents.
    """
    spec = get_spec(intent)
    text = ctx.rag.answer(
        CustomChatPromptTemplate(
            question=ctx.query,
            citation=citation,
            history=ctx.history_messages,
            intent_prompt=spec.intent_prompt,
            boundaries=spec.boundaries,
        )
    )
    return OrchestratorResult(text=text, intent=intent.value)
