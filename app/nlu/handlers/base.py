from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from app.nlu.intent_catalog import get_spec
from app.nlu.intents import Intent
from app.nlu.schemas import IntentPrediction
from app.services.rag_service import CustomChatPromptTemplate, RagService
from app.services.retrieval_service import RetrievalService
from app.services.simulator_service import SimulatorService


@dataclass
class OrchestratorResult:
    text: str
    intent: str
    citations: list[str] = field(default_factory=list)
    simulator: Optional[dict] = None


class IntentContext:
    def __init__(
        self,
        session_id: str,
        query: str,
        prediction: IntentPrediction,
        retrieval: RetrievalService,
        rag: RagService,
        cache,
        history_messages,
        simulator: SimulatorService,
    ):
        self.session_id = session_id
        self.query = query
        self.prediction = prediction
        self.retrieval = retrieval
        self.rag = rag
        self.cache = cache
        self.history_messages = history_messages
        self.simulator = simulator


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


def attach_concept_simulator(ctx: IntentContext, result: OrchestratorResult) -> OrchestratorResult:
    payload = ctx.simulator.resolve_for_concept(ctx.query)
    if payload is not None:
        result.simulator = payload.as_dict()
    return result
