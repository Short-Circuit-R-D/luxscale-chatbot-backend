from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.nlu.schemas import IntentPrediction
from app.services.rag_service import RagService
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