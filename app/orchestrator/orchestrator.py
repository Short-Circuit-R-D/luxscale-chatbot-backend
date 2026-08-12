from app.nlu.handlers.base import IntentContext, OrchestratorResult
from app.nlu.intent_predictor import IntentPredictor
from app.repositories.cache.session_cache_repository import session_cache_repo
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService


class Orchestrator:
    def __init__(self, predictor: IntentPredictor, registry: dict, retrieval: RetrievalService, rag: RagService):
        self.predictor = predictor
        self.registry = registry
        self.retrieval = retrieval
        self.rag = rag

    def run(self, session_id: str, user_message: str) -> OrchestratorResult:
        session_cache_repo.append_turn(session_id, "user", user_message)

        prediction = self.predictor.predict(user_message)
        print(f"Predicted intent: {prediction.intent}, entities: {prediction.entities}")
        handler = self.registry.get(prediction.intent, self.registry["fallback"])
        print(f"Using handler: {handler.__class__.__name__} for intent: {prediction.intent}")

        history = session_cache_repo.get_history_messages(session_id)[:-1]
        ctx = IntentContext(
            session_id=session_id,
            query=user_message,
            prediction=prediction,
            retrieval=self.retrieval,
            rag=self.rag,
            cache=session_cache_repo,
            history_messages=history,
        )
        result = handler.handle(ctx)

        session_cache_repo.append_turn(session_id, "assistant", result.text)
        return result