from app.nlu.handlers.base import IntentContext, OrchestratorResult
from app.nlu.intent_predictor import IntentPredictor
from app.repositories.cache.session_cache_repository import session_cache_repo
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService
from app.services.simulator_service import SimulatorService


class Orchestrator:
    def __init__(
        self,
        predictor: IntentPredictor,
        registry: dict,
        retrieval: RetrievalService,
        rag: RagService,
        simulator: SimulatorService,
    ):
        self.predictor = predictor
        self.registry = registry
        self.retrieval = retrieval
        self.rag = rag
        self.simulator = simulator

    def run(self, session_id: str, user_message: str) -> OrchestratorResult:
        session_cache_repo.append_turn(session_id, "user", user_message)

        prediction = self.predictor.predict(user_message)
        print(f"Predicted intent: {prediction.intent}, entities: {prediction.entities}")
        intent_key = (
            prediction.intent.value
            if hasattr(prediction.intent, "value")
            else prediction.intent
        )
        handler = self.registry.get(intent_key, self.registry["fallback"])
        print(f"Using handler: {handler.__class__.__name__} for intent: {intent_key}")

        history = session_cache_repo.get_history_messages(session_id)[:-1]
        ctx = IntentContext(
            session_id=session_id,
            query=user_message,
            prediction=prediction,
            retrieval=self.retrieval,
            rag=self.rag,
            cache=session_cache_repo,
            history_messages=history,
            simulator=self.simulator,
        )
        result = handler.handle(ctx)

        session_cache_repo.append_turn(session_id, "assistant", result.text)
        return result