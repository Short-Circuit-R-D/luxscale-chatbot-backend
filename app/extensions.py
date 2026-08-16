from pymongo import MongoClient

from app.config import Config
from app.nlu.handlers import build_registry
from app.nlu.intent_predictor import IntentPredictor
from app.orchestrator.orchestrator import Orchestrator
from app.repositories.mongodb.standards_repository import StandardsRepository
from app.repositories.qdrant.qdrant_repository import QdrantRepository
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService
from app.utils.qdrant import create_qdrant_client

qdrant_client = create_qdrant_client()


mongo_client = MongoClient(Config.MONGO_URI)
mongo_db = mongo_client[Config.MONGO_DB]

standards_repo = StandardsRepository(mongo_db)

embedding_service = EmbeddingService(model_name=Config.EMBEDDING_MODEL, device=Config.EMBEDDING_DEVICE)
embedding_service.load()  # warm model at process startup (not on first request)

qdrant_repo = QdrantRepository(
    client=qdrant_client,
    collection_name=Config.QDRANT_COLLECTION,
    embeddings_loader=lambda: embedding_service.lc_embeddings,
)

ingestion_service = IngestionService(
    standards_repo=standards_repo,
    qdrant_repo=qdrant_repo,
    embedding_service=embedding_service,
)

retrieval_service = RetrievalService(
    embedding_service=embedding_service,
    qdrant_repo=qdrant_repo,
    standards_repo=standards_repo,
    top_k=Config.TOP_K,
)

rag_service = RagService(model=Config.LLM_MODEL, api_key=Config.GROQ_API_KEY)

predictor = IntentPredictor(chat=rag_service.chat)
registry = build_registry()
orchestrator = Orchestrator(
    predictor=predictor,
    registry=registry,
    retrieval=retrieval_service,
    rag=rag_service,
)