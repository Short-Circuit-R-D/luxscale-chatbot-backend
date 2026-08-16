import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "") or None

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_URL_LOCALHOST = os.getenv("QDRANT_URL_LOCALHOST", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "standards")

    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB = os.getenv("MONGO_DB", "luxscale_chatbot")
    MONGO_STANDARDS_COLLECTION = os.getenv("MONGO_STANDARDS_COLLECTION", "standards_clauses")

    TOP_K = int(os.getenv("TOP_K", "5"))