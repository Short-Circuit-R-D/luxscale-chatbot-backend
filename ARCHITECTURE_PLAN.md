# Luxscale Chatbot — Architecture Plan (v2)

From-scratch build plan for the luxscale-chatbot repo. Supersedes the previous
ingestion+chat plan and extends `rag-chatbot-architecture.md`.

Exactly **5 endpoints**:
1. `POST /api/v1/ingest` — ingest a standard clause JSON (idempotent, versioned; embeds + indexes)
2. `POST /api/v1/documents` — bulk raw-store of any Mongo-shaped JSON (idempotent by `_id`; no embedding)
3. `POST /api/v1/chat/message` — send a message, get AI response (`intent` included)
4. `GET /api/v1/chat/{session_id}` — fetch a session's history from cache
5. `GET /api/v1/health` — service connectivity status

---

## 1. Stack & LangChain Seam Map

| Layer | Implementation |
|---|---|
| WSGI framework | Flask (single-process dev; multi-worker flagged as a risk) |
| Validation | pydantic v2 |
| LLM | Groq (`GROQ_API_KEY`), model configured via `LLM_MODEL` (default `llama-3.3-70b-versatile`) |
| Embeddings | `BAAI/bge-m3` (1024-dim) via HuggingFace, loaded lazily on first use |
| Vector store | Qdrant (docker), collection `standards`, COSINE |
| Standards store | MongoDB, collection `standards_clauses` |
| Session memory | In-process thread-safe dict (Redis later), TTL 24h |
| **LangChain** | **Only 4 seams** (below); all custom logic (idempotency, versioning, NLU routing, ingestion) stays hand-rolled |

**LangChain seam boundaries (the whole LC story in one table):**

| Seam | LC component | Custom logic that stays ours |
|---|---|---|
| LLM chat | `ChatGroq` + `ChatPromptTemplate` + `StrOutputParser` | citations plumbing, prompt contract |
| Chat history | `BaseChatMessageHistory` implemented by the session cache | TTL, expiry, `session_exists`, keying, REST history endpoint |
| Embeddings | `HuggingFaceEmbeddings(model_name="BAAI/bge-m3")` behind `EmbeddingService` facade | lazy load, device handling, dims check, shared instance |
| Vector I/O | `QdrantVectorStore` (langchain-qdrant) inside `QdrantRepository` | deterministic uuid5 ids, payload schema, `is_latest` patch via `store.client.set_payload` escape hatch |
| | | idempotency/versioning, Mongo repo + join, ingestion service, NLU intents/handlers, orchestrator routing |

---

## 2. What Exactly Gets Embedded

**Embed `searchable_text` only.** Nothing else.

Reasoning:
- It's a purpose-built natural-language synthesis of `standard_code`, `category_title`, `activity`, every lighting parameter, and `specific_requirements` — exactly the shape a user's query will semantically match against.
- Raw structured fields (`parameters.em_r_lx: 100`) don't embed meaningfully; they belong in Qdrant **payload** and Mongo for exact filtering/display.
- Embedding the whole raw JSON (keys like `_id`, `qdrant_point_id`) pollutes the vector with structural noise.

**Rule going forward for any new document type:** if a `searchable_text`-style precomputed field exists, embed only that. If it doesn't exist, build one deterministically from the meaningful fields before embedding — never embed the raw payload as-is.

**Payload fields kept in Qdrant** (filtering + citing, not embedded):
`mongo_id`, `standard_code`, `version_year`, `is_latest`, `category_table_number`, `ref_number`, `activity`, `page`, `content_hash`.

---

## 3. Idempotency Strategy

Two layers:

**a) Document-level idempotency (Mongo `_id`)**
Client-provided `_id` (e.g. `"en12464_1_v2019_6_1_1"`) is the natural key. Re-ingesting the same `_id` is an upsert (`replace_one(..., upsert=True)`), never a duplicate.

**b) Content-change idempotency (`content_hash`)**
`content_hash = sha256(searchable_text)`, stored on the Mongo doc:

| Mongo lookup by `_id` | `content_hash` match? | Action |
|---|---|---|
| Not found | — | Insert Mongo doc, embed, upsert Qdrant point → `"inserted"` |
| Found | Same | Skip embedding + skip Qdrant write entirely → `"skipped_unchanged"` |
| Found | Different | Update Mongo doc, re-embed, overwrite Qdrant point (same point ID) → `"updated"` |

Re-running the same ingestion job does **zero** redundant embedding calls or Qdrant writes for unchanged clauses.

**c) Qdrant point ID — deterministic, not client-supplied**
`qdrant_point_id` from the source JSON is unreliable (numeric IDs from ref numbers like `6.1.1` collide across standards). Derive it deterministically:

```python
import uuid

def derive_point_id(mongo_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, mongo_id))
```

Same `_id` in → same point ID out → `upsert` overwrites in place. The source hint is kept in Mongo **only** for traceability.

---

## 4. Versioning Strategy

A "standard" (`standard_code` + `ref_number`) can have multiple `version_year`s, each a distinct Mongo doc. Only one has `is_latest: true` at a time.

**On every successful insert/update of a doc that is effectively latest:**
1. Write/update the Mongo doc as normal.
2. Query Mongo for siblings: same `standard_metadata.standard_code` + `hierarchy.ref_number`, `is_latest: true`, `version_year != this_version_year`.
3. For each sibling: set `is_latest: false` in Mongo, and patch **only the Qdrant payload** via `set_payload` (`is_latest: false`) — no re-embedding.
4. The incoming doc's own `is_latest` flag is trusted as *intent*, but the supersession step is what enforces only-one-latest.

**Old-version guard (never flip `is_latest` backwards):**
If the incoming `version_year` is *older* than the current latest sibling's `version_year`, it is ingested as historical:
- `is_latest` is **forced to `false`** on the Mongo doc and Qdrant payload, regardless of what the source JSON claims.
- The supersession pass does not run.
- It only ever demotes; it never promotes an older version.

**At retrieval time:** default Qdrant filter is `is_latest = true`. A query that supplies a `version_year` entity (from the NLU predictor) switches the filter to that specific year, dropping the latest constraint.

---

## 5. Mongo Document Schema (as stored)

Same shape as the source JSON, plus metadata fields:

```json
{
  "_id": "en12464_1_v2019_6_1_1",
  "qdrant_point_id": 6112019,
  "content_hash": "a1b2c3...",
  "created_at": "2026-08-09T12:00:00Z",
  "updated_at": "2026-08-09T12:00:00Z",
  "standard_metadata": { "...": "..." },
  "hierarchy": { "...": "..." },
  "activity": "Corridors and circulation areas",
  "parameters": { "...": "..." },
  "specific_requirements": "...",
  "searchable_text": "..."
}
```

`qdrant_point_id` is kept as provided by the source (traceability only — never used as the actual Qdrant point ID).

---

## 6. Qdrant Point Schema

```json
{
  "id": "<uuid5 derived from mongo _id>",
  "vector": [ 1024 floats of bge-m3 ],
  "payload": {
    "mongo_id": "en12464_1_2019_6_1_1",
    "standard_code": "prEN 12464-1",
    "version": "2019",
    "is_latest": true,
    "reference_number": "6.1.1",
    "activity": "Corridors and circulation areas",
    "page": 26,
    "content_hash": "a1b2c3..."
  }
}
```

---

## 7. File Structure (complete, built from scratch)

```
luxscale-chatbot/
├── .env.example               # template for all config
├── .gitignore                 # .env, .venv/, __pycache__/
├── docker-compose.yml         # qdrant + mongo (+ optional mongo-express)
├── requirements.txt           # see §8
├── main.py                    # create_app() factory, blueprint registration, app.run()
├── scripts/
│   ├── create_qdrant_collection.py   # 1024-dim COSINE, idempotent, payload indexes
│   ├── seed_sample_data.py           # sample EN 12464-1 clauses (2 editions)
│   └── verify.py                     # end-to-end idempotency/versioning/chat checks
└── app/
    ├── __init__.py
    ├── config.py             # env-driven settings (class Config)
    ├── extensions.py         # builds all singletons (import-time)
    ├── api/
    │   ├── routes/
    │   │   ├── __init__.py   # registers blueprints
    │   │   ├── health_routes.py
    │   │   ├── ingest_routes.py      # POST /ingest, POST /documents (bulk raw store)
    │   │   └── chat_routes.py
    │   └── schemas/
    │       ├── ingest_schema.py
    │       └── chat_schema.py
    ├── repositories/
    │   ├── qdrant/qdrant_repository.py
    │   ├── mongodb/standards_repository.py
    │   └── cache/__init__.py + session_cache_repository.py
    ├── services/
    │   ├── embedding_service.py
    │   ├── ingestion_service.py
    │   ├── retrieval_service.py
    │   └── rag_service.py
    ├── nlu/
    │   ├── __init__.py
    │   ├── intents.py          # Intent enum + taxonomy
    │   ├── intent_predictor.py # message → (intent, entities)
    │   └── handlers/
    │       ├── __init__.py     # HandlerRegistry
    │       ├── base.py
    │       ├── greeting_handler.py
    │       ├── standard_query_handler.py
    │       ├── historical_query_handler.py
    │       ├── comparison_handler.py     # deferred → fallback reply
    │       ├── scope_handler.py
    │       └── fallback_handler.py
    ├── orchestrator/orchestrator.py
    └── utils/
        ├── hashing.py
        └── ids.py
```

---

## 8. requirements.txt & .env.example

```txt
flask>=3.0
python-dotenv>=1.0
pydantic>=2.7
qdrant-client>=1.12
pymongo>=4.8
sentence-transformers>=3.0
groq>=0.11
langchain>=0.3
langchain-groq>=0.2
langchain-huggingface>=0.1
langchain-qdrant>=0.1
```

`.env.example` keys (no secrets): `GROQ_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL=BAAI/bge-m3`, `EMBEDDING_DEVICE` (blank = CPU), `QDRANT_URL=http://localhost:6333`, `QDRANT_COLLECTION=standards`, `MONGO_URI=mongodb://localhost:27017`, `MONGO_DB=luxscale_chatbot`, `MONGO_STANDARDS_COLLECTION=standards_clauses`, `TOP_K=5`.

---

## 9. Pydantic Schemas (v2)

### `api/schemas/ingest_schema.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class StandardMetadata(BaseModel):
    standard_code: str
    version_year: str
    is_latest: bool

class Hierarchy(BaseModel):
    category_table_number: str
    category_title: str
    ref_number: str
    page: int

class Parameters(BaseModel):
    em_r_lx: Optional[float] = None
    em_u_lx: Optional[float] = None
    uo: Optional[float] = None
    ra: Optional[float] = None
    ugr_rugl: Optional[float] = None
    ez_lx: Optional[float] = None
    em_wall_lx: Optional[float] = None
    em_ceiling_lx: Optional[float] = None

class IngestDocument(BaseModel):
    id: str
    qdrant_point_id: Optional[int] = None      # traceability only
    standard_metadata: StandardMetadata
    hierarchy: Hierarchy
    activity: str
    parameters: Parameters
    specific_requirements: str
    searchable_text: str

class IngestRequest(BaseModel):
    documents: List[IngestDocument]

class IngestResultItem(BaseModel):
    mongo_id: str
    status: str        # "inserted" | "updated" | "skipped_unchanged" | "failed"
    error: Optional[str] = None

class IngestResponse(BaseModel):
    results: List[IngestResultItem]
```

Pydantic v2 note: the route normalizes `_id` → `id` before validation (`IngestDocument` has no alias — normalization happens in the route, keeping the model simple).

### `api/schemas/chat_schema.py`

```python
from pydantic import BaseModel
from typing import Optional, List

class ChatMessageRequest(BaseModel):
    session_id: Optional[str] = None   # generated server-side if omitted
    message: str

class ChatMessageResponse(BaseModel):
    session_id: str
    response: str
    intent: str

class ChatTurn(BaseModel):
    role: str                          # "user" | "assistant"
    content: str
    timestamp: str

class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatTurn]
```

### `nlu/schemas.py` (intent contract, also used for structured LLM output)

```python
from pydantic import BaseModel
from typing import Optional, Literal

class IntentEntities(BaseModel):
    standard_code: Optional[str] = None
    version_year: Optional[str] = None
    ref_number: Optional[str] = None

class IntentPrediction(BaseModel):
    intent: Literal[
        "greeting", "standard_query", "historical_query",
        "comparison", "out_of_scope", "fallback",
    ]
    entities: IntentEntities = IntentEntities()
```

### `api/schemas/store_schema.py` (raw bulk store)

```python
from pydantic import BaseModel
from typing import Any, List

class StoredDocument(BaseModel):
    _id: str
    # everything else is accepted as-is; coercion/typing left to Mongo
    model_config = {"extra": "allow"}

    @classmethod
    def coerce(cls, raw: dict) -> "StoredDocument":
        return cls(**raw)   # raises if "_id" missing/empty
```

---

## 10. Utils

### `utils/hashing.py`
```python
import hashlib
import json

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def doc_json_hash(doc: dict) -> str:
    """Canonical full-document hash (sorted keys) — used by the raw bulk
    store endpoint to detect unchanged re-submissions."""
    canonical = json.dumps(doc, sort_keys=True, ensure_ascii=False,
                           default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

### `utils/ids.py`
```python
import uuid

def derive_point_id(mongo_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, mongo_id))
```

### `utils/versioning.py` (helper for the old-version guard)
```python
def parse_year(version_year: str) -> int | None:
    digits = "".join(c for c in version_year if c.isdigit())
    return int(digits[:4]) if digits else None
```

---

## 11. Repositories

### `repositories/qdrant/qdrant_repository.py` (LC-based)

```python
from qdrant_client.models import PointStruct
from langchain_qdrant import QdrantVectorStore

class QdrantRepository:
    def __init__(self, client, collection_name: str, embeddings_loader):
        self.client = client
        self.collection_name = collection_name
        self._embeddings_loader = embeddings_loader
        self._store = None

    def _store_instance(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore(
                client=self.client,                 # langchain-qdrant >=1.x __init__ takes the raw client
                collection_name=self.collection_name,
                embedding=self._embeddings_loader(),
            )
        return self._store

    def upsert_point(self, point_id: str, vector: list[float], payload: dict):
        # raw client write: keeps uuid5 idempotent upsert, no re-embedding
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def set_payload(self, point_id: str, payload: dict):
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload,
            points=[point_id],
        )

    def point_exists(self, point_id: str) -> bool:
        return bool(self.client.retrieve(
            collection_name=self.collection_name, ids=[point_id]
        ))

    def search(self, embedding: list[float], qdrant_filter, top_k: int):
        return self._store_instance().similarity_search_with_score_by_vector(
            embedding=embedding, k=top_k, filter=qdrant_filter
        )   # → list[(Document, score)]; Document.metadata == our payload
```

Notes (langchain-qdrant 1.x reality):
- In 1.x, `QdrantVectorStore.__init__` takes the raw `QdrantClient` (shared instance); `add_embeddings` was removed, so **writes/patches go through the raw client** (`upsert_point`, `set_payload`, `point_exists`) — preserving deterministic uuid5 ids, idempotent upserts, and never re-embedding text. The LC store is used for `search` only.
- `similarity_search_with_score_by_vector` accepts a raw `embedding` (no re-embed) — same vector used for ingestion and query.

### `repositories/mongodb/standards_repository.py`

```python
class StandardsRepository:
    def __init__(self, db):
        self.collection = db["standards_clauses"]

    def find_by_id(self, mongo_id: str):
        return self.collection.find_one({"_id": mongo_id})

    def upsert(self, doc: dict):
        self.collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def find_siblings_for_versioning(self, standard_code: str, ref_number: str, exclude_id: str):
        return list(self.collection.find({
            "standard_metadata.standard_code": standard_code,
            "hierarchy.ref_number": ref_number,
            "_id": {"$ne": exclude_id},
        }))

    def latest_version_year(self, standard_code: str, ref_number: str) -> int | None:
        doc = self.collection.find_one(
            {  "standard_metadata.standard_code": standard_code,
                "hierarchy.ref_number": ref_number,
                "standard_metadata.is_latest": True},
            {"standard_metadata.version_year": 1},
        )
        return parse_year(doc["standard_metadata"]["version_year"]) if doc else None

    def set_is_latest(self, mongo_id: str, value: bool):
        self.collection.update_one(
            {"_id": mongo_id}, {"$set": {"standard_metadata.is_latest": value}}
        )
```

### `repositories/cache/session_cache_repository.py` — LC-driven

Implements LangChain's `BaseChatMessageHistory` and keeps the REST-facing methods. Same store for both.

```python
import threading
import uuid
from datetime import datetime, timezone, timedelta
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

SESSION_TTL = timedelta(hours=24)

class InMemorySessionCacheRepository(BaseChatMessageHistory):
    """Thread-safe in-process store; implements LC's BaseChatMessageHistory
    while keeping TTL/expiry/session_exists & REST serialization. Swapping to
    Redis/Mongo later = one-file change.
    """

    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()

    # --- custom tier (REST + TTL) ---
    def _key_valid(self, session_id: str) -> bool:  # as before, lazy expiry
        ...

    def create_session_id(self) -> str:
        return str(uuid.uuid4())

    def get_history(self, session_id: str) -> list[dict]:
        # returns raw ["role", "content", "timestamp"] turns for the API
        ...

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return self._key_valid(session_id)

    def append_turn(self, session_id: str, role: str, content: str):
        # appends {"role", "content", "timestamp"} + refreshes expires_at
        ...

    # --- LC interface (BaseChatMessageHistory) ---
    @property
    def messages(self) -> list:
        """LS message objects — plays into RunnableWithMessageHistory and
        the RAG prompt."""
        return [...]

    def add_message(self, message):
        self.append_turn(self.session_id, _lc_role(message), message.content)

    def clear(self):
        ...


session_cache_repo = InMemorySessionCacheRepository()  # singleton (also re-exported from extensions.py)
```

Note: the LC-facing instance is per-session; the cache presents a small adapter
(`SessionChatHistory(session_id)` bound to the singleton) that satisfies
`BaseChatMessageHistory` and is passed to `RunnableWithMessageHistory` via
`configurable_session_id`.

---

## 12. Services

### `services/embedding_service.py` — facade over HuggingFaceEmbeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingService:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device or "cpu"
        self._lc = None

    def _load(self):
        if self._lc is None:
            self._lc = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs={"device": self.device})
        return self._lc

    @property
    def lc_embeddings(self):       # for QdrantRepository / QVectorStore construction
        return self._load()

    def embed(self, text: str) -> list[float]:
        return self._load().embed_query(text)

    @property
    def loaded(self) -> bool:
        return self._lc is not None


embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
```

Lazy-load on first embed — `/health` and startup stay fast; model (~2.3 GB) downloads once.

### `services/ingestion_service.py` — idempotency + versioning + old-version guard

```python
from datetime import datetime, timezone
from app.utils.hashing import content_hash
from app.utils.ids import derive_point_id
from app.utils.version import parse_year

class IngestionService:
    def __init__(self, standards_repo, qdrant_repo, embedding_service):
        self.standards_repo = standards_repo
        self.qdrant_repo = qdrant_repo
        self.embedder = embedding_service

    def ingest_document(self, doc: dict) -> dict:
        mongo_id = doc["id"]
        new_hash = content_hash(doc["searchable_text"])
        existing = self.standards_repo.find_by_id(mongo_id)

        if existing and existing.get("content_hash") == new_hash:
            return {"mongo_id": mongo_id, "status": "skipped_unchanged"}

        # old-version guard: never let an older edition claim is_latest
        is_latest = self._declared_latest(doc)
        doc["standard_metadata"] = {**doc["standard_metadata"], "is_latest": is_latest}

        now = datetime.now(timezone.utc).isoformat()
        mongo_doc = {
            **doc,
            "_id": mongo_id,
            "content_hash": new_hash,
            "updated_at": now,
            "created_at": existing["created_at"] if existing else now,
        }
        self.standards_repo.upsert(mongo_doc)

        vector = self.embedder.embed(doc["searchable_text"])
        point_id = derive_point_id(mongo_id)
        self.qdrant_repo.upsert_point(point_id, vector, self._payload(doc, mongo_id, new_hash))

        if is_latest:
            self._supersede_older_versions(doc)

        status = "updated" if existing else "inserted"
        return {"mongo_id": mongo_id, "status": status}

    def _declared_latest(self, doc: dict) -> bool:
        incoming_year = parse_year(doc["standard_metadata"]["version_year"])
        latest_year = self.standards_repo.latest_version_year(
            doc["standard_metadata"]["standard_code"], doc["hierarchy"]["ref_number"]
        )
        if incoming_year < latest_year:
            return False                    # source claims latest, but a newer exists
        return doc["standard_metadata"]["is_latest"]

    def _supersede_older_versions(self, doc: dict):
        for sibling in self.standards_repo.find_siblings_for_versioning(
            doc["standard_metadata"]["standard_code"],
            doc["hierarchy"]["ref_number"],
            doc["id"],
        ):
            if sibling.get("standard_metadata", {}).get("is_latest"):
                self.standards_repo.set_is_latest(sibling["_id"], False)
                self.qdrant_repo.set_payload(
                    derive_point_id(sibling["_id"]), {"is_latest": False}
                )

    def _payload(self, doc, mongo_id, new_hash):
        return {
            "mongo_id": mongo_id,
            "standard_code": doc["standard_metadata"]["standard_code"],
            "version_year": doc["standard_metadata"]["version_year"],
            "is_latest": doc["standard_metadata"]["is_latest"],
            "category_table_number": doc["hierarchy"]["category_table_number"],
            "ref_number": doc["hierarchy"]["ref_number"],
            "activity": doc["activity"],
            "page": doc["hierarchy"]["page"],
            "content_hash": new_hash,
        }

    def store_document(self, doc: dict) -> dict:
        """Raw bulk store — Mongo only, no embedding, no versioning, no
        Qdrant. Idempotent by `_id`: same id + identical JSON never writes
        twice. Intended for arbitrary Mongo-shaped payloads (metadata,
        documents outside the standards pipeline)."""
        mongo_id = doc.get("_id")
        if not mongo_id:
            raise ValueError("_id is required")

        new_hash = doc_json_hash(doc)
        existing = self.standards_repo.find_by_id(mongo_id)
        if existing:
            if existing.get("content_hash") == new_hash:
                return {"mongo_id": mongo_id, "status": "skipped_unchanged"}
            now = datetime.now(timezone.utc).isoformat()
            self.standards_repo.upsert({
                **doc,
                "content_hash": new_hash,
                "updated_at": now,
                "created_at": existing["created_at"],
            })
            return {"mongo_id": mongo_id, "status": "updated"}

        now = datetime.now(timezone.utc).isoformat()
        self.standards_repo.upsert({
            **doc,
            "content_hash": new_hash,
            "created_at": now,
            "updated_at": now,
        })
        return {"mongo_id": mongo_id, "status": "inserted"}
```

### `services/retrieval_service.py`

```python
class RetrievalService:
    def __init__(self, embedding_service, qdrant_repo, standards_repo, top_k):
        ...

    def search(self, query: str, version_year: str | None = None,
               standard_code: str | None = None, ref_number: str | None = None,
               top_k: int | None = None):
        embedding = self.embedder.embed(query)
        qfilter = self._build_filter(version_year, standard_code, ref_number)
        hits = self.qdrant_repo.search(embedding, qfilter, top_k or self._top_k)
        # join to Mongo for searchable_text (payload is lean by design)
        mongo_docs = self.standards_repo.find_many([h.metadata["mongo_id"] for h in hits])
        return [RetrievedClause(doc=doc, score=score, ref=doc["hierarchy"]["ref_number"]) for ...]

    def _build_filter(self, version_year, standard_code, ref_number):
        # default: is_latest = MatchValue(True)
        # version_year present → "version_year" == version_year (drop is_latest)
        # standard_code / ref_number → AND conditions on payload
        return qdrant_filter.Filter(must=[...FieldCondition(...)...])
```

Default `is_latest=True` at retrieval → current edition for RAG unless a `version_year` entity arrives from the NLU layer.

### `services/rag_service.py` — LC chain

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = (
    "You are a lighting-standards assistant. Answer ONLY from the provided "
    "standard clauses. Cite the standard, ref number, table and page for every "
    "claim. If the clauses don't answer the question, say so clearly."
)

class RagService:
    def __init__(self, llm_model, api_key, top_k_citations=5):
        self._chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("system", "Context:\n{citations}"),
                ("human", "{question}"),
            ])
            | ChatGroq(model=llm_model, api_key=api_key)
            | StrOutputParser()
        )

    def answer(self, question: str, citations: str, messages=None) -> str:
        return self._chain.invoke({"question": question, "citations": citations})
```

Groq is tuned for JSON via `.with_structured_output()`; the LLM boundary is one typed
call everywhere (predictor + RAG).

---

## 13. NLU: Intents, Predictor, Handlers

### `nlu/intents.py` — taxonomy (v1)

| intent | trigger | handler behavior | Qdrant filter |
|---|---|---|---|
| `greeting` | hi / hello / faq cash phrases | canned reply, no RAG | — |
| `standard_query` (default) | anything else about lighting parameters | RAG, current edition | `is_latest=true` |
| `historical_query` | entity `version_year` present ("EN 12464-1 room, corridor levels") | RAG, that edition | `version_year=<year>` |
| `comparison` | "compare/difference 2019 and 2021" | **deferred** → canned "not supported yet; ask about one edition" | fallback |
| `out_of_scope` | other standards, "calculate luminaire count", non-lighting | canned scope reply | — |
| `fallback` | nothing matched / LLM refuses | polite generic reply | — |

### `nlu/intent_predictor.py` — LLM pre-pass (choice B)

```python
class IntentPredictor:
    def __init__(self, chat: ChatGroq):
        self._fast_rules = FastRouteRules()          # greetings + out_of_scope regexes
        self._llm = chat.with_structured_output(IntentPrediction)

    def predict(self, message: str) -> IntentPrediction:
        hit = self._fast_rules.apply(message)       # zero-cost short-circuit
        if hit:
            return hit
        prediction = self._llm.invoke(message)
        # guard: strip out-of-vocabulary intents to FALLBACK
        ...
```

- Fast rules short-circuit `greeting` / `out_of_scope` — no LLM call for them.
- The LLM call returns a typed `IntentPrediction` via `with_structured_output` (JSON schema coercion, no manual JSON parsing).
- Invalid year / unknown intent in LLM output → coerced to `fallback` / nulled `version_year`.

### `nlu/handlers/`

```python
class IntentContext:
    session_id: str
    query: str
    history: list[dict]            # raw turns from cache
    prediction: IntentPrediction
    retrieval: RetrievalService
    rag: RagService
    session_cache: SessionCacheRepository

class BaseHandler(ABC):
    def handle(self, ctx: IntentContext) -> OrchestratorResult: ...

class GreetingHandler(BaseHandler): ...              # canned text, no RAG
class StandardQueryHandler(BaseHandler): ...          # retrieval(top_k) + rag.answer()
class HistoricalQueryHandler(BaseHandler):           # retrieval(version_year=ctx.prediction.entities.version_year)
class ScopeHandler(BaseHandler):                     # canned scope text
class FallbackHandler(BaseHandler):                  # generic second-pass prompt (no retrieval)

class ComparisonHandler(BaseHandler):
    def handle(self, ctx):       # DEFERRED — routes to FallbackHandler's canned reply
        return canned("Comparing editions isn't supported yet. Ask about one "
                      "specific edition (e.g. 'EN 12464-1 room 2019').")

Registry = {intent: handler_cls}   # dict built at import; orchestrator looks up by slug
```

Every handler returns an `OrchestratorResult(text, intent, citations)`.

---

## 14. Orchestrator

```python
@dataclass
class OrchestratorResult:
    text: str
    intent: str
    citations: list[str] = field(default_factory=list)

class Orchestrator:
    def __init__(self, cache, predictor, registry, retrieval, rag):
        ...

    def run(self, session_id: str, user_message: str) -> OrchestratorResult:
        self.cache.append_turn(session_id, "user", user_message)

        prediction = self.predictor.predict(user_message)
        ctx = IntentContext(
            session_id=session_id,
            history=self.cache.get_history(session_id),
            prediction=prediction,
            retrieval=self.retrieval,
            rag=self.rag,
            session_cache=self.cache,
        )
        result = self.registry[prediction.intent].handle(ctx)

        self.cache.append_turn(session_id, "assistant", result.text)
        return result
```

Pure routing — no business logic in the orchestrator. RunnableWithMessageHistory is used at the handler level when the RAG chain needs prior turns + current message as `messages` (LC's 3 roles: `SystemMessage`, `HumanMessage` history slice, final `HumanMessage` question).

---

## 15. API Routes

### `api/routes/ingest_routes.py`

```python
ingest_bp = Blueprint("ingest", __name__, url_prefix="/api/v1")

@ingest_bp.route("/ingest", methods=["POST"])
def ingest():
    body = request.get_json(force=True)
    docs = body["documents"] if "documents" in body else [body]   # single obj OK

    results = []
    for raw_doc in docs:
        try:
            normalized = {"id": raw_doc.pop("_id"), **raw_doc}   # "_id" -> "id"
            result = ingestion_service.ingest_document(normalized)
            results.append(IngestResultItem(**result))
        except Exception as e:
            results.append(IngestResultItem(
                mongo_id=raw_doc.get("_id", "unknown"), status="failed", error=str(e),
            ))
    return jsonify(IngestResponse(results=results).model_dump()), 200

@ingest_bp.route("/documents", methods=["POST"])
def store_documents():
    """Bulk raw store — accepts an array of Mongo-shaped JSON objects OR a
    single object. Idempotent by `_id`; identical re-submissions are
    `skipped_unchanged` (no writes). No embedding, no Qdrant."""
    body = request.get_json(force=True)
    docs = body["documents"] if isinstance(body, dict) and "documents" in body else (body if isinstance(body, list) else [body])

    results = []
    for raw_doc in docs:
        try:
            StoredDocument.coerce(raw_doc)               # validates "_id" present
            result = ingestion_service.store_document(raw_doc)
            results.append(IngestResultItem(**result))
        except Exception as e:
            results.append(IngestResultItem(
                mongo_id=raw_doc.get("_id", "unknown"), status="failed", error=str(e),
            ))
    return jsonify(IngestResponse(results=results).model_dump()), 200
```

### `api/routes/chat_routes.py`

```python
chat_bp = Blueprint("chat", __name__, url_prefix="/api/v1/chat")

@chat_bp.route("/message", methods=["POST"])
def send_message():
    body = ChatMessageRequest(**request.get_json(force=True))
    if not body.message.strip():
        return jsonify({"error": "message must not be empty"}), 400

    session_id = body.session_id or session_cache_repo.create_session_id()
    result = orchestrator.run(session_id=session_id, user_message=body.message)

    return jsonify(ChatMessageResponse(
        session_id=session_id, response=result.text, intent=result.intent,
    ).model_dump()), 200

@chat_bp.route("/<session_id>", methods=["GET"])
def get_session(session_id: str):
    if not session_cache_repo.session_exists(session_id):
        return jsonify({"error": "session not found"}), 404
    return jsonify({
        "session_id": session_id,
        "messages": session_cache_repo.get_history(session_id),
    }), 200
```

### `api/routes/health_routes.py`

```python
health_bp = Blueprint("health", __name__, url_prefix="/api/v1")

@health_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "qdrant": qdrant_ping(),
        "mongo": mongo_ping(),
        "model_loaded": embedding_service.loaded,
    }), 200
```

---

## 16. Qdrant Collection Setup (bge-m3 = 1024 dims)

```python
# scripts/create_qdrant_collection.py — idempotent
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

if collection exists and not --force: print("exists"); exit(0)

client.create_collection(
    collection_name="standards",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)
client.create_payload_index("standards", "is_latest", PayloadSchemaType.BOOL)
client.create_payload_index("standards", "version_year", PayloadSchemaType.KEYWORD)
client.create_payload_index("standards", "standard_code", PayloadSchemaType.KEYWORD)
client.create_payload_index("standards", "ref_number", PayloadSchemaType.NOT_NEEDED)  # KEYWORD
```

## 17. docker-compose additions

```yaml
services:
  qdrant:                                        # unchanged
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    volumes: [qdrant_data:/qdrant/storage]
  mongo:                    # NEW
    image: mongo:7
    ports: ["27017:27017"]
    volumes: [mongo_data:/data/db]
volumes:
  qdrant_data:
  mongo_data:
```

---

## 18. Edge Cases (must hold)

- **Batch ingest partial failure:** one bad doc can't fail the batch — per-doc try/except, per-doc `"failed"` status in the response array.
- **Raw bulk store (`/documents`):** idempotent by `_id` + canonical JSON hash — identical re-submission is `skipped_unchanged` and writes nothing; missing/blank `_id` → that doc `"failed"`, batch continues. No embedding/Qdrant side effects (vector data is only owned by `/ingest`).
- **Missing/deform `searchable_text`:** that single doc → `"failed"`, batch continues.
- **Empty `message`:** `400` before touching the orchestrator.
- **Unknown `session_id` on GET:** `404` — never an empty array (ambiguous with "no messages").
- **Re-ingest older version after newer exists:** forced `is_latest=false` (§4 guard), supersede does not run, and downgrades never promote.
- **LC structured output malformed:** predictor coerces to `fallback` / nulls entities.
- **In-memory session store:** single-process only; flag before scaling to multiple workers (sessions split across workers'd break).
- **QdrantVectorStore metadata keys:** payload must be flat scalars (it is — see §6); nested dicts in `metadatas` get flattened by LCQdrant.

---

## 19. Build Order & Verification

**Build order (each step independently testable):**
1. Config + extensions + health route (deploy qdrant & mongo first).
2. Utils, schemas, repositories (qdrant/mongodb/cache).
3. Embedding + ingestion (+ collection script + seed script) — verify idempotency matrix manually.
4. Retrieval + RAG (LC chain).
5. NLU (predictor + handlers + registry).
6. Orchestrator + chat routes.
7. `scripts/verify.py` end-to-end.

**Verification checklist (`scripts/verify.py`):**
- [ ] Ingest batch twice → first run `inserted`×N, rerun `skipped_unchanged`×N; mutate a doc's `searchable_text` → `updated`.
- [ ] `POST /api/v1/documents` with array → `inserted`×N; identical resubmit → all `skipped_unchanged`; mutate one field → that doc `updated`; missing `_id` → `failed` without breaking the batch.
- [ ] Raw `/documents` store leaves Qdrant untouched (point count unchanged).
- [ ] Point count in Qdrant equals doc count (uuid5 deterministic upsert, no dupes).
- [ ] Ingest `v2021` doc, then re-ingest older `v2019` → v2019 stays `is_latest:false`; retrieval returns v2021.
- [ ] Greeting message → `intent: greeting`, no embedding call (fast path).
- [ ] "What are the corridor light levels in EN 12464-1 2019?" → `historical_query`, results from `version_year=2019`.
- [ ] "Compare 2019 and 2021" → canned fallback text, still `intent: comparison`.
- [ ] Empty message → `400`; unknown session GET → `404`.
- [ ] Two-turn chat keeps 3-role history visible in `GET /chat/{id}` and the LLM's prompt.
- [ ] `/health` reports qdrant/mongo ok while model not yet loaded, then `model_loaded: true` after first chat/ingest.
- [ ] One malformed doc in batch → `"failed"` only for that doc.