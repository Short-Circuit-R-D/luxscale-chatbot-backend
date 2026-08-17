import threading
import uuid
from datetime import datetime, timezone, timedelta

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

SESSION_TTL = timedelta(hours=24)


class InMemorySessionCacheRepository:
    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _key_valid(self, session_id: str) -> bool:
        entry = self._store.get(session_id)
        if entry is None:
            return False
        if entry["expires_at"] < datetime.now(timezone.utc):
            self._store.pop(session_id, None)
            return False
        return True

    def create_session_id(self) -> str:
        return str(uuid.uuid4())

    def get_history(self, session_id: str) -> list[dict]:
        with self._lock:
            if not self._key_valid(session_id):
                return []
            return list(self._store[session_id]["messages"])

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        intent: str | None = None,
        simulator: dict | None = None,
    ):
        with self._lock:
            if session_id not in self._store or not self._key_valid(session_id):
                self._store[session_id] = {"messages": [], "expires_at": None}
            self._store[session_id]["messages"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "intent": intent,
                "simulator": simulator,
            })
            self._store[session_id]["expires_at"] = datetime.now(timezone.utc) + SESSION_TTL

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return self._key_valid(session_id)

    def clear_session(self, session_id: str):
        with self._lock:
            self._store.pop(session_id, None)

    def get_history_messages(self, session_id: str) -> list[BaseMessage]:
        turns = self.get_history(session_id)
        messages = []
        for turn in turns:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
        return messages


class SessionChatHistory(BaseChatMessageHistory):
    """Per-session LC adapter over the shared cache (implements
    BaseChatMessageHistory so LangChain chains can consume it)."""

    def __init__(self, session_id: str, cache: InMemorySessionCacheRepository):
        self.session_id = session_id
        self.cache = cache

    @property
    def messages(self) -> list[BaseMessage]:
        return self.cache.get_history_messages(self.session_id)

    def add_message(self, message: BaseMessage):
        if isinstance(message, HumanMessage):
            self.cache.append_turn(self.session_id, "user", message.content)
        else:
            self.cache.append_turn(self.session_id, "assistant", str(message.content))

    def clear(self):
        self.cache.clear_session(self.session_id)


session_cache_repo = InMemorySessionCacheRepository()