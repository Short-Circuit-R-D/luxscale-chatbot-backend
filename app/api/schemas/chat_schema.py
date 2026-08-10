from pydantic import BaseModel
from typing import Optional, List


class ChatMessageRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatMessageResponse(BaseModel):
    session_id: str
    response: str
    intent: str


class ChatTurn(BaseModel):
    role: str
    content: str
    timestamp: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatTurn]