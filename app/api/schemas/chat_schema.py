from pydantic import BaseModel
from typing import Optional, List


class ChatMessageRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class SimulatorAttachment(BaseModel):
    id: str
    title: str
    iframe_url: str


class ChatMessageResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    simulator: Optional[SimulatorAttachment] = None


class ChatTurn(BaseModel):
    role: str
    content: str
    timestamp: str
    intent: Optional[str] = None
    simulator: Optional[SimulatorAttachment] = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatTurn]