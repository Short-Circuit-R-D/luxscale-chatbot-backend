from flask import Blueprint, request, jsonify

from app.api.schemas.chat_schema import (
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    SimulatorAttachment,
)
from app.extensions import orchestrator
from app.repositories.cache.session_cache_repository import session_cache_repo

chat_bp = Blueprint("chat", __name__, url_prefix="/api/v1/chat")


@chat_bp.route("/message", methods=["POST"])
def send_message():
    body = ChatMessageRequest(**request.get_json(force=True))
    if not body.message.strip():
        return jsonify({"error": "message must not be empty"}), 400

    session_id = body.session_id or session_cache_repo.create_session_id()
    result = orchestrator.run(session_id=session_id, user_message=body.message)

    return jsonify(ChatMessageResponse(
        session_id=session_id,
        response=result.text,
        intent=result.intent,
        simulator=SimulatorAttachment.model_validate(result.simulator)
        if result.simulator
        else None,
    ).model_dump()), 200


@chat_bp.route("/<session_id>", methods=["GET"])
def get_session(session_id: str):
    if not session_cache_repo.session_exists(session_id):
        return jsonify({"error": "session not found"}), 404

    history = session_cache_repo.get_history(session_id)
    return jsonify(ChatHistoryResponse(
        session_id=session_id, messages=history,
    ).model_dump()), 200