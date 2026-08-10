from flask import Blueprint, jsonify

from app.extensions import embedding_service, mongo_client, qdrant_client

health_bp = Blueprint("health", __name__, url_prefix="/api/v1")


@health_bp.route("/health", methods=["GET"])
def health():
    qdrant_ok = mongo_ok = False
    try:
        qdrant_client.get_collections()
        qdrant_ok = True
    except Exception:
        pass
    try:
        mongo_client.admin.command("ping")
        mongo_ok = True
    except Exception:
        pass

    return jsonify({
        "status": "ok" if (qdrant_ok and mongo_ok) else "degraded",
        "qdrant": qdrant_ok,
        "mongo": mongo_ok,
        "model_loaded": embedding_service.loaded,
    }), 200