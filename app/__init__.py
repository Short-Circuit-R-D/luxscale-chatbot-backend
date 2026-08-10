from flask import Flask

from app.api.routes.chat_routes import chat_bp
from app.api.routes.health_routes import health_bp
from app.api.routes.ingest_routes import ingest_bp


def register_blueprints(app: Flask):
    app.register_blueprint(health_bp)
    app.register_blueprint(ingest_bp)
    app.register_blueprint(chat_bp)