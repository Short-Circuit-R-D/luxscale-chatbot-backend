import os

from flask import Flask
from flask_cors import CORS

from app import register_blueprints


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    app.config.from_object("app.config.Config")
    register_blueprints(app)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)