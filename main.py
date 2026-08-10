from app import register_blueprints
from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object("app.config.Config")
    register_blueprints(app)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)