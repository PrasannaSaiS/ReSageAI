"""
app.py — Flask application factory.
Single Responsibility: create and configure the Flask app instance.
All logic lives in config, extractor, ai_client, analyser, and routes.
"""
from __future__ import annotations

import logging
import os

from flask import Flask

import config
from routes import bp


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(config.ROOT_DIR / "templates"),
        static_folder=str(config.ROOT_DIR / "static"),
    )

    app.config.update(
        UPLOAD_FOLDER=str(config.UPLOAD_DIR),
        MAX_CONTENT_LENGTH=config.MAX_UPLOAD_BYTES,
        SECRET_KEY=config.SECRET_KEY,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=not config.APP_DEBUG,
        SESSION_COOKIE_SAMESITE="Lax",
    )

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if config.APP_DEBUG else logging.WARNING
    )

    app.register_blueprint(bp)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        debug=config.APP_DEBUG,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
    )
