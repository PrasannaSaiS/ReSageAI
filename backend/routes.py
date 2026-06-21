"""
routes.py — Flask Blueprint: HTTP routes, error handlers, security headers,
            and upload-related file validation helpers.
Single Responsibility: owns all request/response handling.
Dependency Inversion: depends on extractor and analyser abstractions.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from flask import (
    Blueprint, abort, current_app, jsonify,
    redirect, render_template, request, url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import config
from ai_client import AI_AVAILABLE, check_ai_status
from analyser import detect_criteria, grammar_errors, recommend_jobs, score_and_suggest
from extractor import extract_text_and_photo

logger = logging.getLogger(__name__)
bp = Blueprint("main", __name__)


# ── Upload helpers ─────────────────────────────────────────────────────────────
def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def _make_filename(filename: str) -> str:
    name = secure_filename(filename)
    if not name:
        raise ValueError("Invalid filename.")
    return f"{uuid.uuid4().hex}{Path(name).suffix.lower()}"


def _validate_magic(path: Path, ext: str) -> bool:
    sig = config.MAGIC_BYTES.get(ext)
    if not sig:
        return False
    try:
        with open(path, "rb") as f:
            return f.read(len(sig)) == sig
    except OSError:
        return False


# ── Security headers ───────────────────────────────────────────────────────────
@bp.after_request
def set_security_headers(response):
    csp = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    response.headers.update({
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "X-XSS-Protection": "1; mode=block",
        "Content-Security-Policy": csp,
    })
    if not config.APP_DEBUG:
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains; preload"
        )
    return response


# ── Error handlers ─────────────────────────────────────────────────────────────
@bp.app_errorhandler(RequestEntityTooLarge)
def handle_too_large(_e):
    return render_template(
        "index.html",
        error="File is too large. Maximum upload size is 6 MB.",
        ai_enabled=False,
    ), 413


@bp.app_errorhandler(400)
def handle_400(exc):
    return render_template(
        "results.html",
        score=0, criteria={},
        errors=[getattr(exc, "description", str(exc))],
        suggestions=[], job_roles=[], job_field="N/A",
        ai_enabled=AI_AVAILABLE,
    ), 400


@bp.app_errorhandler(500)
def handle_500(_e):
    logger.exception("Internal server error")
    return render_template(
        "results.html",
        score=0, criteria={},
        errors=["An unexpected error occurred. Please try again."],
        suggestions=[], job_roles=[], job_field="N/A",
        ai_enabled=AI_AVAILABLE,
    ), 500


# ── Routes ─────────────────────────────────────────────────────────────────────
@bp.route("/")
def index():
    ai_ok, ai_msg = check_ai_status() if AI_AVAILABLE else (False, "AI disabled")
    return render_template("index.html", ai_enabled=ai_ok, ai_message=ai_msg)


@bp.route("/upload", methods=["POST"])
def upload():
    resume = request.files.get("resume")
    if not resume or not resume.filename:
        return render_template(
            "index.html",
            error="No file selected. Please choose a PDF, DOC, or DOCX file.",
            ai_enabled=False,
        ), 400

    if not _allowed_file(resume.filename):
        return render_template(
            "index.html",
            error="Unsupported format. Only PDF, DOC, and DOCX files are accepted.",
            ai_enabled=False,
        ), 400

    upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
    try:
        fname = _make_filename(resume.filename)
        dest = upload_folder / fname
        resume.save(dest)
    except Exception:
        logger.exception("File save failed")
        return render_template(
            "index.html",
            error="Could not save the file. Please try again.",
            ai_enabled=False,
        ), 500

    ext = Path(fname).suffix.lstrip(".")
    if not _validate_magic(dest, ext):
        dest.unlink(missing_ok=True)
        return render_template(
            "index.html",
            error="File content doesn't match its extension. Please upload a valid resume.",
            ai_enabled=False,
        ), 400

    return redirect(url_for("main.analysis", filename=fname))


@bp.route("/analysis")
def analysis():
    filename = request.args.get("filename", "").strip()
    if not filename or any(c in filename for c in ("/", "\\", "..")):
        abort(400, "Invalid file reference.")
    upload_path = Path(current_app.config["UPLOAD_FOLDER"]) / filename
    if not upload_path.exists():
        abort(400, "Resume not found. Please upload again.")
    return render_template("analysis.html", filename=filename)


@bp.route("/results")
def results():
    filename = request.args.get("filename", "").strip()
    if not filename or any(c in filename for c in ("/", "\\", "..")):
        abort(400, "Invalid file reference.")

    upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
    upload_path = upload_folder / filename
    try:
        upload_path.resolve().relative_to(upload_folder.resolve())
    except ValueError:
        abort(400, "Invalid file reference.")

    if not upload_path.exists():
        abort(400, "Resume not found. Please upload a new file.")

    try:
        text, has_photo = extract_text_and_photo(upload_path)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        abort(400, str(exc))
    finally:
        upload_path.unlink(missing_ok=True)

    criteria = detect_criteria(text, has_photo)
    if sum(criteria.values()) < 2:
        abort(400, "The uploaded file does not appear to contain recognisable resume content.")

    score, suggestions = score_and_suggest(text)
    errors = grammar_errors(text)
    job_roles, job_field = recommend_jobs(text)

    return render_template(
        "results.html",
        score=score,
        criteria=criteria,
        errors=errors[:10],
        suggestions=suggestions,
        job_roles=job_roles,
        job_field=job_field,
        ai_enabled=AI_AVAILABLE,
    )


@bp.route("/ai_status")
def ai_status():
    ok, msg = check_ai_status() if AI_AVAILABLE else (False, "AI disabled")
    return jsonify({"ai_available": ok, "message": msg})
