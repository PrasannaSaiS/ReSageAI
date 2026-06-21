"""
ReSage AI — Flask backend
Production-grade resume screening application.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import time
import uuid
import warnings
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Optional extraction libraries ─────────────────────────────────────────────
try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore

try:
    import docx
except ImportError:
    docx = None  # type: ignore

try:
    import textract
except ImportError:
    textract = None  # type: ignore

# ── Google Generative AI — new SDK preferred, legacy fallback ──────────────────
genai = None
genai_types = None
_SDK: str | None = None

try:
    from google import genai as _g  # type: ignore
    from google.genai import types as _t  # type: ignore

    genai = _g
    genai_types = _t
    _SDK = "new"
except ImportError:
    pass

if genai is None:
    try:
        import google.generativeai as _g  # type: ignore

        genai = _g
        _SDK = "legacy"
    except ImportError:
        pass

# ── Environment ───────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent
_ROOT = _BASE.parent
load_dotenv(_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", uuid.uuid4().hex)
APP_DEBUG = os.getenv("APP_DEBUG", "false").lower() in ("1", "true", "yes")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── AI client init ────────────────────────────────────────────────────────────
_ai_client = None
AI_AVAILABLE = bool(genai and API_KEY)

if AI_AVAILABLE:
    try:
        if _SDK == "new":
            _ai_client = genai.Client(api_key=API_KEY)  # type: ignore[union-attr]
        else:  # legacy
            genai.configure(api_key=API_KEY)  # type: ignore[union-attr]
            _ai_client = genai
    except Exception:
        AI_AVAILABLE = False

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(_ROOT / "templates"),
    static_folder=str(_ROOT / "static"),
)

app.config.update(
    UPLOAD_FOLDER=str(_BASE / "uploads"),
    MAX_CONTENT_LENGTH=6 * 1024 * 1024,
    SECRET_KEY=SECRET_KEY,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=not APP_DEBUG,
    SESSION_COOKIE_SAMESITE="Lax",
)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

logging.basicConfig(level=logging.DEBUG if APP_DEBUG else logging.WARNING)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: set[str] = {"pdf", "doc", "docx"}

# Magic byte signatures for server-side content validation
_MAGIC: dict[str, bytes] = {
    "pdf": b"%PDF",
    "docx": b"PK\x03\x04",   # ZIP-based (Office Open XML)
    "doc": b"\xd0\xcf\x11\xe0",  # OLE2 Compound Document
}


# ── Utility helpers ────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_filename(filename: str) -> str:
    name = secure_filename(filename)
    if not name:
        raise ValueError("Invalid filename.")
    return f"{uuid.uuid4().hex}{Path(name).suffix.lower()}"


def validate_magic(path: Path, ext: str) -> bool:
    """Verify file header matches the claimed extension."""
    sig = _MAGIC.get(ext)
    if not sig:
        return False
    try:
        with open(path, "rb") as f:
            return f.read(len(sig)) == sig
    except OSError:
        return False


def extract_text_and_photo(path: Path) -> tuple[str, bool]:
    """Extract plain text and detect embedded photo from a resume file."""
    if not path.exists():
        raise FileNotFoundError("Uploaded file is unavailable.")

    ext = path.suffix.lower().lstrip(".")
    text = ""
    has_photo = False

    if ext == "pdf":
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is not installed.")
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            has_photo = any(page.images for page in pdf.pages)

    elif ext == "docx":
        if docx is None:
            raise RuntimeError("python-docx is not installed.")
        doc = docx.Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        has_photo = bool(doc.inline_shapes)

    elif ext == "doc":
        if textract is None:
            raise RuntimeError("textract is required for .doc files.")
        text = textract.process(str(path)).decode("utf-8", errors="ignore")

    else:
        raise ValueError("Unsupported file format.")

    # Fix camelCase word-boundary gaps from PDF extraction,
    # but skip tokens that are URLs, emails, or path-like (contain . / @ \ :)
    def _split_camel(m: re.Match) -> str:
        token_start = m.start()
        while token_start > 0 and text[token_start - 1] not in (" ", "\n", "\t"):
            token_start -= 1
        token = text[token_start : m.end()]
        if any(c in token for c in (".", "/", "@", "\\", ":")):
            return m.group(0)
        return m.group(1) + " " + m.group(2)

    text = re.sub(r"([a-z])([A-Z])", _split_camel, text)
    return text.strip(), has_photo


# ── AI interface ──────────────────────────────────────────────────────────────
def _build_system_prompt() -> str:
    today = _dt.date.today().strftime("%B %d, %Y")
    return (
        "You are a professional AI resume evaluator. "
        "Provide honest, structured scoring and actionable feedback. "
        f"Today's date is {today}. "
        "Use this date when reasoning about timelines, experience durations, "
        "and whether resume dates are past, present, or future."
    )


_FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

# In-process cache: text_hash -> (result_dict, timestamp)
import hashlib as _hashlib
_analysis_cache: dict[str, tuple[dict, float]] = {}
_ANALYSIS_CACHE_TTL = 3600.0  # 1 hour


def ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini; return empty string on any failure."""
    if not AI_AVAILABLE or _ai_client is None:
        return ""

    models_to_try = [GEMINI_MODEL] + [m for m in _FALLBACK_MODELS if m != GEMINI_MODEL]

    for model_name in models_to_try:
        try:
            system_prompt = _build_system_prompt()
            if _SDK == "new" and genai_types is not None:
                cfg = genai_types.GenerateContentConfig(
                    temperature=0.4,
                    top_p=1.0,
                    max_output_tokens=2048,
                    system_instruction=system_prompt,
                )
                resp = _ai_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=cfg,
                )
                return resp.text or ""

            # Legacy google-generativeai SDK
            model = _ai_client.GenerativeModel(  # type: ignore[attr-defined]
                model_name=model_name,
                system_instruction=system_prompt,
                generation_config={"temperature": 0.4, "max_output_tokens": 2048},
            )
            return model.generate_content(prompt).text or ""

        except Exception as exc:
            logger.warning("Gemini API error (model=%s): %s", model_name, exc)
            if "quota" in str(exc).lower() or "429" in str(exc):
                continue  # try next model
            return ""

    return ""


# Cache AI status — use a longer TTL and skip the live ping on index load
_ai_status_cache: tuple[bool, str, float] = (False, "", 0.0)
_AI_CACHE_TTL = 300.0  # 5 minutes


def check_ai_status() -> tuple[bool, str]:
    """Return AI availability without making a live API call."""
    global _ai_status_cache
    ok, msg, ts = _ai_status_cache
    if time.monotonic() - ts < _AI_CACHE_TTL:
        return ok, msg
    # Determine status from configuration only — no ping request
    result: tuple[bool, str] = (
        (True, "AI configured") if AI_AVAILABLE else (False, "AI client not configured")
    )
    _ai_status_cache = (*result, time.monotonic())
    return result


# ── Resume analysis ───────────────────────────────────────────────────────────
def detect_criteria(text: str, has_photo: bool) -> dict[str, bool]:
    low = text.lower()
    return {
        "Profile Photo": has_photo,
        "Summary": bool(re.search(r"\b(objective|summary|professional\s+summary|profile)\b", low)),
        "Skills": bool(re.search(r"\bskills?\b", low)),
        "Education": bool(re.search(r"\b(education|degree|b\.?tech|bachelor|master|phd|m\.?sc|mba|university|college)\b", low)),
        "Experience": bool(re.search(r"\b(experience|work history|employment|internship)\b", low)),
        "Projects": bool(re.search(r"\bprojects?\b", low)),
        "Contact Info": bool(
            re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", low)
            or re.search(r"\+?\d[\d (){}\-]{7,}", low)
        ),
        "Certifications": bool(re.search(r"\b(certif|aws|gcp|azure|pmp|cpa|cfa|comptia)\b", low)),
    }


def _ai_analyse(text: str) -> dict:
    """Single combined API call returning score, suggestions, errors, and job roles.
    Returns an empty dict if AI is unavailable or quota is exhausted.
    Result is cached by content hash for _ANALYSIS_CACHE_TTL seconds.
    """
    text_hash = _hashlib.sha256(text[:4000].encode()).hexdigest()
    cached, ts = _analysis_cache.get(text_hash, ({}, 0.0))
    if cached and time.monotonic() - ts < _ANALYSIS_CACHE_TTL:
        logger.info("AI analysis served from cache")
        return cached

    if not AI_AVAILABLE:
        return {}

    today = _dt.date.today().strftime("%B %d, %Y")
    prompt = (
        f"Today's date is {today}.\n"
        "Analyse the resume below and respond in EXACTLY this format with no extra text:\n\n"
        "SCORE: <integer 0-100>\n"
        "SUGGESTIONS:\n"
        "- <suggestion 1>\n"
        "- <suggestion 2>\n"
        "(up to 5 suggestions; do not flag current or upcoming dates as problems)\n"
        "ERRORS:\n"
        "- <grammar/spelling/writing error 1>\n"
        "(up to 8 real errors only; exclude URLs, usernames, emails, tech names, camelCase; "
        "if none write: - none)\n"
        "FIELD: <strongest career field in 3-5 words>\n"
        "ROLES:\n"
        "- <job role 1>\n"
        "- <job role 2>\n"
        "(3 to 5 roles)\n\n"
        "Use single-line bullets only; do not wrap suggestions, errors, or roles across multiple lines.\n"
        f"{text[:3500]}"
    )
    raw = ask_gemini(prompt)
    if not raw:
        return {}

    result: dict = {}
    section = None
    suggestions: list[str] = []
    errors: list[str] = []
    roles: list[str] = []
    current_item: str | None = None

    def _append_continuation(text_line: str) -> None:
        nonlocal current_item
        if not current_item:
            return
        more = text_line.lstrip("-* ").strip()
        if not more:
            return
        current_item += " " + more
        if section == "suggestions":
            suggestions[-1] = current_item
        elif section == "errors":
            errors[-1] = current_item
        elif section == "roles":
            roles[-1] = current_item

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith("SCORE:"):
            section = None
            current_item = None
            m = re.search(r"(\d{1,3})", stripped)
            if m:
                result["score"] = max(0, min(int(m.group(1)), 100))
        elif upper.startswith("SUGGESTIONS:"):
            section = "suggestions"
            current_item = None
        elif upper.startswith("ERRORS:"):
            section = "errors"
            current_item = None
        elif upper.startswith("FIELD:"):
            section = None
            current_item = None
            result["field"] = stripped.split(":", 1)[1].strip()
        elif upper.startswith("ROLES:"):
            section = "roles"
            current_item = None
        elif stripped.startswith("-"):
            item = stripped.lstrip("-* ").strip()
            current_item = None
            if not item or item.lower() == "none":
                continue
            if section == "suggestions":
                suggestions.append(item)
                current_item = item
            elif section == "errors":
                if (
                    len(item) > 5
                    and not re.search(r"https?://|github\.com|linkedin\.com|@", item, re.I)
                    and not re.search(r"\b(username|url|link|handle|profile)\b", item, re.I)
                ):
                    errors.append(item)
                    current_item = item
            elif section == "roles":
                roles.append(item)
                current_item = item
        elif section in ("suggestions", "errors", "roles") and current_item is not None:
            _append_continuation(stripped)

    result["suggestions"] = suggestions[:5]
    result["errors"] = errors[:8]
    result["roles"] = roles[:5]

    if result:
        _analysis_cache[text_hash] = (result, time.monotonic())
    return result


def score_and_suggest(text: str) -> tuple[int, list[str]]:
    ai = _ai_analyse(text)
    if ai:
        return ai.get("score", 55), ai.get("suggestions", [])

    # Rule-based fallback
    criteria = detect_criteria(text, False)
    score = 40 + sum(criteria.values()) * 7
    suggestions = []
    if not criteria["Summary"]:
        suggestions.append("Add a concise professional summary at the top of your resume.")
    if not criteria["Skills"]:
        suggestions.append("Include a dedicated skills section with role-relevant tools and technologies.")
    if not criteria["Projects"]:
        suggestions.append("Describe 1–2 key projects with measurable outcomes and technologies used.")
    if not criteria["Contact Info"]:
        suggestions.append("Add your email address and phone number in the resume header.")
    if len(text) < 600:
        suggestions.append("Expand with specific achievements, metrics, and impact statements.")
    return max(40, min(score, 92)), suggestions[:5]


def grammar_errors(text: str) -> list[str]:
    ai = _ai_analyse(text)
    if ai:
        return ai.get("errors", [])

    # Static typo fallback
    _TYPOS = {
        "teh": '"teh" should be "the"',
        "recieve": '"recieve" should be "receive"',
        "adress": '"adress" should be "address"',
        "acheivement": '"acheivement" should be "achievement"',
        "managment": '"managment" should be "management"',
        "responsibilty": '"responsibilty" should be "responsibility"',
    }
    return [msg for typo, msg in _TYPOS.items() if re.search(rf"\b{typo}\b", text, re.I)]


def recommend_jobs(text: str) -> tuple[list[str], str]:
    ai = _ai_analyse(text)
    if ai and ai.get("roles"):
        return ai["roles"], ai.get("field", "Not identified")

    low = text.lower()
    if any(k in low for k in ("research", "publication", "experiment", "laboratory")):
        return ["Research Analyst", "Research Scientist", "Academic Researcher"], "Research & Development"
    if any(k in low for k in ("data", "machine learning", "statistics", "analytics")):
        return ["Data Analyst", "Machine Learning Engineer", "Data Scientist", "BI Analyst"], "Data Science & Analytics"
    if any(k in low for k in ("software", "python", "java", "developer", "engineer", "backend", "frontend")):
        return ["Software Engineer", "Backend Developer", "Full Stack Engineer", "DevOps Engineer"], "Software Engineering"
    if any(k in low for k in ("design", "figma", "ui", "ux", "wireframe")):
        return ["UI/UX Designer", "Product Designer", "Frontend Developer"], "Design & Product"
    return ["Project Coordinator", "Operations Analyst", "Business Analyst"], "Business Operations"


# ── Security headers ──────────────────────────────────────────────────────────
@app.after_request
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
    response.headers.update(
        {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "X-XSS-Protection": "1; mode=block",
            "Content-Security-Policy": csp,
        }
    )
    if not APP_DEBUG:
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    return response


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_e):
    return render_template("index.html", error="File is too large. Maximum upload size is 6 MB.", ai_enabled=False), 413


@app.errorhandler(400)
def handle_400(exc):
    return render_template(
        "results.html",
        score=0,
        criteria={},
        errors=[getattr(exc, "description", str(exc))],
        suggestions=[],
        job_roles=[],
        job_field="N/A",
        ai_enabled=AI_AVAILABLE,
    ), 400


@app.errorhandler(500)
def handle_500(_e):
    logger.exception("Internal server error")
    return render_template(
        "results.html",
        score=0,
        criteria={},
        errors=["An unexpected error occurred. Please try again."],
        suggestions=[],
        job_roles=[],
        job_field="N/A",
        ai_enabled=AI_AVAILABLE,
    ), 500


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    ai_ok, ai_msg = check_ai_status() if AI_AVAILABLE else (False, "AI disabled")
    return render_template("index.html", ai_enabled=ai_ok, ai_message=ai_msg)


@app.route("/upload", methods=["POST"])
def upload():
    resume = request.files.get("resume")
    if not resume or not resume.filename:
        return render_template("index.html", error="No file selected. Please choose a PDF, DOC, or DOCX file.", ai_enabled=False), 400

    if not allowed_file(resume.filename):
        return render_template("index.html", error="Unsupported format. Only PDF, DOC, and DOCX files are accepted.", ai_enabled=False), 400

    try:
        fname = make_filename(resume.filename)
        dest = Path(app.config["UPLOAD_FOLDER"]) / fname
        resume.save(dest)
    except Exception:
        logger.exception("File save failed")
        return render_template("index.html", error="Could not save the file. Please try again.", ai_enabled=False), 500

    # Validate file content against magic bytes
    ext = Path(fname).suffix.lstrip(".")
    if not validate_magic(dest, ext):
        dest.unlink(missing_ok=True)
        return render_template("index.html", error="File content doesn't match its extension. Please upload a valid resume.", ai_enabled=False), 400

    return redirect(url_for("analysis", filename=fname))


@app.route("/analysis")
def analysis():
    """Render the loading screen. Client-side JS will redirect to /results."""
    filename = request.args.get("filename", "").strip()
    if not filename or any(c in filename for c in ("/", "\\", "..")):
        abort(400, "Invalid file reference.")
    upload_path = Path(app.config["UPLOAD_FOLDER"]) / filename
    if not upload_path.exists():
        abort(400, "Resume not found. Please upload again.")
    return render_template("analysis.html", filename=filename)


@app.route("/results")
def results():
    """Process the uploaded resume and render the analysis results."""
    filename = request.args.get("filename", "").strip()
    if not filename or any(c in filename for c in ("/", "\\", "..")):
        abort(400, "Invalid file reference.")

    upload_path = Path(app.config["UPLOAD_FOLDER"]) / filename
    # Prevent path traversal: ensure resolved path stays within UPLOAD_FOLDER
    try:
        upload_path.resolve().relative_to(Path(app.config["UPLOAD_FOLDER"]).resolve())
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


@app.route("/ai_status")
def ai_status():
    ok, msg = check_ai_status() if AI_AVAILABLE else (False, "AI disabled")
    return jsonify({"ai_available": ok, "message": msg})


if __name__ == "__main__":
    app.run(debug=APP_DEBUG, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))