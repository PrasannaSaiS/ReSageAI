"""
analyser.py — Resume analysis: scoring, suggestions, quality checks, job matching.
Single Responsibility: owns all analysis logic.
Open/Closed: add new analysis dimensions by extending _ai_analyse output keys
             and adding corresponding public functions — without modifying existing ones.
Dependency Inversion: depends on ai_client.ask_gemini abstraction, not the SDK directly.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import re
import time

import config
from ai_client import AI_AVAILABLE, ask_gemini

logger = logging.getLogger(__name__)

# ── In-process result cache ────────────────────────────────────────────────────
_cache: dict[str, tuple[dict, float]] = {}


# ── Rule-based section detector ───────────────────────────────────────────────
def detect_criteria(text: str, has_photo: bool) -> dict[str, bool]:
    low = text.lower()
    return {
        "Profile Photo": has_photo,
        "Summary": bool(re.search(r"\b(objective|summary|professional\s+summary|profile)\b", low)),
        "Skills": bool(re.search(r"\bskills?\b", low)),
        "Education": bool(re.search(
            r"\b(education|degree|b\.?tech|bachelor|master|phd|m\.?sc|mba|university|college)\b", low
        )),
        "Experience": bool(re.search(r"\b(experience|work history|employment|internship)\b", low)),
        "Projects": bool(re.search(r"\bprojects?\b", low)),
        "Contact Info": bool(
            re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", low)
            or re.search(r"\+?\d[\d (){}\-]{7,}", low)
        ),
        "Certifications": bool(re.search(r"\b(certif|aws|gcp|azure|pmp|cpa|cfa|comptia)\b", low)),
    }


# ── Single combined AI call (1 request per resume) ────────────────────────────
def _ai_analyse(text: str) -> dict:
    """Return a dict with keys: score, suggestions, errors, roles, field.
    Cached by SHA-256 of the first 4000 chars for ANALYSIS_CACHE_TTL seconds.
    Returns empty dict when AI is unavailable or all quotas are exhausted.
    """
    text_hash = hashlib.sha256(text[:4000].encode()).hexdigest()
    cached, ts = _cache.get(text_hash, ({}, 0.0))
    if cached and time.monotonic() - ts < config.ANALYSIS_CACHE_TTL:
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
    section: str | None = None
    suggestions: list[str] = []
    errors: list[str] = []
    roles: list[str] = []
    current_item: str | None = None

    def _append_continuation(line: str) -> None:
        nonlocal current_item
        if not current_item:
            return
        more = line.lstrip("-* ").strip()
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
            section, current_item = None, None
            m = re.search(r"(\d{1,3})", stripped)
            if m:
                result["score"] = max(0, min(int(m.group(1)), 100))
        elif upper.startswith("SUGGESTIONS:"):
            section, current_item = "suggestions", None
        elif upper.startswith("ERRORS:"):
            section, current_item = "errors", None
        elif upper.startswith("FIELD:"):
            section, current_item = None, None
            result["field"] = stripped.split(":", 1)[1].strip()
        elif upper.startswith("ROLES:"):
            section, current_item = "roles", None
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
        _cache[text_hash] = (result, time.monotonic())
    return result


# ── Public analysis functions (each reads from the shared cached AI result) ────
def score_and_suggest(text: str) -> tuple[int, list[str]]:
    ai = _ai_analyse(text)
    if ai:
        return ai.get("score", 55), ai.get("suggestions", [])

    criteria = detect_criteria(text, False)
    score = 40 + sum(criteria.values()) * 7
    suggestions: list[str] = []
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
