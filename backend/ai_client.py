"""
ai_client.py — Gemini AI client initialisation and low-level call wrapper.
Single Responsibility: owns SDK selection, client init, and raw prompt dispatch.
Open/Closed: swap or add a new SDK by adding a branch; callers are unaffected.
"""
from __future__ import annotations

import datetime as _dt
import logging
import time

import config

logger = logging.getLogger(__name__)

# ── SDK detection (new google.genai preferred, legacy fallback) ────────────────
_genai = None
_genai_types = None
_SDK: str | None = None

try:
    from google import genai as _g       # type: ignore
    from google.genai import types as _t  # type: ignore
    _genai = _g
    _genai_types = _t
    _SDK = "new"
except ImportError:
    pass

if _genai is None:
    try:
        import google.generativeai as _g  # type: ignore
        _genai = _g
        _SDK = "legacy"
    except ImportError:
        pass

# ── Client initialisation ──────────────────────────────────────────────────────
_client = None
AI_AVAILABLE = bool(_genai and config.GEMINI_API_KEY)

if AI_AVAILABLE:
    try:
        if _SDK == "new":
            _client = _genai.Client(api_key=config.GEMINI_API_KEY)  # type: ignore
        else:
            _genai.configure(api_key=config.GEMINI_API_KEY)         # type: ignore
            _client = _genai
    except Exception:
        AI_AVAILABLE = False

# ── System prompt (evaluated fresh each call so date is always current) ────────
def _system_prompt() -> str:
    today = _dt.date.today().strftime("%B %d, %Y")
    return (
        "You are a professional AI resume evaluator. "
        "Provide honest, structured scoring and actionable feedback. "
        f"Today's date is {today}. "
        "Use this date when reasoning about timelines, experience durations, "
        "and whether resume dates are past, present, or future."
    )


# ── AI status cache (config-only, no live ping) ────────────────────────────────
_status_cache: tuple[bool, str, float] = (False, "", 0.0)


def check_ai_status() -> tuple[bool, str]:
    """Return AI availability based on configuration — no API request."""
    global _status_cache
    ok, msg, ts = _status_cache
    if time.monotonic() - ts < config.AI_STATUS_CACHE_TTL:
        return ok, msg
    result: tuple[bool, str] = (
        (True, "AI configured") if AI_AVAILABLE else (False, "AI client not configured")
    )
    # assign explicitly to preserve precise tuple typing (bool, str, float)
    _status_cache = (result[0], result[1], time.monotonic())
    return result


# ── Core prompt dispatcher ─────────────────────────────────────────────────────
def ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini across the fallback model chain.
    Returns empty string on total failure — callers handle fallback logic.
    """
    if not AI_AVAILABLE or _client is None:
        return ""

    models = [config.GEMINI_MODEL] + [
        m for m in config.GEMINI_FALLBACKS if m != config.GEMINI_MODEL
    ]

    for model_name in models:
        try:
            sys_prompt = _system_prompt()
            if _SDK == "new" and _genai_types is not None:
                cfg = _genai_types.GenerateContentConfig(
                    temperature=config.GEMINI_TEMPERATURE,
                    top_p=1.0,
                    max_output_tokens=config.GEMINI_MAX_TOKENS,
                    system_instruction=sys_prompt,
                )
                resp = _client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=cfg,
                )
                return resp.text or ""

            # Legacy SDK
            model = _client.GenerativeModel(  # type: ignore
                model_name=model_name,
                system_instruction=sys_prompt,
                generation_config={
                    "temperature": config.GEMINI_TEMPERATURE,
                    "max_output_tokens": config.GEMINI_MAX_TOKENS,
                },
            )
            return model.generate_content(prompt).text or ""

        except Exception as exc:
            logger.warning("Gemini API error (model=%s): %s", model_name, exc)
            if "quota" in str(exc).lower() or "429" in str(exc):
                continue
            return ""

    return ""
