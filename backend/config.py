"""
config.py — Environment and application configuration.
All constants live here; nothing else imports .env directly.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

_BASE = Path(__file__).resolve().parent        # backend/
_ROOT = _BASE.parent                           # project root

load_dotenv(_ROOT / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = _BASE
ROOT_DIR   = _ROOT
UPLOAD_DIR = _BASE / "uploads"

# ── Flask ──────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", uuid.uuid4().hex)
APP_DEBUG  = os.getenv("APP_DEBUG", "false").lower() in ("1", "true", "yes")
MAX_UPLOAD_BYTES = 6 * 1024 * 1024  # 6 MB

# ── Gemini ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACKS   = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
GEMINI_MAX_TOKENS  = 2048
GEMINI_TEMPERATURE = 0.4

# ── File validation ────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS: set[str] = {"pdf", "doc", "docx"}

MAGIC_BYTES: dict[str, bytes] = {
    "pdf":  b"%PDF",
    "docx": b"PK\x03\x04",       # ZIP / Office Open XML
    "doc":  b"\xd0\xcf\x11\xe0", # OLE2 Compound Document
}

# ── Cache ──────────────────────────────────────────────────────────────────────
ANALYSIS_CACHE_TTL = 3600.0  # seconds
AI_STATUS_CACHE_TTL = 300.0  # seconds
