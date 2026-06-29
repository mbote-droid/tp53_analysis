"""
============================================================
Precision Onco Africa - Security Hardening Utilities
utils/security.py
============================================================
Central, reusable defences applied at every untrusted boundary: file uploads,
free-text that reaches the LLM, filenames, and content destined for HTML
components. Pure functions, no I/O, fully unit-tested with adversarial inputs.

Threat model and mitigations are documented in SECURITY.md; this module is the
implementation those mitigations point to.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, Optional, Tuple

# ── Upload limits (env-configurable; conservative for an 8GB host) ──
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))   # 5 MB
MAX_VCF_LINES = int(os.getenv("MAX_VCF_LINES", 200_000))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 10 * 1024 * 1024))    # 10 MB
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", 4000))

# Magic bytes of common executable / archive formats we never accept as data.
_DANGEROUS_MAGIC = (
    b"MZ",            # Windows PE / DOS executable
    b"\x7fELF",       # Linux ELF
    b"PK\x03\x04",    # ZIP (incl. docx/xlsx/jar — and zip bombs)
    b"\x1f\x8b",      # gzip
    b"%PDF",          # PDF (not a VCF)
    b"\xca\xfe\xba\xbe",  # Java class / Mach-O fat
    b"\xff\xd8\xff",  # JPEG (when a VCF is expected)
    b"\x89PNG",       # PNG (when a VCF is expected)
)

# Prompt-injection phrases (detection, not silent mutation — we flag + cap).
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"ignore (all |the )?(previous|prior|above) (instructions|prompts)",
        r"disregard (the |your )?(system|previous) (prompt|instructions)",
        r"you are now\b",
        r"\b(dan|jailbreak|developer mode)\b",
        r"reveal (your |the )?(system )?(prompt|instructions)",
        r"print (your |the )?(system )?(prompt|instructions)",
        r"(act|behave) as (if you are|a) (dan|developer mode|jailbreak)",
        r"</?(system|assistant|user)>",          # role-tag injection
        r"\bBEGIN SYSTEM\b|\bEND SYSTEM\b",
    )
]


def is_probably_binary(data: bytes, sniff: int = 8192) -> bool:
    """Heuristic: null bytes or a high non-text ratio ⇒ binary (e.g. an
    executable disguised as a .vcf). True ⇒ reject as a text upload."""
    if not data:
        return False
    chunk = bytes(data[:sniff])
    if b"\x00" in chunk:
        return True
    # Count bytes outside the printable/whitespace ASCII range + common UTF-8.
    text_bytes = set(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0D}
    nontext = sum(1 for b in chunk if b not in text_bytes and b < 0x80)
    return (nontext / max(len(chunk), 1)) > 0.30


def has_dangerous_magic(data: bytes) -> bool:
    """True if the file starts with known executable/archive magic bytes."""
    head = bytes(data[:8])
    return any(head.startswith(m) for m in _DANGEROUS_MAGIC)


def safe_filename(name: Optional[str]) -> str:
    """Strip directories, control chars and traversal sequences from a filename.
    Always returns a non-empty, path-safe basename."""
    raw = str(name or "").strip()
    raw = raw.replace("\\", "/").split("/")[-1]          # drop any path
    raw = unicodedata.normalize("NFKC", raw)
    raw = re.sub(r"[\x00-\x1f]", "", raw)                # control chars
    raw = re.sub(r"[^A-Za-z0-9._-]", "_", raw)           # whitelist
    raw = raw.lstrip(".") or "upload"                    # no leading dots
    return raw[:120]


def validate_upload(data: bytes, filename: str = "",
                    max_bytes: int = MAX_UPLOAD_BYTES,
                    allowed_ext: Tuple[str, ...] = (),
                    require_text: bool = True) -> Dict:
    """Validate an uploaded file's size, extension, and content nature.

    Returns {ok, reason, friendly} — never raises. ``friendly`` is a
    user-facing message safe to show in the UI.
    """
    if data is None:
        return {"ok": False, "reason": "no_data",
                "friendly": "No file received — please choose a file and retry."}
    size = len(data)
    if size == 0:
        return {"ok": False, "reason": "empty",
                "friendly": "That file is empty. Please upload a valid file."}
    if size > max_bytes:
        mb = max_bytes / (1024 * 1024)
        return {"ok": False, "reason": "too_large",
                "friendly": f"That file is too large (limit {mb:.0f} MB). "
                            "Please upload a smaller file."}
    if has_dangerous_magic(data):
        return {"ok": False, "reason": "dangerous_type",
                "friendly": "That file looks like a program or archive, not the "
                            "expected data file. Upload was blocked for safety."}
    if allowed_ext:
        name = safe_filename(filename).lower()
        if not name.endswith(tuple(e.lower() for e in allowed_ext)):
            exts = ", ".join(allowed_ext)
            return {"ok": False, "reason": "bad_extension",
                    "friendly": f"Unsupported file type. Please upload one of: {exts}."}
    if require_text and is_probably_binary(data):
        return {"ok": False, "reason": "binary",
                "friendly": "That doesn't look like a text data file. Please "
                            "upload a valid, uncorrupted file."}
    return {"ok": True, "reason": "ok", "friendly": ""}


def looks_like_vcf(text: str) -> bool:
    """True if the text has VCF structure: a ## header or a tab/space-delimited
    line whose 2nd column is an integer position."""
    t = str(text or "")
    if "##fileformat=VCF" in t or "#CHROM" in t:
        return True
    for line in t.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = re.split(r"[\t ]+", line)
        if len(cols) >= 5 and cols[1].isdigit():
            return True
    return False


def detect_prompt_injection(text: str) -> Dict:
    """Flag (do not silently rewrite) likely prompt-injection attempts."""
    t = str(text or "")
    hits = [p.pattern for p in _INJECTION_PATTERNS if p.search(t)]
    return {"flagged": bool(hits), "patterns": hits}


def sanitize_for_prompt(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Neutralise role-tag injection and cap length before text reaches the LLM.
    Conservative: strips control chars and fake chat-role tags, truncates."""
    t = str(text or "")
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", t)        # control chars
    t = re.sub(r"</?(system|assistant|user)\s*>", " ", t, flags=re.IGNORECASE)
    if len(t) > max_chars:
        t = t[:max_chars] + " …"
    return t.strip()
