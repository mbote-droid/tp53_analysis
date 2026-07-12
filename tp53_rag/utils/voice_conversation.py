"""
============================================================
Precision Onco Africa - Conversational Voice Engine
utils/voice_conversation.py
============================================================
Powers the "Talk to Gemma" panel: a hands-light, multi-turn spoken conversation
where the clinician speaks a question, Gemma answers (with the running
conversation as context), Jarvis reads it back, and a polite sign-off is
understood.

Two testable pieces (all on-device, offline-first):

  1. faster-whisper transcription — 4x faster / ~half the RAM of openai-whisper
     on a CPU laptop, same models, no torch. Falls back to openai-whisper.

  2. dismiss-intent detection — recognises when the clinician is politely
     closing the conversation ("thank you Gemma, that will be all") so Gemma
     can sign off gracefully instead of treating it as a new question.

Multi-turn memory itself is handled upstream by `safe_query` (session memory),
so this module stays focused on speech-in and intent.
"""
from __future__ import annotations

import re
from typing import Optional

from utils.logger import log

# ── Dismiss intent ──────────────────────────────────────────────────────────
# Explicit closing phrases only — a bare "thank you" is NOT a dismissal (the
# clinician may be thanking and continuing), so we require a closing cue.
_DISMISS_PHRASES = (
    "that will be all", "that'll be all", "thatll be all", "that would be all",
    "that's all", "thats all", "that is all", "that's all for now",
    "we're done", "we are done", "were done", "i'm done", "im done",
    "no more questions", "no further questions",
    "goodbye", "good bye", "bye for now", "bye gemma",
    "that's enough", "thats enough", "that will do",
    "you can stop", "stop there", "let's stop", "lets stop", "dismissed",
)

DISMISS_RESPONSE = "Anytime, doctor. I'll be here when you need me."


def detect_dismiss_intent(text: Optional[str]) -> bool:
    """True when the clinician is politely ending the conversation.

    Matches explicit closing phrases (e.g. "thank you Gemma, that will be all")
    after light normalisation, so ordinary thanks-and-continue turns do NOT
    trigger a sign-off.
    """
    if not text:
        return False
    t = re.sub(r"[^a-z0-9\s']", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return any(phrase in t for phrase in _DISMISS_PHRASES)


# ── faster-whisper speech-to-text ───────────────────────────────────────────
_FW_MODEL = None


def faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _get_fw_model(size: str = "base"):
    """Lazy singleton so the model loads once (int8 on CPU = fast + low RAM)."""
    global _FW_MODEL
    if _FW_MODEL is None:
        from faster_whisper import WhisperModel
        log.info(f"Loading faster-whisper '{size}' (int8/cpu)…")
        _FW_MODEL = WhisperModel(size, device="cpu", compute_type="int8")
    return _FW_MODEL


def transcribe_fast(audio_path: str, language: str = "en") -> str:
    """Transcribe an audio file with faster-whisper. Raises on failure so the
    caller can fall back to openai-whisper."""
    model = _get_fw_model()
    segments, _info = model.transcribe(audio_path, language=language)
    return "".join(seg.text for seg in segments).strip()
