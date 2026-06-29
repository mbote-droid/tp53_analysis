"""
============================================================
Precision Onco Africa - Voice Output ("Jarvis")
utils/voice_output.py
============================================================
Spoken responses, the lightweight way: the platform speaks its answers using
the browser's built-in Web Speech API (``speechSynthesis``). That means the
audio is generated in the *user's* browser — no server-side TTS engine, no
extra Python dependency, and it works fully offline once the page is loaded.

Pairs with the existing Whisper voice *input* to make the platform fully
conversational (speak a question → hear the answer).

Pure: returns a self-contained HTML/JS snippet. Injection-safe — the spoken
text is embedded as a JSON-encoded string, never as raw HTML.
"""
from __future__ import annotations

import json
import re
from typing import Optional

# Cap spoken length so a huge report doesn't monologue for minutes.
_MAX_SPEAK_CHARS = 600


def is_speakable(text: Optional[str]) -> bool:
    """True if there is meaningful, non-whitespace text to speak."""
    return bool(str(text or "").strip())


def _clean_for_speech(text: str) -> str:
    """Strip markdown/markup and collapse whitespace so speech sounds natural."""
    t = str(text or "")
    t = re.sub(r"[`*_#>|]", " ", t)            # markdown symbols
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)  # links → label
    t = re.sub(r"<[^>]+>", " ", t)             # any stray HTML tags
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > _MAX_SPEAK_CHARS:
        t = t[:_MAX_SPEAK_CHARS].rsplit(" ", 1)[0] + " …"
    return t


def speak_html(text: Optional[str], autoplay: bool = True,
               rate: float = 1.0, label: str = "Speak response") -> str:
    """Return a self-contained HTML/JS control that speaks *text* in-browser.

    Always returns non-empty HTML. If there is nothing to speak, returns a
    small disabled control rather than an empty string.
    """
    cleaned = _clean_for_speech(text) if is_speakable(text) else ""
    payload = json.dumps(cleaned)              # safe JS string literal
    try:
        rate_val = max(0.5, min(2.0, float(rate)))
    except (TypeError, ValueError):
        rate_val = 1.0
    auto = "true" if autoplay and cleaned else "false"
    safe_label = json.dumps(str(label))

    template = """
<div class="jv-root">
  <style>
    .jv-root{font-family:'Inter',system-ui,sans-serif;}
    .jv-btn{display:inline-flex;align-items:center;gap:8px;background:#0d1117;
        border:1px solid #00d4ff;color:#00d4ff;border-radius:22px;
        padding:7px 16px;font-size:.82rem;cursor:pointer;transition:all .2s;}
    .jv-btn:hover{background:#00d4ff1a;}
    .jv-btn:disabled{border-color:#2a3240;color:#6b7685;cursor:default;}
    .jv-wave{display:inline-block;width:8px;height:8px;border-radius:50%;
        background:#00d4ff;animation:jvp 1s infinite;}
    @keyframes jvp{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.7);opacity:.4}}
  </style>
  <button class="jv-btn" id="jv-btn" __DISABLED__>
    <span class="jv-wave" id="jv-wave" style="display:none"></span>
    <span id="jv-label">🔊 __LABEL_TXT__</span>
  </button>
<script>
(function(){
  var TEXT = __PAYLOAD__;
  var btn = document.getElementById('jv-btn');
  var wave = document.getElementById('jv-wave');
  var lbl = document.getElementById('jv-label');
  function speak(){
    if(!TEXT || !('speechSynthesis' in window)){
      lbl.textContent = '🔇 voice not available'; return;
    }
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(TEXT);
    u.rate = __RATE__;
    u.onstart = function(){ wave.style.display='inline-block'; lbl.textContent='Speaking…'; };
    u.onend = function(){ wave.style.display='none'; lbl.textContent='🔊 ' + __LABEL__; };
    window.speechSynthesis.speak(u);
  }
  if(btn){ btn.addEventListener('click', speak); }
  if(__AUTO__){ setTimeout(speak, 300); }
})();
</script>
</div>
"""
    return (template
            .replace("__PAYLOAD__", payload)
            .replace("__RATE__", str(rate_val))
            .replace("__AUTO__", auto)
            .replace("__LABEL__", safe_label)
            .replace("__LABEL_TXT__", "Speak response" if cleaned else "Nothing to speak")
            .replace("__DISABLED__", "" if cleaned else "disabled"))
