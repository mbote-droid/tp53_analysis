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
  var played = false;
  // Pick the most natural English voice available in this browser/OS.
  // Edge/Windows expose "…(Natural)" voices; Chrome has "Google US English".
  function pickVoice(){
    var vs = (window.speechSynthesis.getVoices() || []).filter(function(v){
      return v.lang && v.lang.toLowerCase().indexOf('en') === 0; });
    if(!vs.length) return null;
    var prefs = [/natural/i, /google us english/i, /google uk english/i,
                 /aria/i, /jenny/i, /guy/i, /libby/i, /sonia/i,
                 /microsoft/i, /female/i];
    for(var p=0;p<prefs.length;p++){
      for(var i=0;i<vs.length;i++){ if(prefs[p].test(vs[i].name)) return vs[i]; }
    }
    return vs[0];
  }
  function speak(){
    if(!TEXT || !('speechSynthesis' in window)){
      lbl.textContent = '🔇 voice not available'; return;
    }
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(TEXT);
    var v = pickVoice(); if(v) u.voice = v;
    u.rate = __RATE__ * 0.97; u.pitch = 1.0;
    u.onstart = function(){ wave.style.display='inline-block'; lbl.textContent='Speaking…'; };
    u.onend = function(){ wave.style.display='none'; lbl.textContent='🔊 ' + __LABEL__; };
    window.speechSynthesis.speak(u);
  }
  function autoplay(){ if(__AUTO__ && !played){ played = true; speak(); } }
  if(btn){ btn.addEventListener('click', function(){ played = true; speak(); }); }
  // Voices load asynchronously — wait for them so we don't fall back to the
  // default robotic voice on the first utterance.
  if((window.speechSynthesis.getVoices() || []).length){ setTimeout(autoplay, 250); }
  else { window.speechSynthesis.onvoiceschanged = function(){ setTimeout(autoplay, 80); }; }
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
            .replace("__DISABLED__", "" if cleaned else "disabled")
            .replace("#00d4ff", "#8b7cf6"))   # Amethyst Nucleus palette


def speak_html_bargein(text: Optional[str], rate: float = 1.0) -> str:
    """Jarvis speaks the answer AND politely yields if interrupted — a human
    barge-in. Detection is 100% ON-DEVICE: it watches the microphone's audio
    energy (Web Audio API, getUserMedia + AnalyserNode) — NO transcription, NO
    third-party service. When you start talking, Jarvis cancels its speech, says
    "Go ahead…", and stops. Requires mic permission (the page is HTTPS).

    Returns non-empty HTML; degrades gracefully where the APIs are unavailable.
    """
    cleaned = _clean_for_speech(text) if is_speakable(text) else ""
    payload = json.dumps(cleaned)
    try:
        rate_val = max(0.5, min(2.0, float(rate)))
    except (TypeError, ValueError):
        rate_val = 1.0

    template = """
<div class="jvb-root">
  <style>
    .jvb-root{font-family:'Inter',system-ui,sans-serif;}
    .jvb-btn{display:inline-flex;align-items:center;gap:8px;background:#0b0e1a;
        border:1px solid #8b7cf6;color:#8b7cf6;border-radius:22px;
        padding:7px 16px;font-size:.82rem;cursor:pointer;}
    .jvb-btn:hover{background:#8b7cf61a;}
    .jvb-dot{display:inline-block;width:8px;height:8px;border-radius:50%;
        background:#8b7cf6;}
    .jvb-note{font-size:.72rem;color:#98a0bd;margin-top:5px;}
  </style>
  <button class="jvb-btn" id="jvb-btn">
    <span class="jvb-dot" id="jvb-dot"></span>
    <span id="jvb-lbl">🔊 Speak with barge-in</span>
  </button>
  <div class="jvb-note" id="jvb-note">On-device only — mic energy, no transcription.</div>
<script>
(function(){
  var TEXT = __PAYLOAD__;
  var btn=document.getElementById('jvb-btn');
  var lbl=document.getElementById('jvb-lbl');
  var note=document.getElementById('jvb-note');
  var audioCtx=null, stream=null, raf=null, speaking=false, floor=null, hot=0;
  function stopMic(){
    if(raf) cancelAnimationFrame(raf); raf=null;
    if(stream){ stream.getTracks().forEach(function(t){t.stop();}); stream=null; }
    if(audioCtx){ try{audioCtx.close();}catch(e){} audioCtx=null; }
  }
  function goAhead(){
    if(!speaking) return; speaking=false;
    window.speechSynthesis.cancel(); stopMic();
    lbl.textContent='👂 Go ahead…';
    var u=new SpeechSynthesisUtterance('Go ahead.'); u.rate=1.05;
    u.onend=function(){ lbl.textContent='🔊 Speak with barge-in'; };
    window.speechSynthesis.speak(u);
  }
  function watch(analyser, buf){
    analyser.getByteTimeDomainData(buf);
    var sum=0; for(var i=0;i<buf.length;i++){var d=(buf[i]-128)/128; sum+=d*d;}
    var rms=Math.sqrt(sum/buf.length);
    if(floor===null){ floor=rms; } else { floor=floor*0.95+rms*0.05; }
    // interruption = sustained energy clearly above the running noise floor
    if(rms > floor + 0.06 && rms > 0.05){ hot++; } else { hot=Math.max(0,hot-1); }
    if(hot>=3){ goAhead(); return; }
    if(speaking){ raf=requestAnimationFrame(function(){watch(analyser,buf);}); }
  }
  function start(){
    if(!('speechSynthesis' in window)){ lbl.textContent='🔇 no voice'; return; }
    window.speechSynthesis.cancel();
    var u=new SpeechSynthesisUtterance(TEXT); u.rate=__RATE__;
    u.onstart=function(){ speaking=true; lbl.textContent='🗣️ Speaking… (interrupt me)'; };
    u.onend=function(){ if(speaking){ speaking=false; stopMic();
        lbl.textContent='🔊 Speak with barge-in'; } };
    window.speechSynthesis.speak(u);
    // start listening for a barge-in (mic energy only)
    navigator.mediaDevices.getUserMedia({audio:true}).then(function(s){
      stream=s;
      audioCtx=new (window.AudioContext||window.webkitAudioContext)();
      var src=audioCtx.createMediaStreamSource(s);
      var analyser=audioCtx.createAnalyser(); analyser.fftSize=1024;
      src.connect(analyser);
      var buf=new Uint8Array(analyser.fftSize); floor=null; hot=0;
      raf=requestAnimationFrame(function(){watch(analyser,buf);});
    }).catch(function(){ note.textContent='Mic blocked — speaking without barge-in.'; });
  }
  if(!TEXT){ lbl.textContent='🔇 Nothing to speak'; btn.disabled=true; return; }
  btn.addEventListener('click', start);
})();
</script>
</div>
"""
    return (template.replace("__PAYLOAD__", payload)
            .replace("__RATE__", str(rate_val)))
