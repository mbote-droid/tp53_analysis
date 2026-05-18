"""
============================================================
Voice-to-Text Transcription Module
============================================================
Lightweight speech recognition using Whisper API.
Optimized for clinical queries on TP53 analysis.

Usage:
    transcriber = VoiceTranscriber()
    text = transcriber.transcribe(audio_bytes)
    text = transcriber.transcribe_from_file("audio.wav")
============================================================
"""

import logging
import io
from typing import Optional, Tuple
from pathlib import Path

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

log = logging.getLogger(__name__)


class VoiceTranscriber:
    """Speech-to-text using OpenAI Whisper (local or API)."""

    def __init__(self, model: str = "base", use_api: bool = False):
        """
        Initialize voice transcriber.
        
        Args:
            model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            use_api: If True, uses Whisper API instead of local model
        """
        self.model_name = model
        self.use_api = use_api
        self.model = None
        
        if not use_api:
            if not WHISPER_AVAILABLE:
                raise ImportError("Install openai-whisper: pip install openai-whisper")
            log.info(f"Loading Whisper {model} model...")
            self.model = whisper.load_model(model)
        
        log.info(f"VoiceTranscriber initialized (API={use_api}, model={model})")

    def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Audio file bytes (WAV, MP3, etc.)
            language: Language code (e.g., 'en', 'es'). Auto-detect if None.
            
        Returns:
            Transcribed text
        """
        if self.use_api:
            return self._transcribe_api(audio_bytes)
        else:
            return self._transcribe_local(audio_bytes, language)

    def _transcribe_local(self, audio_bytes: bytes, language: Optional[str]) -> str:
        """Transcribe using local Whisper model."""
        try:
            # Convert bytes to temporary file
            audio_buffer = io.BytesIO(audio_bytes)
            
            # Whisper expects file-like object or path
            options = {"language": language} if language else {}
            result = self.model.transcribe(audio_buffer, **options)
            
            text = result.get("text", "").strip()
            log.info(f"Transcribed {len(audio_bytes)} bytes → {len(text)} chars")
            return text
        except Exception as e:
            log.error(f"Transcription error: {e}")
            raise

    def _transcribe_api(self, audio_bytes: bytes) -> str:
        """Transcribe using OpenAI Whisper API (requires API key)."""
        try:
            import openai
            audio_file = io.BytesIO(audio_bytes)
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            text = transcript.get("text", "").strip()
            log.info(f"API transcribed {len(audio_bytes)} bytes → {len(text)} chars")
            return text
        except Exception as e:
            log.error(f"API transcription error: {e}")
            raise

    def transcribe_from_file(self, filepath: str) -> str:
        """
        Transcribe audio file.
        
        Args:
            filepath: Path to audio file (WAV, MP3, M4A, etc.)
            
        Returns:
            Transcribed text
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        with open(filepath, "rb") as f:
            audio_bytes = f.read()
        
        return self.transcribe(audio_bytes)

    def transcribe_streamlit_audio(self):
        """
        Streamlit component for recording and transcribing audio.
        
        Returns:
            Transcribed text or None if no audio
        """
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Install streamlit: pip install streamlit")
        
        st.subheader("🎤 Voice Input")
        
        # Audio recorder
        audio_bytes = st.audio_input("Record your query:")
        
        if audio_bytes:
            st.info("Transcribing...")
            try:
                text = self.transcribe(audio_bytes.getvalue())
                st.success(f"✓ Transcribed: {text}")
                return text
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                return None
        
        return None


def transcribe_clinical_query(audio_bytes: bytes, model: str = "base") -> str:
    """
    Quick function to transcribe clinical voice query.
    
    Args:
        audio_bytes: Audio file bytes
        model: Whisper model size
        
    Returns:
        Transcribed text
    """
    transcriber = VoiceTranscriber(model=model)
    return transcriber.transcribe(audio_bytes, language="en")


if __name__ == "__main__":
    # Test: python -m utils.voice_transcriber
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m utils.voice_transcriber <audio_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    transcriber = VoiceTranscriber(model="base")
    text = transcriber.transcribe_from_file(filepath)
    print(f"\nTranscribed:\n{text}")
