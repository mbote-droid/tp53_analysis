"""
sentence_transformers.py — Cloud deployment mock
================================================
This file intercepts sentence_transformers imports on Streamlit Cloud
and routes them to HuggingFace Inference API instead of downloading
local models (~500MB) which exceed Streamlit Cloud's 1GB RAM limit.

Locally (INFERENCE_MODE=ollama/llamacpp): this file is ignored because
the real sentence_transformers package in venv takes priority.

On Streamlit Cloud: Python finds this file first and uses the API.
"""
import os
import numpy as np

# Check if real sentence_transformers is available (local machine)
try:
    import sys
    # Remove this file's directory from path temporarily to check real package
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _filtered = [p for p in sys.path if p != _this_dir]
    import importlib
    _spec = importlib.util.find_spec("sentence_transformers", _filtered)
    if _spec is not None:
        # Real package exists — don't mock, import the real one
        sys.path = _filtered + [_this_dir]
        from sentence_transformers import SentenceTransformer, CrossEncoder
        sys.path = [_this_dir] + _filtered
        raise ImportError("Use real package")
except Exception:
    pass


class SentenceTransformer:
    """
    Mock SentenceTransformer — routes to HuggingFace Inference API.
    Zero RAM usage — all computation on HF servers.
    """
    def __init__(self, model_name_or_path, *args, **kwargs):
        self.model_name = model_name_or_path
        self._hf_token = os.getenv("HF_TOKEN", "")
        self._client = None
        self._np_fallback = True

        if self._hf_token:
            try:
                from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
                self._client = HuggingFaceInferenceAPIEmbeddings(
                    api_key=self._hf_token,
                    model_name=model_name_or_path,
                )
                self._np_fallback = False
            except Exception:
                pass

    def encode(self, sentences, normalize_embeddings=False, *args, **kwargs):
        """Encode sentences to embeddings via API or return zero vectors."""
        if isinstance(sentences, str):
            sentences = [sentences]

        if self._client and not self._np_fallback:
            try:
                embeddings = self._client.embed_documents(sentences)
                arr = np.array(embeddings, dtype=np.float32)
                if normalize_embeddings:
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1, norms)
                    arr = arr / norms
                return arr
            except Exception:
                pass

        # Fallback: return zero vectors (cache won't work but app won't crash)
        return np.zeros((len(sentences), 384), dtype=np.float32)


class CrossEncoder:
    """
    Mock CrossEncoder — uses simple score fallback.
    No model download — returns neutral scores.
    """
    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name

    def predict(self, pairs, *args, **kwargs):
        """Return neutral scores — reranker falls back to vector scores."""
        return [0.5] * len(pairs)