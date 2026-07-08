"""
============================================================
Precision Onco Africa - Gemma Vision Agent
agents/gemma_vision.py
============================================================
Uses Gemma 4's native multimodal capability (image + text -> text, via the
Google GenAI backend) to read pathology slide images and photographed paper
lab reports directly — no separate CNN foundation model (UNI/ResNet, torch/
timm) and no separate OCR engine (Tesseract). One model, two capabilities.

This is deliberately a *narration* layer, not a diagnostic classifier: it
asks Gemma to describe what is visually present and extract legible text/
structure, then (for lab reports) cross-checks any mutation it read against
the curated ClinVar baseline before treating it as structured data. It never
claims a tissue-classification confidence score the way agents/
pathology_vision.py's CNN path does — that would overstate what a vision-
language model reading an image once can actually support.

Requires GOOGLE_API_KEY (same credential as the existing `api` inference
mode). If absent, every method returns a graceful {"success": False} rather
than raising — consistent with the rest of the platform's optional-dependency
pattern.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

from utils.logger import log

try:
    from config.settings import GOOGLE_API_KEY, GOOGLE_MODEL
except Exception:  # pragma: no cover
    GOOGLE_API_KEY, GOOGLE_MODEL = "", "gemma-4-26b-a4b-it"

PATHOLOGY_SYSTEM_PROMPT = (
    "You are a pathology assistant describing a microscopy image for a "
    "clinician. Describe tissue architecture, cellularity, and any features "
    "suggestive of malignancy (nuclear atypia, mitotic figures, necrosis, "
    "architectural distortion) in plain clinical language. Do not state a "
    "definitive diagnosis — describe what is visually present and flag "
    "features worth a pathologist's attention. If the image does not "
    "resemble a histology/pathology slide, say so plainly instead of "
    "inventing tissue findings."
)

LAB_REPORT_SYSTEM_PROMPT = (
    "You are transcribing a photographed paper laboratory/genomic report. "
    "Read the visible text exactly as printed — do not infer or complete "
    "values you cannot actually read. Extract: gene name, mutation/variant "
    "(HGVS or protein notation), variant allele frequency (VAF) if present, "
    "sample type, and the reporting lab/date if visible. If a field is not "
    "legible or not present, say so rather than guessing. Respond with a "
    "short structured summary, not a full transcript."
)

_HGVS_PATTERN = re.compile(r"\b([A-Z]\d{2,4}[A-Z*])\b")


class GemmaVisionAgent:
    """Gemma-4-vision-backed reader for pathology images and lab report
    photos. Stateless aside from the lazily-constructed backend client."""

    def __init__(self, api_key: str = GOOGLE_API_KEY, model: str = GOOGLE_MODEL):
        self._api_key = api_key
        self._model = model
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            from agents.rag_chain import GoogleGenAIBackend
            self._backend = GoogleGenAIBackend(api_key=self._api_key, model=self._model)
        return self._backend

    def health(self) -> bool:
        return bool(self._api_key)

    def read_pathology_slide(self, image_bytes: bytes, mime_type: str,
                             mutation_data: Optional[Dict] = None) -> Dict:
        """Narrate a pathology slide image with Gemma 4 vision. Never raises."""
        if not self._api_key:
            return {"success": False,
                    "error": "GOOGLE_API_KEY not configured — Gemma vision "
                             "requires the Google AI Studio backend."}
        try:
            prompt = "Describe this pathology image."
            if mutation_data and mutation_data.get("mutations"):
                muts = ", ".join(
                    m.get("amino_acid_change", "") for m in mutation_data["mutations"][:3]
                )
                prompt += (f" The patient's TP53 pipeline detected: {muts}. "
                          "If the visual findings are consistent with a "
                          "malignancy associated with these mutations, say so; "
                          "otherwise note that correlation is not visually "
                          "determinable from a single image.")
            narration = self._get_backend().generate_vision(
                system_prompt=PATHOLOGY_SYSTEM_PROMPT,
                image_bytes=image_bytes, mime_type=mime_type,
                user_prompt=prompt,
            )
            return {"success": True, "narration": narration,
                    "model": self._model, "method": "gemma_vision"}
        except Exception as e:
            log.error(f"Gemma pathology vision failed: {e}")
            return {"success": False, "error": str(e)}

    def read_lab_report_photo(self, image_bytes: bytes, mime_type: str) -> Dict:
        """Extract structured fields from a photographed paper lab report.
        Cross-checks any mutation Gemma reads against the curated ClinVar
        baseline so a misread doesn't silently become a 'confirmed' variant.
        Never raises."""
        if not self._api_key:
            return {"success": False,
                    "error": "GOOGLE_API_KEY not configured — Gemma vision "
                             "requires the Google AI Studio backend."}
        try:
            summary = self._get_backend().generate_vision(
                system_prompt=LAB_REPORT_SYSTEM_PROMPT,
                image_bytes=image_bytes, mime_type=mime_type,
                user_prompt="Transcribe and summarise the key fields on this report.",
            )
            candidate_mutations = sorted(set(_HGVS_PATTERN.findall(summary)))
            clinvar_notes = []
            if candidate_mutations:
                try:
                    from agents.clinvar_conflict_checker import check_conflicts
                    for mut in candidate_mutations[:5]:
                        res = check_conflicts(mutation=mut)
                        clinvar_notes.append({"mutation": mut,
                                              "verdict": res.get("verdict", "no_claims")})
                except Exception as e:  # pragma: no cover
                    log.warning(f"ClinVar cross-check unavailable: {e}")
            return {"success": True, "summary": summary,
                    "candidate_mutations": candidate_mutations,
                    "clinvar_cross_check": clinvar_notes,
                    "model": self._model, "method": "gemma_vision",
                    "caution": "Extracted by a vision-language model reading a "
                               "photograph — verify every field against the "
                               "original document before clinical use."}
        except Exception as e:
            log.error(f"Gemma lab-report vision failed: {e}")
            return {"success": False, "error": str(e)}
