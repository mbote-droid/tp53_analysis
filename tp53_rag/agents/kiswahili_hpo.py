"""
============================================================
Precision Onco Africa - Kiswahili → HPO / ICD-10 Alignment
agents/kiswahili_hpo.py
============================================================
Maps colloquial Kiswahili symptom / phenotype descriptions to standardised
clinical codes — Human Phenotype Ontology (HPO) and ICD-10 — so a clinician who
enters an observation in Kiswahili gets it anchored to machine-readable clinical
terms BEFORE anything reaches the knowledge base. This upgrades the equity
layer from "translate the words" to "map the meaning to the ontology".

Honest by construction:
  * The mapping table is a small, curated STARTER set — clearly flagged for
    clinical review, not presented as an exhaustive validated ontology.
  * Primary matching is exact substring lookup (deterministic).
  * The optional embedding fallback carries a CONFIDENCE GATE: below the
    threshold it returns "no confident mapping" and passes the raw text through
    rather than forcing a wrong-but-confident code.

No model internals; pure Python + an injectable embedder. Never raises.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

MAPPING_DISCLAIMER = ("Curated Kiswahili→HPO/ICD-10 starter set for decision "
                      "support — requires clinical review; not an exhaustive or "
                      "validated ontology. Research use only.")

DEFAULT_THRESHOLD = 0.60

# Curated starter table. term (Kiswahili) → English, HPO id, ICD-10.
# Codes chosen from well-established HPO/ICD-10 symptom terms; review before
# clinical use.
KISWAHILI_HPO: Dict[str, Dict[str, str]] = {
    "maumivu ya tumbo": {"en": "abdominal pain", "hpo": "HP:0002027", "icd10": "R10.9"},
    "maumivu ya kichwa": {"en": "headache", "hpo": "HP:0002315", "icd10": "R51"},
    "maumivu": {"en": "pain", "hpo": "HP:0012531", "icd10": "R52"},
    "homa": {"en": "fever", "hpo": "HP:0001945", "icd10": "R50.9"},
    "uchovu": {"en": "fatigue", "hpo": "HP:0012378", "icd10": "R53.83"},
    "kupungua uzito": {"en": "weight loss", "hpo": "HP:0001824", "icd10": "R63.4"},
    "kukohoa": {"en": "cough", "hpo": "HP:0012735", "icd10": "R05"},
    "kutapika": {"en": "vomiting", "hpo": "HP:0002013", "icd10": "R11.10"},
    "kichefuchefu": {"en": "nausea", "hpo": "HP:0002018", "icd10": "R11.0"},
    "kukosa hamu ya kula": {"en": "loss of appetite (anorexia)",
                            "hpo": "HP:0002039", "icd10": "R63.0"},
    "manjano": {"en": "jaundice", "hpo": "HP:0000952", "icd10": "R17"},
    "kutokwa na damu": {"en": "abnormal bleeding", "hpo": "HP:0001892", "icd10": "R58"},
    "upungufu wa damu": {"en": "anaemia", "hpo": "HP:0001903", "icd10": "D64.9"},
    "kupumua kwa shida": {"en": "dyspnoea", "hpo": "HP:0002094", "icd10": "R06.0"},
    "kuharisha": {"en": "diarrhoea", "hpo": "HP:0002014", "icd10": "R19.7"},
    "kuvimbiwa": {"en": "constipation", "hpo": "HP:0002019", "icd10": "K59.00"},
    "uvimbe wa tezi": {"en": "lymphadenopathy", "hpo": "HP:0002716", "icd10": "R59.9"},
    "uvimbe": {"en": "swelling / oedema", "hpo": "HP:0000969", "icd10": "R60.9"},
}

# Surface-form aliases → canonical key. Clinicians type conjugated verbs
# ("ana kohoa" = has a cough) and noun forms ("kikohozi" = a cough), not the
# dictionary infinitive ("kukohoa" = to cough). Map those real forms too.
_ALIASES: Dict[str, str] = {
    # cough
    "kikohozi": "kukohoa", "anakohoa": "kukohoa", "kohoa": "kukohoa",
    # vomiting
    "anatapika": "kutapika", "tapika": "kutapika", "kutapika": "kutapika",
    # diarrhoea
    "anaharisha": "kuharisha", "kuhara": "kuharisha", "anahara": "kuharisha",
    "harisha": "kuharisha",
    # fatigue
    "amechoka": "uchovu", "kuchoka": "uchovu",
    # weight loss
    "amepungua uzito": "kupungua uzito", "kupungua kwa uzito": "kupungua uzito",
    # constipation
    "amevimbiwa": "kuvimbiwa",
    # bleeding
    "anatokwa na damu": "kutokwa na damu", "kuvuja damu": "kutokwa na damu",
    # dyspnoea
    "kushindwa kupumua": "kupumua kwa shida",
    "anashindwa kupumua": "kupumua kwa shida", "pumzi fupi": "kupumua kwa shida",
    # loss of appetite
    "hana hamu ya kula": "kukosa hamu ya kula",
    "amekosa hamu ya kula": "kukosa hamu ya kula",
    # nausea
    "anahisi kichefuchefu": "kichefuchefu",
    # jaundice
    "rangi ya manjano": "manjano",
}

# Full lookup = canonical entries + aliases pointing at the same codes.
_LOOKUP: Dict[str, Dict[str, str]] = dict(KISWAHILI_HPO)
for _alias, _canon in _ALIASES.items():
    if _canon in KISWAHILI_HPO:
        _LOOKUP[_alias] = KISWAHILI_HPO[_canon]

# Longest phrases first so "maumivu ya tumbo" wins over "maumivu".
_TERMS_BY_LEN = sorted(_LOOKUP.keys(), key=len, reverse=True)


def _normalise(s: str) -> str:
    return " ".join((s or "").lower().split())


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def map_text(text: str, embed_fn: Optional[Callable[[str], List[float]]] = None,
             threshold: float = DEFAULT_THRESHOLD) -> Dict:
    """Map Kiswahili observation text to HPO/ICD-10 codes.

    Exact substring matches first (deterministic). If an embedder is supplied,
    an additional best-effort fuzzy match is attempted for the WHOLE text
    against each term, but only accepted at/above `threshold` — otherwise it is
    reported as an unconfident suggestion, never silently applied. Never raises.
    """
    norm = _normalise(text)
    if not norm:
        return {"mappings": [], "unconfident": [], "matched": False,
                "note": "Empty input."}

    mappings: List[Dict] = []
    matched_terms = set()
    consumed = norm
    for term in _TERMS_BY_LEN:
        if term in consumed:
            e = _LOOKUP[term]
            mappings.append({"kiswahili": term, "english": e["en"],
                             "hpo": e["hpo"], "icd10": e["icd10"],
                             "match": "exact"})
            matched_terms.add(term)
            consumed = consumed.replace(term, " ")  # don't double-count overlaps

    unconfident: List[Dict] = []
    if embed_fn is not None and not mappings:
        try:
            tvec = embed_fn(norm)
            best_term, best_sim = None, -1.0
            for term, e in _LOOKUP.items():
                sim = _cosine(tvec, embed_fn(term))
                if sim > best_sim:
                    best_term, best_sim = term, sim
            if best_term is not None:
                e = _LOOKUP[best_term]
                entry = {"kiswahili": best_term, "english": e["en"],
                         "hpo": e["hpo"], "icd10": e["icd10"],
                         "similarity": round(best_sim, 2)}
                if best_sim >= threshold:
                    entry["match"] = "fuzzy"
                    mappings.append(entry)
                else:
                    entry["match"] = "below_threshold"
                    unconfident.append(entry)
        except Exception:
            pass

    return {"mappings": mappings, "unconfident": unconfident,
            "matched": bool(mappings),
            "note": (MAPPING_DISCLAIMER if mappings else
                     "No confident mapping — passing the raw text through."),
            "disclaimer": MAPPING_DISCLAIMER}


def to_clinical_terms(text: str, **kw) -> str:
    """Convenience: a compact 'english (HPO/ICD-10)' string for prompt
    enrichment, or the original text if nothing mapped."""
    res = map_text(text, **kw)
    if not res["mappings"]:
        return text
    parts = [f"{m['english']} ({m['hpo']}, ICD-10 {m['icd10']})"
             for m in res["mappings"]]
    return f"{text}  [clinical terms: {'; '.join(parts)}]"
