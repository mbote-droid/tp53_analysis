"""
Precision Onco Africa — Streamlit Web App
Clean rebuild — no streamlit_option_menu dependency
"""
import os
import sys
import json
import time
import logging
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.viz import (
    CANONICAL_HOTSPOTS,
    P53_DOMAINS,
    agent_status_badge,
    animated_vaf_timeline,
    animated_hotspot_bar,
    agent_architecture_diagram,
    build_agent_graph_data,
    agent_graph_3d_html,
    variant_annotation_table,
    variant_effect_gauge,
    alphafold_viewer_html,
    plddt_profile_chart,
    domain_legend_chart,
    parse_residues,
    protein_viewer_html,
    dock_candidates,
    docking_affinity_chart,
    docking_pose_html,
    tnm_stage_bar,
    pathogenicity_gauge,
    tme_donut,
    vaf_gauge,
    pathway_diverging_bar,
    african_atlas_map,
    african_burden_bar,
    clinvar_conflict_chart,
    chembl_phase_chart,
    trials_priority_chart,
    ind_section_chart,
    synthetic_lethal_network,
    docking_affinity_gauge,
    structural_profile_radar,
)
from utils.variant_annotation import annotate_variant
from utils.variant_effect import predict_effect
from utils.alphafold_client import get_tp53_structure
from utils.export_disclaimer import stamp_markdown, stamp_json, stamp_fhir

st.set_page_config(
    page_title="Precision Onco Africa",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    """Apply the dark bioinformatics theme + web fonts.

    Fonts load from Google Fonts when online; offline/edge deployments
    fall back to the system stacks so nothing breaks without a network.
    Pure CSS — no user input, no runtime cost, no memory footprint.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&family=JetBrains+Mono:wght@400;700&display=swap');

        :root {
            --tp53-accent: #00d4ff;
            --tp53-bg: #0d1117;
            --tp53-panel: #161b22;
            --tp53-text: #e6edf3;
            --tp53-sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --tp53-mono: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace;
        }

        html, body, [class*="css"], .stMarkdown, .stTextInput, .stSelectbox,
        button, input, textarea, .stTabs [data-baseweb="tab"] {
            font-family: var(--tp53-sans);
        }

        code, pre, kbd, samp, .stCode, [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {
            font-family: var(--tp53-mono);
        }

        h1, h2, h3, h4 {
            font-family: var(--tp53-sans);
            letter-spacing: -0.01em;
        }

        /* Accent the active tab + primary buttons */
        .stTabs [aria-selected="true"] {
            color: var(--tp53-accent);
            border-bottom-color: var(--tp53-accent);
        }
        .stButton > button[kind="primary"] {
            background: var(--tp53-accent);
            color: #03121a;
            border: none;
        }

        /* ── Real-time agent status badges ── */
        .tp53-badge {
            display: flex; align-items: center; gap: 10px;
            font-family: var(--tp53-mono); font-size: 0.86rem;
            padding: 8px 12px; margin: 6px 0; border-radius: 8px;
            background: var(--tp53-panel);
            border: 1px solid #232b36;
        }
        .tp53-badge .dot {
            width: 10px; height: 10px; border-radius: 50%;
            flex: 0 0 auto;
        }
        .tp53-badge .name { flex: 1 1 auto; color: var(--tp53-text); }
        .tp53-badge .time { color: #8b98a5; font-size: 0.78rem; }

        .tp53-badge.running { border-color: var(--tp53-accent); }
        .tp53-badge.running .dot {
            background: var(--tp53-accent);
            animation: tp53-spin 0.9s linear infinite;
            box-shadow: 0 0 0 0 rgba(0,212,255,0.6);
        }
        .tp53-badge.running .name::after {
            content: ' …'; color: var(--tp53-accent);
        }
        .tp53-badge.complete .dot { background: #2ecc71; }
        .tp53-badge.complete { border-color: #1e3a2a; }
        .tp53-badge.failed   .dot { background: #ff4b4b; }
        .tp53-badge.failed   { border-color: #3a1e1e; }

        @keyframes tp53-spin {
            0%   { box-shadow: 0 0 0 0 rgba(0,212,255,0.55); }
            70%  { box-shadow: 0 0 0 7px rgba(0,212,255,0); }
            100% { box-shadow: 0 0 0 0 rgba(0,212,255,0); }
        }

        /* ── Tabs: never let a tab row overflow — scroll horizontally ── */
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            scrollbar-width: thin;
        }
        .stTabs [data-baseweb="tab"] { white-space: nowrap; }

        /* ── Two-level tab hierarchy: 13 flat tabs are grouped into 6
           top-level sections, each with its own inner tab row. Give the
           outer (group) row a bolder, pill-styled look so it reads as
           "section" while nested tab rows stay compact — otherwise two
           identical-looking tab rows stacked on top of each other are
           genuinely hard to parse. Selector verified against the live
           rendered DOM (Streamlit 1.55): the outer tabs' stTabs container
           is the sole direct grandchild of stMainBlockContainer via
           stVerticalBlock; every nested/inner stTabs sits several levels
           deeper inside a tab panel, so this selector matches only the
           outer row. ── */
        div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"]
            > div[data-testid="stTabs"] > div > div > div[data-baseweb="tab-list"] {
            gap: 4px;
            background: var(--tp53-panel);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid #232b36;
        }
        div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"]
            > div[data-testid="stTabs"] > div > div > div[data-baseweb="tab-list"] button[data-baseweb="tab"] {
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            padding: 8px 16px;
        }
        div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"]
            > div[data-testid="stTabs"] > div > div > div[data-baseweb="tab-list"] button[aria-selected="true"] {
            background: rgba(0, 212, 255, 0.12);
        }

        /* Charts and tables stay inside the viewport on any width */
        [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
            max-width: 100%;
            overflow-x: auto;
        }

        /* ── Mobile / small screens (phones, narrow windows) ── */
        @media (max-width: 768px) {
            /* reclaim the wasted side padding */
            .block-container {
                padding: 1rem 0.7rem 2rem 0.7rem !important;
            }
            h1 { font-size: 1.5rem !important; }
            h2 { font-size: 1.2rem !important; }
            h3 { font-size: 1.05rem !important; }
            /* tighter tab labels so more fit before scrolling */
            .stTabs [data-baseweb="tab"] {
                padding: 6px 8px;
                font-size: 0.82rem;
            }
            /* status badges wrap gracefully */
            .tp53-badge { font-size: 0.78rem; padding: 6px 9px; }
            /* keep metrics from overflowing */
            [data-testid="stMetricValue"] { font-size: 1.1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Honest compute-backend probe (logs ROCm/CUDA/CPU as actually present) ──
try:
    from utils.hardware_probe import log_compute_banner
    log_compute_banner(log)
except Exception:
    pass

# ── Import RAG modules ────────────────────────────────────────────
RAG_AVAILABLE = False
try:
    from agents.rag_chain import TP53RAGChain
    from agents.dispatcher import AgentDispatcher
    from knowledge_base.vector_store import TP53VectorStore
    RAG_AVAILABLE = True
except Exception as e:  # surface the REAL chained cause, not just the final message
    import traceback as _tb
    _details = _tb.format_exc()
    st.error(f"RAG modules not found: {e}. Run: python main.py build")
    with st.expander("🔍 Import error details (for debugging)"):
        st.code(_details, language="text")
    log.error("RAG import failed:\n%s", _details)

# ── Session state ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing knowledge base…")
def init_rag_system():
    if not RAG_AVAILABLE:
        return None, None
    try:
        store = TP53VectorStore()
        if not store.is_built():
            # Auto-build from the in-code curated knowledge so the app is
            # self-sufficient on cloud (no pre-built DB / no `main.py build`
            # needed). The documents ship in the package; only the embeddings
            # need the network, which is already configured (HF API on cloud).
            try:
                from knowledge_base.ingestion import TP53DocumentIngester
                ingester = TP53DocumentIngester()
                docs = ingester.load_curated_knowledge()
                chunks = ingester.chunk_documents(docs)
                if not chunks:
                    log.error("Auto-build skipped: no curated documents found.")
                    return None, None
                store.build(chunks)
                log.info(f"Auto-built knowledge base with {len(chunks)} chunks.")
            except Exception as be:
                log.error(f"Knowledge-base auto-build failed: {be}")
                return None, None
        store.load()
        rag = TP53RAGChain(vector_store=store)
        return rag, store
    except Exception as e:
        log.error(f"RAG init failed: {e}")
        return None, None

# FIXED: wrapped in try/except so torch/torchvision failures on cloud don't crash the app
@st.cache_resource
def load_pathology_agent():
    try:
        from agents.pathology_vision import PathologyVisionAgent
        return PathologyVisionAgent(
            rag_chain=st.session_state.get("rag")
        )
    except ImportError as e:
        log.warning(f"Pathology agent unavailable: {e}")
        return None
    except Exception as e:
        log.warning(f"Pathology agent failed to load: {e}")
        return None


@st.cache_resource
def load_gemma_vision_agent():
    """Gemma-4 native multimodal reader (no torch/timm, no OCR). Needs only
    GOOGLE_API_KEY. Returns None gracefully if unavailable so the app never
    crashes on this optional capability."""
    try:
        from agents.gemma_vision import GemmaVisionAgent
        agent = GemmaVisionAgent()
        return agent if agent.health() else None
    except Exception as e:
        log.warning(f"Gemma vision agent unavailable: {e}")
        return None


# Bundled, clearly-watermarked SYNTHETIC demo images for the one-click path.
SAMPLE_SLIDE_PATH = Path(__file__).parent / "assets" / "sample_pathology_slide.jpg"
SAMPLE_REPORT_PATH = Path(__file__).parent / "assets" / "sample_lab_report.jpg"
SAMPLE_SANGER_PATH = Path(__file__).parent / "assets" / "sample_sanger.ab1"
SAMPLE_SANGER_REF_PATH = Path(__file__).parent / "assets" / "sample_sanger_reference.txt"

if "rag" not in st.session_state:
    st.session_state.rag, st.session_state.store = init_rag_system()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_data" not in st.session_state:
    st.session_state.pipeline_data = {}
if "tnm_result" not in st.session_state:
    st.session_state.tnm_result = {}
if "last_pathology_result" not in st.session_state:
    st.session_state.last_pathology_result = {}
if "session_id" not in st.session_state:
    # Isolated per browser-session by default (no cross-user leakage on a shared
    # deployment). Set MEMORY_SESSION_ID locally for persistent single-user
    # memory that survives restarts ("don't start from zero").
    import os as _os, uuid as _uuid
    st.session_state.session_id = _os.getenv("MEMORY_SESSION_ID") or _uuid.uuid4().hex[:12]
if "memory" not in st.session_state:
    try:
        from utils.memory import get_memory
        st.session_state.memory = get_memory()
    except Exception as _e:
        log.warning(f"Conversation memory unavailable: {_e}")
        st.session_state.memory = None

# ── Helper functions ──────────────────────────────────────────────
import threading
_inference_lock = threading.Semaphore(2)
def safe_query(question: str, agent_type=None) -> dict:
    if not st.session_state.rag:
        return {
            "answer": "The analysis engine is starting up or offline. Please try "
                      "again in a moment.",
            "agent_used": "offline",
            "sources": [],
            "cache_hit": False,
            "retries": 0,
        }
    # Security: neutralise role-tag injection and cap length before the LLM.
    try:
        from utils.security import sanitize_for_prompt
        question = sanitize_for_prompt(question)
    except Exception:
        pass
    mem = st.session_state.get("memory")
    sid = st.session_state.get("session_id", "default")
    # Pull recent (PII-scrubbed) turns so the model has continuity across reruns
    # and restarts — the conversation doesn't start from zero.
    history = mem.history_strings(sid, limit=6) if mem else None
    try:
        # Cap concurrent inferences so parallel sessions can't exhaust the
        # 8GB RAM budget. Acquire/release is handled by the context manager.
        with _inference_lock:
            result = st.session_state.rag.query(
                question=question,
                pipeline_data=st.session_state.pipeline_data or None,
                agent_type=agent_type,
                conversation_history=history,
            )
        if mem:
            mem.remember(sid, question, result.get("answer", ""),
                         result.get("agent_used", ""))
        # Token-efficient router: record how this query was (or could be) served
        # so the platform can show measured token/cost savings over a session.
        try:
            from utils.token_router import get_router
            get_router().route(question, cache_hit=bool(result.get("cache_hit")))
        except Exception:
            pass
        return result
    except Exception as e:
        log.warning(f"safe_query failed: {e}")
        return {
            "answer": "Something went wrong while analysing that. Please "
                      "rephrase your question or try again in a moment.",
            "agent_used": agent_type or "error",
            "sources": [],
            "cache_hit": False,
            "retries": 0,
        }


# Cached builders for expensive, deterministic visuals. Streamlit reruns the
# whole script on every interaction; without caching the DNA code-graph would
# re-walk the filesystem and AST-parse every module on each rerender. Cached
# per-session keeps the demo snappy without changing outputs.
@st.cache_data(show_spinner=False)
def _cached_codegraph():
    from utils.codegraph import build_helix_codegraph
    return build_helix_codegraph()


@st.cache_data(show_spinner=False)
def _cached_command_center():
    from agents.command_center import command_center_snapshot
    return command_center_snapshot()


def format_sources(sources: list) -> str:
    if not sources:
        return "*No sources retrieved*"
    lines = []
    for i, src in enumerate(sources[:5], 1):
        lines.append(
            f"**{i}.** [{src.get('category','?')}] "
            f"{src.get('source','?')} "
            f"(score: {src.get('relevance_score', 0):.2f})"
        )
    return "\n".join(lines)


def render_clinvar_safety(answer_text: str, key_prefix: str = "") -> None:
    """🛡 Shared ClinVar hallucination-guard panel — vets AI text against
    ClinVar and renders verdict + concordance chart. Used by the Query and
    Analysis tabs (single source of truth, no copy-paste)."""
    try:
        from agents.clinvar_conflict_checker import ClinVarConflictChecker
        chk = ClinVarConflictChecker().check(text=answer_text or "")
        findings = chk["findings"]
        if not findings:
            return  # nothing to verify — stay quiet
        if chk["conflicts_found"]:
            st.error(f"🛡 ClinVar Safety Check — {chk['message']}")
        elif chk["verdict"] == "concordant":
            st.success(f"🛡 ClinVar Safety Check — {chk['message']}")
        else:
            st.info(f"🛡 ClinVar Safety Check — {chk['message']}")
        with st.expander("ClinVar concordance details",
                         expanded=chk["conflicts_found"] > 0):
            st.plotly_chart(clinvar_conflict_chart(findings),
                            width="stretch",
                            key=f"clinvar_chart_{key_prefix}")
            for f in findings:
                icon = {"high": "🔴", "medium": "🟠"}.get(f["severity"], "🟢")
                st.markdown(
                    f"{icon} **{f['mutation']}** — AI: *{f['ai_classification']}* · "
                    f"ClinVar: *{f['clinvar_classification']}* "
                    f"([verify]({f['evidence_url']}))"
                )
            st.caption("Cross-checked against a curated ClinVar reference. "
                       "Conflicts flag possible AI hallucination — verify before clinical use.")
    except Exception as e:
        log.warning(f"ClinVar safety check skipped: {e}")


# ── Voice transcription (shared by Tab 6 narration + Tab 7) ────────
@st.cache_resource
def load_whisper():
    """Lazy-load the tiny Whisper model once. Returns None if unavailable."""
    try:
        import whisper  # type: ignore[import-not-found]  # optional dep, not in requirements
        return whisper.load_model("tiny")
    except Exception as e:
        log.error(f"Failed to load Whisper: {e}")
        return None


def transcribe(audio_bytes) -> str:
    """Transcribe recorded audio to text. Always returns a string; on any
    failure returns a message prefixed 'Transcription error'.

    Uses a NamedTemporaryFile so it never writes into the (read-only on
    Streamlit Cloud) app directory.
    """
    model = load_whisper()
    if model is None:
        return "Transcription error: Whisper model failed to load"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            if hasattr(audio_bytes, "getvalue"):
                tmp.write(audio_bytes.getvalue())
            elif hasattr(audio_bytes, "read"):
                if hasattr(audio_bytes, "seek"):
                    audio_bytes.seek(0)
                tmp.write(audio_bytes.read())
            else:
                tmp.write(bytes(audio_bytes))
            tmp_path = Path(tmp.name)

        try:
            result = model.transcribe(str(tmp_path), language="en", fp16=False)
        except TypeError:
            result = model.transcribe(str(tmp_path), language="en")
        return result.get("text", "").strip()
    except Exception as e:
        return f"Transcription error: {str(e)[:200]}"
    finally:
        try:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
        except Exception as cleanup_err:
            log.warning(f"Failed to cleanup temp audio: {cleanup_err}")


def whisper_available() -> bool:
    try:
        import whisper  # type: ignore[import-not-found]  # noqa: F401  (optional dep)
        return True
    except ImportError:
        return False


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Precision Onco Africa")
    st.markdown("**Multi-agent oncology bioinformatics**")
    st.divider()

    if st.session_state.rag:
        st.success("✅ RAG system ready")
        try:
            stats = st.session_state.rag.cache_stats()
            st.caption(f"Cache: {stats.get('hits',0)} hits, {stats.get('misses',0)} misses")
        except Exception:
            pass
    else:
        st.error("❌ RAG offline — run: python main.py build")

    st.divider()
    agents = [
        "mutation_analysis", "drug_discovery", "clinical_interpretation",
        "liquid_biopsy", "gene_expression", "orf_analysis",
        "phylogenetic_analysis", "domain_annotation",
    ]
    selected_agent = st.selectbox(
        "Force agent type:",
        ["auto-detect"] + agents,
    )
    forced_agent = None if selected_agent == "auto-detect" else selected_agent

    st.divider()
    _mode = os.getenv("INFERENCE_MODE", "ollama").lower()
    if _mode == "api":
        _backend = "Google AI Studio API (cloud)"
        _model = os.getenv("GOOGLE_MODEL", "gemma-4-26b-a4b-it")
        _privacy = "Cloud inference — API mode"
    elif _mode == "llamacpp":
        _backend = "llama.cpp CPU (local)"
        _model = "Gemma 4 (GGUF, Q4_K_M)"
        _privacy = "100% local — no cloud"
    elif _mode == "fireworks":
        _backend = "Fireworks AI on AMD Instinct (cloud)"
        _model = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/minimax-m3")
        _privacy = "Cloud inference — AMD-hosted"
    else:
        _backend = "Ollama (local)"
        _model = os.getenv("OLLAMA_MODEL", "gemma4-lowmem")
        _privacy = "100% local — no cloud"
    st.markdown(f"""
**Model:** {_model}
**Backend:** {_backend}
**Mode:** `INFERENCE_MODE={_mode}`
**Privacy:** {_privacy}
""")

# ── Persistent Research-Use-Only banner (shown above every tab) ───
st.markdown(
    """
    <div style="background:#fff6e0;border-left:4px solid #e0a200;
         padding:8px 14px;border-radius:6px;margin:0 0 10px 0;
         font-size:0.85rem;color:#5a4500;line-height:1.4;">
      <b>⚠️ Research Use Only.</b> Not a diagnostic device and not for clinical
      decisions. All outputs are informational — confirm with a CLIA-certified
      laboratory and a qualified clinician. <b>Do not enter real, identifiable
      patient data.</b>
      <a href="https://github.com/mbote-droid/tp53_analysis/blob/main/tp53_rag/DISCLAIMER.md"
         target="_blank" style="color:#9a6b00;">Full disclaimer&nbsp;↗</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────
# 13 flat tabs used to overflow the tab bar (forcing horizontal scroll to
# find anything past ~7). Grouped into 6 logical top-level tabs instead;
# each group holds an inner st.tabs() for its members. Streamlit tab/
# container objects are position-independent once created, so every
# `with tabN:` block below is completely unchanged — only this definition
# block changed.
group_analyze, group_board, group_molecular, group_reports, group_global, \
    group_tools = st.tabs([
        "🔍 Analyze",
        "⭐ Tumour Board",
        "🧬 Molecular",
        "📊 Reports & Staging",
        "🌍 Global & Trials",
        "🎤 Voice & Tools",
    ])

with group_analyze:
    tab1, tab2 = st.tabs(["🔍 Query", "🧬 Analysis"])

with group_board:
    tab13 = st.container()

with group_molecular:
    tab3, tab6, tab9 = st.tabs(["💊 Drug Discovery", "🔬 Structure", "🔬 Pathology"])

with group_reports:
    tab4, tab5, tab10 = st.tabs(["📊 Visualization", "📋 Report", "📍 TNM Staging"])

with group_global:
    tab11, tab12 = st.tabs(["🌍 African Atlas", "🧪 Clinical Trials"])

with group_tools:
    tab7, tab8 = st.tabs(["🎤 Voice", "🛠 Debug"])

# ── TAB 1: Query ──────────────────────────────────────────────────
with tab1:
    st.markdown("## 🔍 TP53 Knowledge Query")
    st.markdown("Ask any question about TP53 mutations, drug targets, or clinical significance.")

    question = st.text_area(
        "Your question:",
        placeholder="e.g., What are the clinical implications of R175H mutation?",
        height=100,
    )

    # ── 🌍 Kiswahili → clinical codes (equity: map meaning to the ontology) ──
    with st.expander("🌍 Kiswahili symptom → HPO / ICD-10 codes"):
        st.caption("Enter a clinical observation in Kiswahili; it is mapped to "
                   "standard HPO / ICD-10 codes (with a confidence gate) before "
                   "querying — so the equity layer anchors *meaning*, not just "
                   "words.")
        sw_text = st.text_input("Observation (Kiswahili):",
                                placeholder="e.g., mgonjwa ana homa na maumivu ya tumbo",
                                key="sw_input")
        if sw_text:
            from agents.kiswahili_hpo import map_text
            _swres = map_text(sw_text)
            if _swres["mappings"]:
                st.dataframe(pd.DataFrame([
                    {"Kiswahili": m["kiswahili"], "Clinical term": m["english"],
                     "HPO": m["hpo"], "ICD-10": m["icd10"]}
                    for m in _swres["mappings"]]),
                    width="stretch", hide_index=True)
                st.caption(f"⚠️ {_swres['disclaimer']}")
            else:
                st.info(_swres["note"])

    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("🚀 Ask Gemma 4", width="stretch")
    with col2:
        if st.button("🗑 Clear", width="stretch"):
            st.session_state.messages = []
            st.success("Cleared")
    stream_mode = st.checkbox(
        "⚡ Stream the answer (lower perceived latency)", value=True,
        help="Shows tokens as they are generated. Uncheck for the full "
             "self-correcting pipeline (slightly slower, validated).")
    measure_certainty = st.checkbox(
        "🎯 Measure epistemic certainty (samples the model 3×, slower)",
        value=False,
        help="Re-asks the model several times and measures how much the "
             "answers agree — an honest confidence signal. Runs on the "
             "non-streaming path.")

    if submit and question and stream_mode and st.session_state.get("rag"):
        # Streaming path: tokens appear live via st.write_stream. Single-pass
        # (no self-correction); the non-stream path below is the validated one.
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown("### Answer")
        try:
            from utils.security import sanitize_for_prompt
            safe_q = sanitize_for_prompt(question)
            mem = st.session_state.get("memory")
            sid = st.session_state.get("session_id", "default")
            history = mem.history_strings(sid, limit=6) if mem else None
            with _inference_lock:
                streamed = st.write_stream(
                    st.session_state.rag.query_stream(
                        safe_q, pipeline_data=st.session_state.pipeline_data or None,
                        agent_type=forced_agent, conversation_history=history))
            answer_text = streamed if isinstance(streamed, str) else "".join(streamed)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer_text})
            if mem:
                mem.remember(sid, question, answer_text, forced_agent or "")
            try:
                from utils.token_router import get_router
                get_router().route(question, cache_hit=False)
            except Exception:
                pass
            # Guardrails on the streamed answer
            try:
                from utils.guardrails import run_guardrails
                from utils.viz import guardrails_html
                components.html(guardrails_html(run_guardrails(answer_text)),
                                height=200, scrolling=False)
            except Exception:
                pass
            render_clinvar_safety(answer_text, key_prefix="query_stream")
        except Exception as e:
            st.error("Streaming hit a snag — try unchecking stream mode. "
                     f"({str(e)[:100]})")

    elif submit and question:
        with st.spinner("Querying Gemma 4 via RAG..."):
            result = safe_query(question, agent_type=forced_agent)

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

        st.markdown("### Answer")
        st.markdown(result["answer"])

        # 🎯 Epistemic certainty — sample the model N× and measure agreement
        if measure_certainty:
            st.markdown("#### 🎯 Epistemic certainty")
            try:
                from agents.uncertainty import sample_and_measure
                from agents.rag_chain import _build_backend
                _bk = _build_backend()

                def _gen(system, user):
                    return _bk.generate(system, user, max_tokens=512,
                                        temperature=0.7)

                _sys = ("You are a concise TP53 clinical-genomics assistant. "
                        "Answer accurately and briefly.")
                with st.spinner("Sampling the model 3× to gauge agreement…"):
                    unc = sample_and_measure(_sys, question, _gen, n=3)
                if unc.get("success") and unc.get("uncertainty") is not None:
                    _b = unc["band"]
                    _color = {"green": "🟢", "amber": "🟠",
                              "red": "🔴"}.get(_b, "⚪")
                    cc1, cc2 = st.columns(2)
                    cc1.metric("Epistemic Uncertainty Index",
                               f"{unc['uncertainty']*100:.0f}%")
                    cc2.metric("Model agreement",
                               f"{_color} {unc['agreement']*100:.0f}%")
                    if _b == "green":
                        st.success("High certainty — the model's repeated "
                                   "answers agree closely.")
                    elif _b == "amber":
                        st.warning("Moderate uncertainty — answers vary; treat "
                                   "specifics with caution.")
                    else:
                        st.error("High uncertainty — the model's answers "
                                 "diverge. Verify independently.")
                    st.caption(f"⚠️ {unc.get('disclaimer', '')}")
                else:
                    st.caption("Certainty unavailable: "
                               f"{unc.get('reason', 'insufficient samples')[:100]}")
            except Exception as e:
                st.caption(f"Certainty check unavailable: {str(e)[:100]}")

        # 🔊 Jarvis voice — speak the answer in-browser (Web Speech API)
        if st.checkbox("🔊 Read answer aloud", key="tts_query"):
            try:
                from utils.voice_output import speak_html
                components.html(speak_html(result.get("answer", "")), height=70)
            except Exception as e:
                st.caption(f"Voice output unavailable: {str(e)[:120]}")

        # ⛉ dual guardrails — form (syntactic) + fact (ClinVar) gates
        try:
            from utils.guardrails import run_guardrails
            from utils.viz import guardrails_html
            verdict = run_guardrails(result.get("answer", ""))
            components.html(guardrails_html(verdict), height=200, scrolling=False)
        except Exception as e:
            st.caption(f"Guardrails unavailable: {str(e)[:120]}")

        # 🛡 hallucination guard on the answer
        render_clinvar_safety(result.get("answer", ""), key_prefix="query")

        # 📚 on-demand PubMed citations for this question
        with st.expander("📚 Find supporting literature (PubMed)"):
            if st.button("🔎 Search PubMed", key="pubmed_go"):
                try:
                    from utils.pubmed_citations import PubMedClient
                    cited = PubMedClient().cite(question, max_results=4)
                    if cited["live"]:
                        st.success(cited["message"])
                    else:
                        st.info(cited["message"])
                    for c in cited["citations"]:
                        if c.get("pmid"):
                            st.markdown(f"- **{c['title']}** — {c.get('source','')} "
                                        f"{c.get('year','')} · "
                                        f"[PMID {c['pmid']}]({c['url']})")
                        else:
                            st.markdown(f"- 🔗 [{c['title']}]({c['url']})")
                    st.caption("Live from NCBI PubMed (Entrez); verify relevance at source.")
                except Exception as e:
                    st.error(f"PubMed lookup unavailable: {str(e)[:160]}")

        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Agent", result.get("agent_used", "?"))
        c2.metric("Retries", result.get("retries", 0))
        c3.metric("Cache Hit", "✅" if result.get("cache_hit") else "❌")
        c4.metric("Sources", len(result.get("sources", [])))

        st.markdown("### Sources")
        st.markdown(format_sources(result.get("sources", [])))

        if st.checkbox("Show raw JSON"):
            st.json(result)

    if st.session_state.messages:
        st.divider()
        st.markdown("### Chat History")
        for msg in st.session_state.messages[-10:]:
            prefix = "**You**" if msg["role"] == "user" else "**Gemma 4**"
            st.markdown(f"{prefix}: {msg['content'][:300]}")

# ── TAB 2: Analysis ───────────────────────────────────────────────
with tab2:
    st.markdown("## 🧬 Multi-Agent Analysis")

    # ── 📁 Optional: upload a patient VCF (TP53 variants auto-extracted) ──
    with st.expander("📁 Upload a patient VCF (auto-extract TP53 variants)"):
        up = st.file_uploader("VCF file (.vcf / .txt)", type=["vcf", "txt"], key="vcf_up")
        use_sample = st.checkbox("Use a sample VCF instead", key="vcf_sample")
        vcf_bytes = up.getvalue() if up is not None else None
        is_sample = False
        if vcf_bytes is None and use_sample:
            from utils.vcf_parser import sample_vcf
            vcf_bytes = sample_vcf().encode()
            is_sample = True
        # Security gate: validate uploaded files (size, type, binary/exe sniff,
        # actual VCF structure) before parsing. Sample data bypasses the gate.
        gate_ok = True
        if vcf_bytes and not is_sample:
            from utils.security import validate_upload, looks_like_vcf
            v = validate_upload(vcf_bytes, getattr(up, "name", ""),
                                allowed_ext=(".vcf", ".txt"))
            if not v["ok"]:
                st.warning(f"⚠️ {v['friendly']}")
                gate_ok = False
            elif not looks_like_vcf(vcf_bytes.decode("utf-8", errors="replace")):
                st.warning("⚠️ This file doesn't look like a VCF. Expected a "
                           "`##fileformat=VCF` header or tab-delimited variant "
                           "rows (CHROM POS ID REF ALT…).")
                gate_ok = False
        if vcf_bytes and gate_ok:
            try:
                from utils.vcf_parser import parse_vcf_bytes
                from utils.viz import vcf_variant_chart
                parsed = parse_vcf_bytes(vcf_bytes)
                if parsed.get("truncated"):
                    st.info("Large file — only the first portion was parsed.")
                st.caption(f"**{parsed['tp53_count']}** TP53 variant(s) found "
                           f"· {parsed['skipped']} malformed line(s) skipped")
                if parsed["variants"]:
                    st.plotly_chart(vcf_variant_chart(parsed["variants"]),
                                    width="stretch")

                    # Needle / lollipop map: variants positioned along the p53
                    # protein over the domain track. Residue is taken from the
                    # file's own HGVS protein change — no significance invented.
                    from utils.viz import needle_plot
                    needle_variants = []
                    for v in parsed["variants"]:
                        aac = (v.get("amino_acid_change") or "").strip()
                        digits = ""
                        for ch in aac:
                            if ch.isdigit():
                                digits += ch
                            elif digits:
                                break
                        if digits:
                            needle_variants.append(
                                {"position": int(digits), "label": aac})
                    if needle_variants:
                        st.plotly_chart(
                            needle_plot(needle_variants,
                                        title="TP53 variants across the protein"),
                            width="stretch",
                        )
                    st.dataframe(
                        pd.DataFrame([
                            {"AA": v["amino_acid_change"] or "—",
                             "Pos": f"{v['chrom']}:{v['pos']}", "REF>ALT": f"{v['ref']}>{v['alt']}",
                             "QUAL": v["qual"], "FILTER": v["filter"],
                             "Hotspot": "🔴" if v["is_hotspot"] else ""}
                            for v in parsed["variants"]
                        ]),
                        width="stretch", hide_index=True,
                    )
                    opts = [v["amino_acid_change"] for v in parsed["variants"]
                            if v.get("amino_acid_change")]
                    if opts:
                        picked = st.selectbox("Use a variant for analysis:", opts, key="vcf_pick")
                        if st.button("→ Use this mutation", key="vcf_use"):
                            st.session_state["vcf_picked_mut"] = picked
                            st.rerun()
                    else:
                        st.info("No annotated protein changes in this VCF — run it through "
                                "VEP/SnpEff for amino-acid annotation, or enter a mutation manually.")
                st.caption("⚠️ VCF parsed locally; amino-acid changes come from the file's own "
                           "HGVS annotation. Unannotated variants are shown genomic-only.")
            except Exception as e:
                st.error(f"VCF parsing failed: {str(e)[:160]}")

    # ── 🧬 Sanger .ab1 chromatogram → variant calling ──────────────
    # expanded once a trace is loaded so results stay visible across the
    # button-triggered reruns (Streamlit collapses expanders otherwise).
    _ab1_loaded = bool(st.session_state.get("ab1_sample")) or \
        st.session_state.get("ab1_up") is not None
    with st.expander("🧬 Sanger .ab1 chromatogram → variant calling (novel input)",
                     expanded=_ab1_loaded):
        st.caption(
            "Reads a real Sanger trace (ABIF/.ab1): quality QC, base calls, "
            "**heterozygous double-peak detection**, and — with a reference — "
            "variant calls with per-base PHRED confidence. Research use only.")
        ab1_up = st.file_uploader("Chromatogram (.ab1):", type=["ab1"],
                                  key="ab1_up")
        ab1_use_sample = st.button("✨ Use sample trace (synthetic demo)",
                                   key="ab1_sample_btn")

        if ab1_use_sample and SAMPLE_SANGER_PATH.exists():
            st.session_state["ab1_sample"] = {
                "bytes": SAMPLE_SANGER_PATH.read_bytes(),
                "ref": (SAMPLE_SANGER_REF_PATH.read_text(encoding="utf-8").strip()
                        if SAMPLE_SANGER_REF_PATH.exists() else ""),
            }

        ab1_bytes, ab1_is_sample, ab1_ref_default = None, False, ""
        if ab1_up is not None:
            ab1_bytes = ab1_up.getvalue()
            st.session_state.pop("ab1_sample", None)
        elif st.session_state.get("ab1_sample"):
            ab1_bytes = st.session_state["ab1_sample"]["bytes"]
            ab1_ref_default = st.session_state["ab1_sample"]["ref"]
            ab1_is_sample = True

        # Security gate on real uploads (size / exe-magic-byte); samples bypass.
        ab1_ok = True
        if ab1_bytes and not ab1_is_sample:
            from utils.security import validate_upload
            v = validate_upload(ab1_bytes, getattr(ab1_up, "name", ""),
                                allowed_ext=(".ab1",), require_text=False)
            if not v["ok"]:
                st.warning(f"⚠️ {v['friendly']}")
                ab1_ok = False

        if ab1_bytes and ab1_ok:
            if ab1_is_sample:
                st.caption("Loaded **synthetic** demo trace (not real patient data).")
            ref_seq = st.text_area(
                "Optional reference segment (for variant calling):",
                value=ab1_ref_default, height=68, key="ab1_ref",
                help="Leave blank for QC + heterozygous detection only.")
            if st.button("🔬 Analyse trace", key="ab1_analyse"):
                from agents.sanger_ab1 import analyze_ab1
                with st.spinner("Reading chromatogram…"):
                    res = analyze_ab1(ab1_bytes,
                                      reference=(ref_seq.strip() or None))
                if res.get("success"):
                    qc = res["qc"]
                    _het_n = len(res.get("heterozygous_sites", []))
                    _var_n = len(res.get("variants", []))
                    st.session_state["fusion_sanger"] = (
                        f"Sanger read length {qc['length']}, mean Q "
                        f"{qc['mean_quality']}, {_het_n} heterozygous "
                        f"double-peak site(s), {_var_n} variant(s) vs reference.")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Read length", qc["length"])
                    c2.metric("Mean quality", qc["mean_quality"])
                    c3.metric("≥Q20", f"{qc['q20_fraction']*100:.0f}%")
                    st.caption(("✅ " if qc["usable"] else "⚠️ ") + qc["note"])
                    st.code(res["sequence"], language="text")
                    het = res.get("heterozygous_sites", [])
                    if het:
                        st.markdown("**Heterozygous (double-peak) sites:**")
                        st.dataframe(pd.DataFrame(het), width="stretch",
                                     hide_index=True)
                    else:
                        st.caption("No heterozygous double-peaks detected.")
                    if "variants" in res:
                        if res["variants"]:
                            st.markdown("**Variants vs reference:**")
                            st.dataframe(pd.DataFrame(res["variants"]),
                                         width="stretch", hide_index=True)
                        else:
                            st.caption("No variants vs the supplied reference.")
                    st.caption(f"⚠️ {res['disclaimer']}")
                else:
                    st.error(f"Could not read trace: {res.get('error')}")

    _default_mut = st.session_state.get("vcf_picked_mut") or "R175H"
    mutation = st.text_input("TP53 Mutation:", placeholder="e.g., R175H", value=_default_mut)
    cancer = st.selectbox("Cancer type:", ["Colorectal", "Breast", "Ovarian", "Lung", "Gastric"])
    vaf = st.number_input("Variant Allele Frequency (%):", 0.0, 100.0, 50.0)

    # ── Real variant annotation (Ensembl VEP · ClinVar · gnomAD) ──
    with st.expander("🔬 Real variant annotation (Ensembl VEP · ClinVar · gnomAD)", expanded=True):
        use_live_anno = st.checkbox(
            "Fetch live from Ensembl / MyVariant APIs",
            value=False,
            help="Off = instant curated baseline. On = live SIFT/PolyPhen/CADD/"
                 "gnomAD allele-frequency/ClinVar (needs internet; a few seconds).",
        )
        try:
            anno = annotate_variant(mutation, use_live=use_live_anno)
            badge = "🟢 live (Ensembl/MyVariant)" if anno.get("method") == "live" \
                else "⚪ curated baseline"
            srcs = ", ".join(anno.get("sources", []))
            st.caption(f"Source: {badge}" + (f" · {srcs}" if srcs else ""))
            st.plotly_chart(
                variant_annotation_table(anno), width="stretch",
                config={"displayModeBar": False},
            )
            if anno.get("notes"):
                st.info(anno["notes"])
        except Exception as e:
            st.warning(f"Annotation unavailable: {str(e)[:160]}")

        # ── ESM-2 protein-language-model variant effect (offline, precomputed) ──
        st.markdown("**ESM-2 variant effect** — protein language model "
                    "(masked-marginal log-likelihood)")
        try:
            eff = predict_effect(mutation)
            st.plotly_chart(variant_effect_gauge(eff), width="stretch",
                            config={"displayModeBar": False})
            if not eff.get("available") and eff.get("notes"):
                st.caption(f"ℹ️ {eff['notes']}")
        except Exception as e:
            st.warning(f"ESM-2 effect unavailable: {str(e)[:160]}")

    if st.button("🧬 Run Multi-Agent Analysis", width="stretch"):
        st.session_state.pipeline_data = {
            "mutation": mutation,
            "cancer_type": cancer,
            "vaf": vaf,
            "timestamp": datetime.now().isoformat(),
        }

        agent_queries = {
            "mutation_analysis": f"Clinical significance of {mutation} in {cancer} cancer",
            "drug_discovery":    f"Best drugs for {mutation} mutation — include Kenya/KEML availability",
            "clinical_interpretation": f"Classify {mutation} clinically. Prognosis for {cancer}?",
            "liquid_biopsy":    f"VAF thresholds for {mutation}. Current VAF: {vaf}%",
        }

        # ── Live agent status board ──
        st.markdown("#### Agent status")
        status_slots = {agent: st.empty() for agent in agent_queries}
        for agent in agent_queries:
            status_slots[agent].markdown(
                agent_status_badge(agent, "running"), unsafe_allow_html=True
            )

        results: dict = {}
        for agent, query in agent_queries.items():
            start = time.perf_counter()
            try:
                result = safe_query(query, agent_type=agent)
                state = "failed" if result.get("agent_used") == "error" else "complete"
            except Exception as e:  # defensive — safe_query already guards
                result = {"answer": f"Query error: {str(e)[:300]}", "sources": []}
                state = "failed"
            elapsed = time.perf_counter() - start
            results[agent] = result
            status_slots[agent].markdown(
                agent_status_badge(agent, state, elapsed), unsafe_allow_html=True
            )

        st.divider()
        cols = st.columns(2)
        for idx, (agent, result) in enumerate(results.items()):
            with cols[idx % 2]:
                with st.expander(f"🔬 {agent.replace('_', ' ').title()}", expanded=(idx < 2)):
                    answer = result.get("answer", "")
                    st.markdown(answer[:500] + "..." if len(answer) > 500 else answer)
                    st.caption(f"Sources: {len(result.get('sources', []))}")

        # 🛡 hallucination guard across ALL multi-agent answers
        _combined = "\n\n".join(r.get("answer", "") for r in results.values())
        render_clinvar_safety(_combined, key_prefix="analysis")

        # ── Molecular profile (at-a-glance visuals from structured agents) ──
        st.divider()
        st.markdown("### 🔬 Molecular Profile")
        st.caption("Rule-based structured outputs (no LLM) — instant, deterministic.")
        mcol1, mcol2 = st.columns(2)

        with mcol1:
            try:  # Variant Curator → pathogenicity gauge
                from agents.variant_curator import VariantCurator
                c = VariantCurator().classify(mutation).get("classification", {})
                st.plotly_chart(
                    pathogenicity_gauge(c.get("clinical_significance"),
                                        c.get("confidence_score")),
                    width="stretch",
                )
            except Exception as e:
                st.caption(f"Variant classification unavailable: {str(e)[:80]}")

            try:  # Liquid Biopsy → VAF burden gauge (from the VAF input)
                st.plotly_chart(vaf_gauge(vaf), width="stretch")
            except Exception as e:
                st.caption(f"VAF gauge unavailable: {str(e)[:80]}")

        with mcol2:
            try:  # Immunogenicity → TME donut
                from agents.immunogenicity import ImmunogenicityPredictor
                p = ImmunogenicityPredictor().predict(
                    mutation, cancer_type=cancer, vaf=vaf).get("prediction", {})
                st.plotly_chart(
                    tme_donut(p.get("immune_infiltration_score", p.get("t_cell_fraction", 0.2)),
                              p.get("immune_status", "")),
                    width="stretch",
                )
            except Exception as e:
                st.caption(f"Immunogenicity unavailable: {str(e)[:80]}")

        try:  # Gene Expression → pathway diverging bar
            from agents.gene_expression import get_expression_profile
            prof = get_expression_profile(mutation)
            st.plotly_chart(
                pathway_diverging_bar(prof.pathways_activated, prof.pathways_suppressed),
                width="stretch",
            )
        except Exception as e:
            st.caption(f"Pathway profile unavailable for {mutation}: {str(e)[:80]}")

        if st.button("💾 Download JSON"):
            st.download_button(
                "Download",
                stamp_json({"mutation": mutation, "cancer": cancer}),
                f"tp53_{mutation}_{datetime.now().strftime('%Y%m%d')}.json",
                "application/json",
            )

# ── TAB: Live AI Tumour Board (hero) ──────────────────────────────
with tab13:
    st.markdown("## ⭐ Live AI Tumour Board")
    st.caption(
        "Six AI specialists deliberate a TP53 case, cross-examine each other, "
        "and vote toward a consensus recommendation — each with a confidence "
        "earned from the strength of the underlying evidence."
    )
    tb_col1, tb_col2, tb_col3 = st.columns([2, 2, 1])
    with tb_col1:
        tb_mut = st.text_input("TP53 mutation", value="R175H", key="tb_mut")
    with tb_col2:
        tb_cancer = st.selectbox(
            "Cancer type",
            ["Breast", "Colorectal", "Lung", "Ovarian", "Gastric", "Other"],
            key="tb_cancer",
        )
    with tb_col3:
        tb_stage = st.selectbox("Stage", ["I", "II", "III", "IV", "Unknown"],
                                key="tb_stage")

    if st.button("🧑‍⚕️ Convene the board", width="stretch", key="tb_go"):
        try:
            from agents.tumor_board import convene_tumor_board
            from utils.viz import tumor_board_html
            board = convene_tumor_board(
                tb_mut, {"cancer": tb_cancer, "stage": tb_stage})
            st.session_state["tb_board"] = board
        except Exception as e:
            st.error(f"Tumour board unavailable: {str(e)[:160]}")

    board = st.session_state.get("tb_board")
    if board:
        from utils.viz import tumor_board_html, explainability_panel_html
        components.html(tumor_board_html(board), height=760, scrolling=True)

        # ── Explainability "Why?" trace for the same case ──
        st.markdown("### 🔎 Why? — evidence behind the assessment")
        try:
            from agents.explainability import explain_variant
            exp = explain_variant(board.get("mutation", ""),
                                  {"cancer": tb_cancer, "stage": tb_stage})
            components.html(explainability_panel_html(exp), height=660,
                            scrolling=True)
        except Exception as e:
            st.caption(f"Explainability unavailable: {str(e)[:120]}")

        # ── Evidence scenario explorer ("digital twin") ──
        st.markdown("### 🧪 Explore scenarios (evidence digital twin)")
        try:
            from agents.digital_twin import explore_twin
            from utils.viz import scenario_explorer_html
            twin = explore_twin(board.get("mutation", ""),
                                {"cancer": tb_cancer, "stage": tb_stage})
            components.html(scenario_explorer_html(twin), height=520,
                            scrolling=True)
        except Exception as e:
            st.caption(f"Scenario explorer unavailable: {str(e)[:120]}")

        # ── 🔴 Skeptic's cross-examination (adversarial evidence layer) ──
        st.markdown("### 🔴 Skeptic's cross-examination")
        st.caption("Before you trust the consensus, an adversarial skeptic "
                   "hunts for *contradicting* evidence — ClinVar conflicts and "
                   "trials that were stopped early — and challenges the "
                   "recommendation in a bounded 2-turn exchange (never run to "
                   "convergence, so it can't stall on a laptop).")
        if st.button("🔴 Run skeptic cross-examination", key="adv_go"):
            _rec = (board.get("consensus", {}) or {}).get("recommendation", "")
            _mut = board.get("mutation", "")
            if not _rec:
                st.info("No consensus recommendation to challenge.")
            else:
                from agents.adversarial_evidence import adversarial_review
                with st.spinner("Retrieving contradicting evidence + debating…"):
                    rev = adversarial_review(_mut, _rec, cancer=tb_cancer)
                ev = rev.get("evidence", {})
                if ev.get("viability") is not None:
                    st.metric("Counterfactual trial viability",
                              f"{ev['viability']*100:.0f}%")
                if ev.get("contradictions"):
                    st.markdown("**Contradicting evidence found:**")
                    for c in ev["contradictions"]:
                        st.markdown(f"- {c}")
                else:
                    st.caption("No specific contradicting evidence retrieved "
                               "— the recommendation largely stands.")
                deb = rev.get("debate", {})
                if deb.get("success"):
                    for t in deb.get("turns", []):
                        icon = "🔴" if t["role"] == "skeptic" else "🟢"
                        st.markdown(f"{icon} **{t['role'].title()}:** {t['text']}")
                    st.caption(deb.get("note", ""))
                else:
                    st.caption("Debate unavailable: "
                               f"{deb.get('reason', 'unknown')[:120]}")

        with st.expander("⬇️ Export consensus (JSON, RUO-stamped)"):
            st.download_button(
                "Download tumour-board consensus",
                stamp_json(board),
                file_name=f"tumour_board_{board.get('mutation','case')}.json",
                mime="application/json",
                key="tb_dl",
            )
    else:
        st.info("Enter a mutation and convene the board to watch the debate.")


# ── TAB 3: Drug Discovery ─────────────────────────────────────────
with tab3:
    st.markdown("## 💊 Drug Discovery & Targeting")

    mut_input = st.text_input("Mutation for drug search:", value="R175H")

    if st.button("🔍 Find Therapeutic Targets", width="stretch"):
        with st.spinner("Searching drug databases..."):
            result = safe_query(
                f"Best drug candidates for {mut_input}? Focus on mechanism and Kenya/KEML availability.",
                agent_type="drug_discovery"
            )
        st.markdown(result["answer"])

    st.markdown("### 💊 TP53-Pathway Drugs (ChEMBL)")
    try:
        from utils.chembl_client import ChEMBLClient
        use_live = st.checkbox("Query live ChEMBL API", value=True,
                               help="Off = curated offline set only (faster).")
        with st.spinner("Loading TP53-pathway compounds..."):
            data = ChEMBLClient().compounds(use_live=use_live)
        if data["live"]:
            st.success(f"🟢 Live ChEMBL + curated — {data['count']} compounds")
        else:
            st.info(f"⚪ Curated offline set — {data['count']} compounds "
                    "(live ChEMBL unavailable or disabled)")

        drug_df = pd.DataFrame([
            {"Drug": d["name"], "Mechanism": d["mechanism"],
             "Target": d.get("target", "—"), "Phase": d.get("phase_label", "?"),
             "Source": d.get("source", "?")}
            for d in data["compounds"]
        ])
        st.dataframe(drug_df, width="stretch", hide_index=True)
        st.plotly_chart(chembl_phase_chart(data["compounds"]),
                        width="stretch")
        st.caption("Real compound/clinical-phase data from ChEMBL (EBI), "
                   "with a curated TP53-pathway fallback. ChEMBL IDs link out for verification.")
    except Exception as e:
        st.error(f"ChEMBL drug data unavailable: {str(e)[:120]}")

    # ── Candidate ranking + 3D docking pose (illustrative) ──
    st.divider()
    st.markdown("### 🧪 Candidate Ranking & Docking Pose")
    st.caption(
        "Illustrative binding-affinity ranking for the mutation above — shows "
        "*why* one drug is favoured over another. Heuristic estimate, **not a "
        "real docking simulation** (AutoDock Vina is a separate module)."
    )

    candidates = dock_candidates(mut_input)
    rank_df = pd.DataFrame([
        {"Rank": c["rank"], "Drug": c["name"], "Mechanism": c["mechanism"],
         "ΔG (kcal/mol)": c["affinity"], "Why": c["rationale"]}
        for c in candidates
    ])
    st.dataframe(rank_df, width="stretch", hide_index=True)

    dcol1, dcol2 = st.columns([3, 2])
    with dcol1:
        st.plotly_chart(docking_affinity_chart(candidates), width="stretch")
    with dcol2:
        top = candidates[0]
        st.markdown(f"**Top candidate:** {top['name']}  \nΔG ≈ **{top['affinity']} kcal/mol**")
        pocket = parse_residues(mut_input)  # mutation residue + canonical hotspots
        components.html(
            docking_pose_html("2OCJ", pocket, top["name"], top["affinity"]),
            height=540,
        )
        st.caption("Yellow cloud = proposed binding pocket on p53 (illustrative).")

    # ── 🧬 Molecular Docking (AutoDock Vina or estimate) ──
    st.divider()
    st.markdown("### 🧬 Molecular Docking")
    st.caption("Dock a specific candidate against TP53. Uses **AutoDock Vina** when "
               "installed; otherwise a clearly-labelled heuristic **estimate**.")
    try:
        cand_names = [c["name"] for c in dock_candidates(mut_input)]
        dock_drug_pick = st.selectbox("Drug to dock:", cand_names, key="dock_pick")
        dock_mech = next((c["mechanism"] for c in dock_candidates(mut_input)
                          if c["name"] == dock_drug_pick), "")
        if st.button("🧬 Run Docking", width="stretch", key="dock_go"):
            from agents.molecular_docking import MolecularDockingAgent
            dres = MolecularDockingAgent().dock(mut_input, dock_drug_pick,
                                                mechanism=dock_mech)
            if dres["method"] == "autodock_vina":
                st.success(f"🟢 AutoDock Vina — {dres['message']}")
            else:
                st.info(f"⚪ Heuristic estimate (Vina not installed) — {dres['message']}")
            dgcol1, dgcol2 = st.columns([3, 2])
            with dgcol1:
                st.plotly_chart(docking_affinity_gauge(dres), width="stretch")
            with dgcol2:
                st.markdown("**Predicted interactions:**")
                for it in dres["interactions"]:
                    st.markdown(f"- {it}")
                st.caption(f"Pocket residues: {', '.join(map(str, dres['pocket_residues']))}")
            st.caption(f"⚠️ {dres['disclaimer']}")
    except Exception as e:
        st.error(f"Docking unavailable: {str(e)[:160]}")

    # ── 🕸 Synthetic Lethality (DepMap-derived) ──
    st.divider()
    st.markdown("### 🕸 Synthetic-Lethal Targets")
    st.caption("Genes whose inhibition selectively kills TP53-mutant cells "
               "(curated from DepMap dependency signals + published p53 SL screens).")
    try:
        from agents.synthetic_lethality import SyntheticLethalityModeler
        sl = SyntheticLethalityModeler().model(mut_input)
        st.plotly_chart(synthetic_lethal_network(sl), width="stretch")
        sl_df = pd.DataFrame([
            {"Target": t["gene"], "Mechanism": t["mechanism"],
             "Drug": t.get("drug", "—"), "Evidence": t["evidence"],
             "Druggability": t["druggability"]}
            for t in sl["targets"]
        ])
        st.dataframe(sl_df, width="stretch", hide_index=True)
        st.caption(f"⚠️ {sl['disclaimer']}")
    except Exception as e:
        st.error(f"Synthetic-lethality model unavailable: {str(e)[:160]}")

# ── TAB 4: Visualization ──────────────────────────────────────────
with tab4:
    st.markdown("## 📊 Visualization & Metrics")

    # ── DNA double-helix codebase graph (signature visual) ──
    st.markdown("### 🧬 Codebase as DNA")
    st.caption(
        "The platform's own modules (nodes) and their imports (edges), laid out "
        "on a DNA double helix — join the dots and they trace the structure this "
        "project is built to understand. Drag to rotate."
    )
    try:
        from utils.codegraph import build_helix_codegraph
        from utils.viz import codegraph_helix_html
        components.html(codegraph_helix_html(_cached_codegraph()),
                        height=640, scrolling=False)
    except Exception as e:
        st.caption(f"Code graph unavailable: {str(e)[:120]}")
    st.divider()

    # ── Edge device: mock portable sequencer (hardware abstraction) ──
    with st.expander("🧪 Edge device — portable sequencer (simulated interface)"):
        st.caption("A software mock of a portable sequencer's device API "
                   "(develop-before-hardware pattern). No physical instrument "
                   "is attached; a real driver drops in unchanged.")
        if st.button("▶ Run a mock sequencing cycle", key="mockdev_go"):
            try:
                from utils.mock_hardware import run_mock_demo_sequence
                from utils.viz import mock_device_html
                components.html(mock_device_html(run_mock_demo_sequence()),
                                height=320, scrolling=False)
            except Exception as e:
                st.caption(f"Mock device unavailable: {str(e)[:120]}")

    # ── Microfluidic QC: the lab that knows when to stop ──
    with st.expander("💧 Microfluidic QC — abort-to-save-compute (simulated)"):
        st.caption("An intelligent fluidics QC policy: it aborts a run the moment "
                   "it detects an unrecoverable fault (bubble/occlusion), saving "
                   "sequencing compute. Simulated telemetry — not real imaging.")
        try:
            from utils.microfluidic import demo_scenarios
            from utils.viz import microfluidic_html
            scen = demo_scenarios()
            pick = st.radio("Scenario", ["fluidics_fault", "clean_run"],
                            horizontal=True, key="mf_pick",
                            format_func=lambda k: "Fluidics fault (bubble)"
                            if k == "fluidics_fault" else "Clean run")
            components.html(microfluidic_html(scen[pick]), height=300,
                            scrolling=False)
        except Exception as e:
            st.caption(f"Microfluidic QC unavailable: {str(e)[:120]}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ctDNA VAF Timeline")
        st.caption("Press ▶ Play to watch the treatment response unfold.")
        fig = animated_vaf_timeline(
            days=[0, 5, 10, 15, 20, 25],
            vafs=[50, 48, 45, 42, 38, 35],
            mrd_threshold=5.0,
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("### TP53 Hotspot Frequency")
        st.caption("Press ▶ Play to watch the frequencies build up.")
        fig2 = animated_hotspot_bar(
            codons=["175", "248", "273", "249", "282", "220"],
            freqs=[8.0, 7.5, 7.0, 6.5, 4.0, 3.5],
        )
        st.plotly_chart(fig2, width="stretch")

    st.markdown("### TP53 Protein Domain Map")
    domain_df = pd.DataFrame({
        "Domain":   ["TAD1", "TAD2", "PRD", "DBD", "NLS", "TET", "REG"],
        "Start":    [1, 40, 67, 94, 316, 323, 364],
        "End":      [40, 67, 98, 292, 325, 356, 393],
        "Function": ["Transactivation", "Transactivation", "Proline-rich",
                     "DNA-binding (hotspot region)", "Nuclear signal",
                     "Tetramerization", "Regulatory"],
    })
    st.dataframe(domain_df, width="stretch", hide_index=True)

    st.markdown("### Multi-Agent Network")
    st.caption(
        "Interactive 3D map of how the agents connect — drag to rotate, "
        "scroll to zoom, click a node to focus. Colours group agents by domain."
    )
    components.html(
        agent_graph_3d_html(build_agent_graph_data(), height=560),
        height=580,
    )

    with st.expander("▶ 2D dispatch animation (offline-safe fallback)"):
        st.caption("Press ▶ Trace dispatch to watch the orchestrator fan out to each agent.")
        arch_fig = agent_architecture_diagram([
            "mutation_analysis", "drug_discovery", "clinical_interpretation",
            "liquid_biopsy", "gene_expression", "domain_annotation",
            "pathology_vision", "tnm_staging", "variant_curator", "immunogenicity",
        ])
        st.plotly_chart(
            arch_fig, width="stretch",
            # Lock zoom so a stray click/scroll can't jump the diagram, and
            # drop the zoom buttons from the toolbar (it's a fixed layout).
            config={
                "scrollZoom": False, "displaylogo": False, "doubleClick": False,
                "modeBarButtonsToRemove": [
                    "zoom2d", "zoomIn2d", "zoomOut2d", "pan2d",
                    "autoScale2d", "select2d", "lasso2d", "resetScale2d",
                ],
            },
        )

# ── TAB 5: Report ─────────────────────────────────────────────────
with tab5:
    st.markdown("## 📋 Clinical Report Generator")

    patient_id   = st.text_input("Patient ID (will be hashed):", value="DEMO-001")
    rep_mutation = st.text_input("Mutation:", value="R175H")
    rep_vaf      = st.slider("VAF (%):", 0, 100, 50)
    rep_cancer   = st.selectbox("Cancer:", ["Colorectal", "Breast", "Ovarian", "Lung"])
    include_keml = st.checkbox("Include Kenya/KEML resources", value=True)

    if st.button("📋 Generate Clinical Report", width="stretch"):
        with st.spinner("Generating comprehensive report via Gemma 4..."):
            query = (
                f"Generate a comprehensive clinical report for: "
                f"Mutation: {rep_mutation}, VAF: {rep_vaf}%, Cancer: {rep_cancer}. "
                f"{'Include Kenya drug availability.' if include_keml else ''} "
                f"Include: Executive summary, mutation significance, drug options, recommendations."
            )
            result = safe_query(query, agent_type="clinical_interpretation")

        st.markdown("### Clinical Report")
        st.markdown(result["answer"])

        st.download_button(
            "⬇️ Download Report",
            stamp_markdown(result["answer"], title=f"TP53 Clinical Report — {rep_mutation}"),
            file_name=f"report_{rep_mutation}_{datetime.now().strftime('%Y%m%d')}.md",
        )

    # ── 🧩 Multimodal fusion — one summary across every modality ──
    st.divider()
    st.markdown("### 🧩 Multimodal Case Fusion")
    st.caption("Synthesises everything you've gathered — variant, Gemma-vision "
               "slide + lab-report readings, Sanger trace, and your notes — into "
               "one unified case summary. Only modalities you actually provided "
               "are used.")
    _fusion_inputs = {
        "mutation": rep_mutation,
        "cancer": rep_cancer,
        "vaf": f"{rep_vaf}%",
        "pathology_narration": st.session_state.get("fusion_pathology"),
        "lab_report_summary": st.session_state.get("fusion_lab_report"),
        "sanger_summary": st.session_state.get("fusion_sanger"),
    }
    _available = [k for k in ("pathology_narration", "lab_report_summary",
                              "sanger_summary") if st.session_state.get(k)]
    st.caption("Modalities detected this session: "
               + (", ".join(_available) if _available
                  else "variant/report only (add a slide, lab photo, or Sanger "
                       "trace for a richer fusion)."))
    fusion_notes = st.text_area("Clinician notes (optional):", height=80,
                                key="fusion_notes")
    if st.button("🧩 Fuse case summary", key="fusion_go"):
        _fusion_inputs["notes"] = fusion_notes
        from agents.multimodal_fusion import fuse_case
        with st.spinner("Fusing the case across modalities via Gemma 4…"):
            fused = fuse_case(_fusion_inputs, llm=st.session_state.get("rag"))
        if fused.get("success"):
            st.success("Fused: " + ", ".join(fused["modalities_used"]))
            st.markdown(fused["summary"])
            st.caption(f"⚠️ {fused['disclaimer']}")
            st.download_button(
                "⬇️ Download fused summary",
                stamp_markdown(fused["summary"],
                               title="Multimodal Case Fusion"),
                file_name="fused_case_summary.md", key="fusion_dl")
        else:
            st.warning(fused.get("reason", "Nothing to fuse yet."))

    # ── 🌍 Swahili / multilingual patient report ──
    st.divider()
    st.markdown("### 🌍 Patient report — English & Kiswahili")
    st.caption("Patient-facing explanations in English and culturally-nuanced "
               "Swahili, for clinicians serving Swahili-speaking communities.")
    if st.button("🌍 Generate Swahili patient report", key="swahili_go"):
        if st.session_state.get("rag"):
            with st.spinner("Generating multilingual reports..."):
                try:
                    from agents.multilingual import MultilingualReportAgent
                    agent = MultilingualReportAgent(st.session_state.rag)
                    reports = agent.generate({"mutation": rep_mutation,
                                              "cancer": rep_cancer})
                    st.markdown("#### 🇰🇪 Ripoti ya Mgonjwa (Kiswahili)")
                    st.markdown(reports.get("patient_report_sw", ""))
                    with st.expander("🇬🇧 English patient report"):
                        st.markdown(reports.get("patient_report_en", ""))
                    st.download_button(
                        "⬇️ Download Swahili report",
                        stamp_markdown(reports.get("patient_report_sw", ""),
                                       title=f"Ripoti ya TP53 — {rep_mutation}"),
                        file_name=f"ripoti_{rep_mutation}.md", key="sw_dl",
                    )
                except Exception as e:
                    st.error(f"Multilingual report unavailable: {str(e)[:160]}")
        else:
            st.warning("RAG system offline — needs an active inference backend.")

    # ── 📑 IND Draft Generator (regulatory) ──
    st.divider()
    st.markdown("### 📑 IND Draft Generator")
    st.caption("Draft FDA Investigational New Drug skeleton for a mutation + lead "
               "candidate. Rule-based scaffold — **not a submission-ready document**.")
    icol1, icol2 = st.columns(2)
    with icol1:
        ind_mut = st.text_input("Mutation:", value="R175H", key="ind_mut")
    with icol2:
        ind_cancer = st.text_input("Cancer type:", value="breast cancer", key="ind_cancer")

    if st.button("📑 Generate IND Draft", width="stretch", key="ind_go"):
        try:
            from agents.ind_generator import INDGenerator
            from utils.viz import dock_candidates
            cands = dock_candidates(ind_mut)  # reuse ranked candidates as leads
            gen = INDGenerator()
            ind = gen.generate(ind_mut, ind_cancer, drug_candidates=cands)
            st.success(f"{ind['message']} · readiness {ind['readiness_pct']}%")
            st.plotly_chart(ind_section_chart(ind), width="stretch")
            md = gen.render_markdown(ind)
            with st.expander("Preview IND draft", expanded=True):
                st.markdown(md)
            st.download_button(
                "⬇️ Download IND Draft (Markdown)", md,
                file_name=f"IND_draft_{ind_mut}_{datetime.now().strftime('%Y%m%d')}.md",
                key="ind_dl",
            )
            st.caption(f"⚠️ {ind['draft']['disclaimer']}")
        except Exception as e:
            st.error(f"IND generation unavailable: {str(e)[:160]}")

# ── TAB 6: Structure ──────────────────────────────────────────────
with tab6:
    st.markdown("## 🔬 3D Structure Visualization")
    st.markdown(
        "Interactive p53 protein structure — domain-coloured, auto-rotating, "
        "with hotspot residues labelled. Colours match the domain map below."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        # EXPANDED: added 3 more clinically relevant PDB structures
        pdb_options = {
            "2OCJ — p53 DBD wildtype":         "2OCJ",
            "1TUP — p53 bound to DNA":          "1TUP",
            "2OCO — p53 R175H mutant":          "2OCO",
            "2J1X — p53 R248W mutant":          "2J1X",
            "2LZH — p53 tetramerization domain":"2LZH",
            "4HJE — p53 R248W (contact mutant)":"4HJE",
            "6FF9 — p53 bound to MDM2":         "6FF9",
        }
        selected_pdb = st.selectbox("Select structure:", list(pdb_options.keys()))
        pdb_id = pdb_options[selected_pdb]

        highlight = st.text_input("Highlight residue:", value="175")

        st.markdown(f"**Structure:** `{pdb_id}` — [Open in RCSB](https://www.rcsb.org/structure/{pdb_id})")

        enhanced = st.toggle(
            "Enhanced viewer — auto-rotate + hotspot highlight",
            value=True,
            help="Spins the structure and highlights hotspot residues 175/248/273 "
                 "plus your selected residue. Uncheck for the standard Mol* viewer.",
        )

        if enhanced:
            residues = parse_residues(highlight)
            st.caption(f"Highlighting residues: {', '.join(map(str, residues))}")
            components.html(protein_viewer_html(pdb_id, residues), height=500)
        else:
            # Mol* viewer embed
            components.iframe(
                f"https://molstar.org/viewer/?pdb={pdb_id}",
                height=480,
                scrolling=False,
            )

        # ── AlphaFold predicted structure (real, pLDDT-coloured) ──
        with st.expander("🧬 AlphaFold predicted structure (real, pLDDT confidence)"):
            if st.checkbox("Load AlphaFold model for TP53 (UniProt P04637)",
                           value=False,
                           help="Fetches the real AlphaFold-predicted structure and "
                                "colours it by per-residue confidence (pLDDT)."):
                with st.spinner("Fetching AlphaFold model…"):
                    struct = get_tp53_structure(use_live=True)
                if struct.available:
                    st.caption(
                        f"Source: 🟢 AlphaFold DB · mean pLDDT **{struct.mean_plddt}** · "
                        f"{struct.n_residues} residues · [model file]({struct.model_url})"
                    )
                    components.html(
                        alphafold_viewer_html(
                            struct.pdb_text, residues=parse_residues(highlight),
                            mean_plddt=struct.mean_plddt),
                        height=500)
                    st.plotly_chart(
                        plddt_profile_chart(struct.per_residue, mean_plddt=struct.mean_plddt),
                        width="stretch")
                    hp = ", ".join(f"{r}: {v:.0f}" for r, v in struct.hotspot_plddt.items()
                                   if v is not None)
                    if hp:
                        st.caption(f"Hotspot residue confidence (pLDDT): {hp}")

                    # ── Mutation-aware view: highlight the patient's residue +
                    #    druggable sites on the same real structure ──
                    st.markdown("##### 🎯 Mutation-aware view")
                    mut_in = st.text_input(
                        "Highlight a specific mutation on the structure",
                        value="R175H", key="struct_mut")
                    from utils.viz import mutation_structure_html
                    components.html(
                        mutation_structure_html(struct.pdb_text, mut_in),
                        height=500)
                    st.caption("Gold = the patient's mutated residue · cyan = "
                               "druggable/reactivation sites · purple = zinc cluster.")
                else:
                    st.info(struct.notes or "AlphaFold model unavailable.")

    with col2:
        st.markdown("### Colour Legend")
        domain_ref = pd.DataFrame({
            "Domain":   [d["name"] for d in P53_DOMAINS],
            "Residues": [f"{d['start']}–{d['end']}" for d in P53_DOMAINS],
            "Function": [d["function"] for d in P53_DOMAINS],
        })
        st.dataframe(
            domain_ref.style.apply(
                lambda row: [f"background-color: {P53_DOMAINS[row.name]['color']}33"] * len(row),
                axis=1,
            ),
            width="stretch", hide_index=True,
        )
        st.caption("🔴 Red spheres = hotspot residues (175 / 248 / 273) + your selection.")

        st.markdown("### Hotspot Positions")
        hotspot_ref = pd.DataFrame({
            "Mutation": ["R175H", "R248W", "R248Q", "R273H", "R273C", "R282W"],
            "Residue":  [175, 248, 248, 273, 273, 282],
            "Type":     ["Conform.", "Contact", "Contact",
                         "Contact", "Contact", "Conform."],
        })
        st.dataframe(hotspot_ref, width="stretch", hide_index=True)

    # ── Domain map (side chart, colours match the 3D structure) ──
    st.markdown("### Structure Colour Map")
    st.plotly_chart(domain_legend_chart(), width="stretch")

    # ── Multimodal RAG narration: type or speak ──
    st.markdown("### 🧠 Structure Narration")
    st.caption("Ask about this structure by typing or speaking.")
    nar_col1, nar_col2 = st.columns([2, 1])
    with nar_col1:
        struct_q = st.text_input(
            "Question about this structure:",
            value=f"Explain the structure of p53 PDB {pdb_id}, focusing on the "
                  f"DNA-binding domain and residue {highlight}.",
            key="struct_narration_q",
        )
    with nar_col2:
        struct_audio = st.audio_input("🎙️ Or speak", key="struct_narration_audio")

    if struct_audio is not None:
        if not whisper_available():
            st.warning("⚠️ Voice needs Whisper: `pip install openai-whisper`")
        else:
            with st.spinner("Transcribing with Whisper..."):
                spoken = transcribe(struct_audio)
            if spoken.startswith("Transcription error"):
                st.error(spoken)
            else:
                st.success(f"**Heard:** {spoken}")
                struct_q = spoken

    if st.button("🧠 Explain structure", key="explain_structure_btn"):
        if not struct_q.strip():
            st.info("Type or speak a question first.")
        else:
            with st.spinner("Querying Gemma 4..."):
                result = safe_query(struct_q, agent_type="domain_annotation")
            st.markdown(result["answer"])

    # ── 🔩 Structural Mechanics & Cavity Analysis ──
    st.divider()
    st.markdown("### 🔩 Structural Mechanics & Cavity Analysis")
    st.caption("ΔΔG destabilisation, binding-pocket druggability and residue "
               "contacts for drug design (curated biophysical estimates).")
    sa_mut = st.text_input("Mutation:", value="Y220C", key="struct_analyze_mut")
    try:
        from agents.structural_analyzer import StructuralAnalyzer
        sa = StructuralAnalyzer().analyse(sa_mut)
        scol1, scol2 = st.columns([3, 2])
        with scol1:
            st.plotly_chart(structural_profile_radar(sa), width="stretch")
        with scol2:
            st.metric("ΔΔG (destabilisation)", f"{sa['ddG_kcal_mol']} kcal/mol")
            st.metric("Druggability", f"{sa['druggability']:.0%}")
            st.markdown(f"**Pocket:** {sa['pocket']}")
            st.markdown(f"**Strategy:** {sa['strategy']}")
            if sa["contact_residues"]:
                st.caption(f"Contact residues: {', '.join(map(str, sa['contact_residues']))}")
        st.info(sa["note"])
        st.caption(f"⚠️ {sa['disclaimer']} · source: {sa['structure_source']}")
    except Exception as e:
        st.error(f"Structural analysis unavailable: {str(e)[:160]}")

# ── TAB 7: Voice ──────────────────────────────────────────────────
with tab7:
    st.markdown("## 🎤 Voice Input (Beta)")
    st.markdown("Speak your question — transcribed locally via Whisper")

    WHISPER_AVAILABLE = whisper_available()
    if WHISPER_AVAILABLE:
        st.success("✅ Whisper ready")
    else:
        st.warning("⚠️ Install Whisper: `pip install openai-whisper`")

    audio_bytes = st.audio_input("🎙️ Record your question (max 30s):")

    if audio_bytes and WHISPER_AVAILABLE:
        with st.spinner("Transcribing with Whisper..."):
            text = transcribe(audio_bytes)
        if not text.startswith("Transcription error"):
            st.success(f"**Transcribed:** {text}")
            with st.spinner("Querying Gemma 4..."):
                result = safe_query(text, agent_type=forced_agent)
            st.markdown("### Answer")
            st.markdown(result["answer"])
        else:
            st.error(text)

    st.divider()
    st.markdown("### ⌨️ Or Type Your Question")
    voice_text = st.text_area("Type here:", height=100,
                              placeholder="e.g., What is the prognosis for R175H?")
    if st.button("🚀 Submit", width="stretch"):
        if voice_text:
            with st.spinner("Processing..."):
                result = safe_query(voice_text, agent_type=forced_agent)
            st.markdown(result["answer"])

# ── TAB 8: Debug ──────────────────────────────────────────────────
with tab8:
    st.markdown("## 🛠 Debug & Admin")

    c1, c2, c3 = st.columns(3)
    c1.metric("RAG System", "✅ Online" if st.session_state.rag else "❌ Offline")
    c2.metric("RAG Available", str(RAG_AVAILABLE))
    c3.metric("Messages", len(st.session_state.messages))

    # ── Live compute backend (honest probe) ──
    try:
        from utils.hardware_probe import detect_compute
        hw = detect_compute()
        acc = hw.get("accelerator", "cpu")
        icon = {"amd_rocm": "🟢 AMD ROCm", "nvidia_cuda": "🟢 CUDA",
                "cpu": "⚪ CPU-only"}.get(acc, acc)
        st.markdown(f"**Compute backend:** {icon} — {hw.get('summary','')}")
    except Exception as e:
        st.caption(f"Compute probe unavailable: {str(e)[:120]}")

    # ── AMD deployment & hardware benchmarks ──
    st.markdown("### ⚡ AMD deployment & benchmarks")
    try:
        from utils.amd_benchmark import load_benchmark, DEPLOYMENT_TIERS
        from utils.viz import deployment_panel_html, amd_benchmark_chart
        components.html(deployment_panel_html(DEPLOYMENT_TIERS), height=320,
                        scrolling=True)
        bench = load_benchmark()
        if bench.get("available"):
            st.plotly_chart(amd_benchmark_chart(bench), width="stretch")
            st.caption(f"Measured {bench.get('generated_utc', '')} on "
                       f"{bench.get('device', {}).get('device_name', 'AMD hardware')}.")
        else:
            st.info(f"⚙️ {bench.get('reason', 'Benchmark not yet run.')} "
                    "Run `python tools/benchmark_amd.py` on an AMD GPU host, "
                    "then commit `data/amd_benchmark.json`.")
    except Exception as e:
        st.caption(f"AMD panel unavailable: {str(e)[:120]}")

    # ── Token-efficient router savings ──
    st.markdown("### ⚡ Token-efficient router")
    st.caption("Every query is routed to the cheapest correct path — cache or "
               "deterministic agent before the LLM. Savings are measured live.")
    try:
        from utils.token_router import get_router
        from utils.viz import token_router_chart
        rep = get_router().report()
        if rep.get("queries"):
            st.plotly_chart(token_router_chart(rep), width="stretch",
                            config={"displayModeBar": False})
            st.caption(f"{rep['llm_calls_avoided']}/{rep['queries']} queries "
                       f"avoided the LLM · ~{rep['tokens_saved']:,} tokens "
                       f"(≈ ${rep['usd_saved_est']}) saved this session.")
        else:
            st.info("Run a few queries to see routing savings accumulate.")
    except Exception as e:
        st.caption(f"Router panel unavailable: {str(e)[:120]}")
    st.divider()

    # ── Agent evaluation harness ──
    st.markdown("### 📏 Agent evaluation harness")
    st.caption("Deterministic per-agent metrics — latency, success rate, "
               "calibration, citation coverage, uncertainty flagging.")
    if st.button("▶ Run agent evaluation", key="eval_go"):
        try:
            from benchmarks.agent_eval import run_agent_eval
            from utils.viz import agent_eval_table
            report = run_agent_eval()
            st.plotly_chart(agent_eval_table(report), width="stretch",
                            config={"displayModeBar": False})
            st.caption("✅ All agents passing" if report["all_passing"]
                       else "⚠️ Some agents below threshold")
        except Exception as e:
            st.caption(f"Agent eval unavailable: {str(e)[:120]}")
    st.divider()

    # ── Offline Cancer Copilot readiness ──
    st.markdown("### 📡 Offline readiness")
    try:
        from utils.offline_status import offline_capabilities
        from utils.viz import offline_readiness_html
        components.html(offline_readiness_html(offline_capabilities()),
                        height=440, scrolling=True)
    except Exception as e:
        st.caption(f"Offline readiness unavailable: {str(e)[:120]}")
    st.divider()

    if st.session_state.rag:
        try:
            stats = st.session_state.rag.cache_stats()
            st.markdown("### Cache Stats")
            st.json(stats)
        except Exception:
            pass

        # ── ⚡ Cache warming: pre-compute related hotspots ──
        st.markdown("### ⚡ Cache warming")
        from agents.rag_chain import related_hotspots
        warm_mut = st.text_input("Warm related hotspots for:", value="R175H",
                                 key="warm_mut")
        st.caption("Related hotspots: "
                   + (", ".join(related_hotspots(warm_mut)) or "none known"))
        if st.button("⚡ Warm cache", key="warm_go"):
            cache = getattr(st.session_state.rag, "cache", None)
            if cache is None or not related_hotspots(warm_mut):
                st.info("Nothing to warm for that mutation.")
            else:
                from agents.rag_chain import _build_backend
                _bk = _build_backend()

                def _wgen(q):
                    return _bk.generate(
                        "You are a concise TP53 clinical-genomics assistant.",
                        q, max_tokens=384)

                with st.spinner("Pre-computing related-hotspot answers…"):
                    wres = cache.warm(warm_mut, "clinical_interpretation", _wgen)
                st.success(f"Warmed: {', '.join(wres['warmed']) or 'none'} · "
                           f"already cached: {', '.join(wres['skipped']) or 'none'}")

    # ── Conversation memory (persistent, PII-scrubbed) ──
    _mem = st.session_state.get("memory")
    if _mem is not None:
        st.markdown("### 🧠 Conversation Memory")
        st.caption(
            "Past turns are stored locally (PII-scrubbed) so conversations "
            "resume across restarts. Session: "
            f"`{st.session_state.get('session_id', '?')}`"
        )
        try:
            st.json(_mem.stats())
        except Exception:
            pass
        mc1, mc2 = st.columns(2)
        if mc1.button("🗑 Clear this session's memory"):
            _mem.clear(st.session_state.get("session_id"))
            st.success("Session memory cleared")
        if mc2.button("🗑 Clear ALL memory"):
            _mem.clear(None)
            st.success("All conversation memory cleared")

    st.markdown("### Test Query")
    test_q = st.text_input("Test question:", value="What is R175H?")
    if st.button("Run test"):
        with st.spinner("Testing..."):
            result = safe_query(test_q)
        st.json(result)

    st.markdown("### Pipeline Data")
    st.json(st.session_state.pipeline_data if st.session_state.pipeline_data else {"empty": True})

    if st.button("🗑 Clear cache"):
        try:
            st.session_state.rag.cache.reset_stats()
            st.success("Cache cleared")
        except Exception:
            st.session_state.messages = []
            st.success("Session cleared")

# ── TAB 9: Pathology ──────────────────────────────────────────────
with tab9:
    st.markdown("## 🔬 Multimodal Pathology — Gemma 4 Vision")
    st.markdown(
        "Gemma 4 reads images **directly** — a stained slide *or* a photographed "
        "paper lab report — with **no separate OCR engine**. One model sees the "
        "image, reasons over it, and cross-checks any mutation it reads against "
        "ClinVar. This is the platform's *Gemma-as-multimodal-core*."
    )

    gv = load_gemma_vision_agent()

    col1, col2 = st.columns([2, 1])

    with col1:
        input_mode = st.radio(
            "What are you sharing?",
            ["🧫 Pathology slide", "📄 Photographed lab report"],
            horizontal=True,
        )
        is_slide = input_mode.startswith("🧫")
        exts = ["jpg", "jpeg", "png", "tiff"] if is_slide else ["jpg", "jpeg", "png"]

        up = st.file_uploader(
            f"Upload {'an H&E slide' if is_slide else 'a photo of a paper lab report'} "
            f"({'/'.join(e.upper() for e in exts)}):",
            type=exts, key=f"path_up_{is_slide}",
        )
        use_sample = st.button(
            f"✨ Use sample {'slide' if is_slide else 'report'} (synthetic demo)",
            width="stretch",
        )

        # The "Use sample" button is one-shot (True only on its own rerun), so
        # persist the chosen sample in session_state — otherwise clicking a
        # downstream button (e.g. "Read report") reruns with the sample gone and
        # the click is lost. A real upload persists on its own via the widget.
        if use_sample:
            sample_path = SAMPLE_SLIDE_PATH if is_slide else SAMPLE_REPORT_PATH
            if sample_path.exists():
                st.session_state["path_sample"] = {
                    "bytes": sample_path.read_bytes(),
                    "name": sample_path.name,
                    "is_slide": is_slide,
                }
            else:
                st.warning("Sample image not found in assets/.")

        # Resolve image bytes from a real upload or the persisted sample.
        img_bytes, img_name, is_sample = None, "", False
        if up is not None:
            img_bytes, img_name = up.getvalue(), up.name
            st.session_state.pop("path_sample", None)  # a real upload wins
        else:
            s = st.session_state.get("path_sample")
            if s and s.get("is_slide") == is_slide:
                img_bytes, img_name, is_sample = s["bytes"], s["name"], True

        # Security gate on real uploads (size / exe-magic-byte / extension).
        # Synthetic bundled samples bypass the gate.
        gate_ok = True
        if img_bytes and not is_sample:
            from utils.security import validate_upload
            v = validate_upload(img_bytes, img_name,
                                allowed_ext=tuple("." + e for e in exts),
                                require_text=False)
            if not v["ok"]:
                st.warning(f"⚠️ {v['friendly']}")
                gate_ok = False

        if img_bytes and gate_ok:
            st.image(img_bytes,
                     caption="Synthetic demo image" if is_sample else "Your image",
                     width="stretch")
            mime = "image/png" if img_name.lower().endswith(".png") else "image/jpeg"

            if gv is None:
                st.info(
                    "Gemma vision needs `GOOGLE_API_KEY` in `.env` (the same key "
                    "as `INFERENCE_MODE=api`). The classic CNN slide path below "
                    "still works without it."
                )

            # ── Slide path ──────────────────────────────────────────
            if is_slide:
                if gv is not None and st.button(
                        "👁️ Read slide with Gemma Vision",
                        type="primary", width="stretch"):
                    with st.spinner("Gemma 4 is reading the slide…"):
                        res = gv.read_pathology_slide(
                            img_bytes, mime,
                            mutation_data=st.session_state.pipeline_data)
                    if res.get("success"):
                        st.session_state["fusion_pathology"] = res["narration"]
                        st.markdown("### 🧠 Gemma 4 Vision — slide narration")
                        st.markdown(res["narration"])
                        st.caption(f"Model: {res.get('model')} · direct image "
                                   "reading, no separate CNN or OCR")
                        st.download_button(
                            "⬇️ Download narration",
                            data=stamp_markdown(
                                res["narration"],
                                title="Gemma Vision Pathology Narration"),
                            file_name="gemma_vision_pathology.md",
                            mime="text/markdown")
                    else:
                        st.error(f"Gemma vision failed: {res.get('error')}")

                # Classic CNN tissue-classification path — kept as a secondary
                # option (needs torch/timm; unavailable on the slim cloud image).
                with st.expander("Classic CNN tissue classification (UNI/ResNet)"):
                    if st.button("🔬 Analyse with CNN", key="cnn_analyse"):
                        agent = load_pathology_agent()
                        if agent is None:
                            st.error(
                                "CNN pathology agent unavailable here "
                                "(torch/torchvision not installed). Gemma Vision "
                                "above needs no such dependency.")
                        else:
                            suffix = Path(img_name).suffix or ".jpg"
                            with tempfile.NamedTemporaryFile(
                                    delete=False, suffix=suffix) as _tf:
                                _tf.write(img_bytes)
                                tmp = Path(_tf.name)
                            try:
                                with st.spinner("Analysing slide with CNN…"):
                                    result = agent.process_slide(
                                        str(tmp),
                                        mutation_data=st.session_state.pipeline_data)
                            finally:
                                try:
                                    if tmp.exists():
                                        tmp.unlink()
                                except Exception as ce:
                                    log.warning(f"temp slide cleanup: {ce}")
                            if result["success"]:
                                st.session_state.last_pathology_result = result
                                st.success(f"Top tissue: **{result['top_tissue']}**")
                                st.dataframe(
                                    pd.DataFrame(result["tissue_classifications"]),
                                    width="stretch", hide_index=True)
                                if result["mutation_correlations"]:
                                    for corr in result["mutation_correlations"]:
                                        st.markdown(f"**{corr['mutation']}**:")
                                        for c, f in corr["cancer_correlations"].items():
                                            st.progress(f, text=f"{c}: {f:.0%}")
                                st.markdown(result["llm_narration"])
                            else:
                                st.error(f"CNN analysis failed: {result.get('error')}")

            # ── Lab-report photo path ───────────────────────────────
            else:
                if gv is not None and st.button(
                        "📑 Read report with Gemma Vision",
                        type="primary", width="stretch"):
                    with st.spinner("Gemma 4 is reading the report…"):
                        res = gv.read_lab_report_photo(img_bytes, mime)
                    if res.get("success"):
                        st.session_state["fusion_lab_report"] = res["summary"]
                        st.markdown("### 📄 Extracted fields")
                        st.markdown(res["summary"])
                        muts = res.get("candidate_mutations") or []
                        if muts:
                            st.markdown("### 🔬 Mutations read → ClinVar cross-check")
                            for note in res.get("clinvar_cross_check", []):
                                st.markdown(
                                    f"- **{note['mutation']}** → ClinVar: "
                                    f"`{note['verdict']}`")
                        else:
                            st.info("No mutation notation was confidently read "
                                    "from the image.")
                        if res.get("caution"):
                            st.warning(res["caution"])
                    else:
                        st.error(f"Gemma vision failed: {res.get('error')}")

        if img_bytes is None:
            st.info("Upload an image or click **Use sample** to watch Gemma read it.")
            st.markdown("""
**Free slide datasets:**
- [TCGA Portal](https://portal.gdc.cancer.gov)
- [Kaggle Histopathology](https://www.kaggle.com/datasets?search=histopathology)
""")

    with col2:
        st.markdown("### How this works")
        st.markdown("""
**Primary:** Gemma 4 Vision reads the image **directly** — slide *or*
photographed paper report — then reasons over it. No OCR engine, no separate
CNN, one model for sight + language.

**Lab-report reader:** extracts gene / variant / VAF / sample type, then
**cross-checks every mutation it reads against ClinVar** before trusting it.

**Fallback:** a classic UNI/ResNet CNN tissue classifier (local only, needs
torch/timm).
""")

        st.markdown("### Status")
        if gv is not None:
            st.success("✅ Gemma Vision ready (GOOGLE_API_KEY set)")
        else:
            st.warning("⚠️ Gemma Vision needs GOOGLE_API_KEY in `.env`")
        try:
            cnn = load_pathology_agent()
            if cnn is not None and getattr(cnn, "model", None):
                st.caption("CNN fallback: model loaded")
            else:
                st.caption("CNN fallback: unavailable (cloud mode)")
        except Exception:
            st.caption("CNN fallback: unavailable")

        st.info(
            "Sample images are **synthetic, watermarked demo assets** — not real "
            "patient data.")

# ── TAB 10: TNM Staging ───────────────────────────────────────────
with tab10:
    st.markdown("## 📍 TNM Cancer Staging")
    st.markdown(
        "AJCC/UICC 8th Edition TNM staging from TP53 mutation profile + "
        "pathology slide findings. Includes Kenya-contextualised clinical roadmap."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Input")
        tnm_mutation = st.text_input("Primary mutation:", value="R175H", key="tnm_mutation")
        tnm_cancer = st.selectbox(
            "Cancer type:",
            ["Colorectal", "Breast", "Lung", "Liver", "Ovarian", "Gastric"],
            key="tnm_cancer"
        )
        tnm_vaf = st.number_input("VAF (%):", 0.0, 100.0, 47.3, key="tnm_vaf")
        tnm_patient_id = st.text_input(
            "Patient ID (will be hashed):", value="DEMO-KE-001", key="tnm_patient"
        )
        use_pathology = st.checkbox(
            "Use pathology results from Pathology tab",
            value=bool(st.session_state.last_pathology_result),
        )

        if st.button("📍 Run TNM Staging", width="stretch"):
            try:
                from agents.tnm_staging import TNMStagingAgent
                pipeline_input = {
                    "mutations": [{"amino_acid_change": tnm_mutation, "position": 0}],
                    "cancer_type": tnm_cancer,
                    "vaf": tnm_vaf,
                    "accession": "NM_000546",
                    "patient_id": tnm_patient_id,
                }
                pathology_input = (
                    st.session_state.last_pathology_result
                    if use_pathology and st.session_state.last_pathology_result
                    else {
                        "success": False,
                        "top_tissue": "Unknown",
                        "tissue_classifications": [],
                        "mutation_correlations": [],
                    }
                )
                with st.spinner("Running TNM staging..."):
                    tnm_agent = TNMStagingAgent(rag_chain=st.session_state.rag)
                    result = tnm_agent.stage(
                        pathology_result=pathology_input,
                        pipeline_data=pipeline_input,
                    )
                    st.session_state.tnm_result = result
            except ImportError:
                st.error("TNM agent not found. Ensure agents/tnm_staging.py is in place.")
            except Exception as e:
                st.error(f"TNM staging failed: {e}")

    with col2:
        st.markdown("### About TNM Staging")
        st.markdown("""
**T — Primary Tumour**
Size and extent of invasion into local tissues.

**N — Regional Lymph Nodes**
Absence or presence of regional lymph node metastasis.

**M — Distant Metastasis**
Absence or presence of distant metastasis.

**Stage Groups**
- Stage I — localised, curative intent
- Stage II — local extension, surgery ± adjuvant
- Stage III — regional spread, multimodal therapy
- Stage IV — distant metastasis, palliative focus

*Note: Definitive staging requires CT imaging and/or
lymph node biopsy. This is AI-assisted clinical support,
not a replacement for pathological staging.*
""")

    if st.session_state.tnm_result:
        result = st.session_state.tnm_result
        st.divider()

        stage = result.get("stage_group", "?")
        t_code = result.get("T", {}).get("code", "?")
        n_code = result.get("N", {}).get("code", "?")
        m_code = result.get("M", {}).get("code", "?")

        stage_colors = {
            "I": "🟢", "IIA": "🟡", "IIB": "🟡",
            "IIIA": "🟠", "IIIB": "🟠", "IIIC": "🔴", "IV": "🔴"
        }
        badge = stage_colors.get(stage, "⚪")
        st.markdown(f"## {badge} Stage **{stage}** — {t_code} {n_code} {m_code}")
        st.plotly_chart(tnm_stage_bar(stage), width="stretch")

        equity = result.get("equity_flag")
        if equity:
            st.warning(equity)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**{t_code}**")
            st.caption(result.get("T", {}).get("description", ""))
        with c2:
            st.markdown(f"**{n_code}**")
            st.caption(result.get("N", {}).get("description", ""))
        with c3:
            st.markdown(f"**{m_code}**")
            st.caption(result.get("M", {}).get("description", ""))

        st.divider()
        st.markdown("### 🧠 Clinical Narration")
        st.markdown(result.get("llm_narration", ""))

        st.divider()
        st.markdown("### 📋 Clinical Roadmap (Kenya-Contextualised)")
        for step in result.get("next_steps", []):
            with st.expander(
                f"**{step['priority']}.** {step['action']} — {step.get('timeframe', '')}",
                expanded=(step["priority"] <= 2)
            ):
                st.markdown(step.get("detail", ""))
                st.markdown(f"🏥 **Kenya Resource:** {step.get('kenya_resource', '')}")

        st.divider()
        col_img, col_mdt = st.columns(2)
        with col_img:
            st.markdown("### 🖥 Imaging Workup")
            for img in result.get("imaging_workup", []):
                st.markdown(f"- {img}")
        with col_mdt:
            st.markdown("### 👥 MDT Referrals")
            for ref in result.get("mdt_referrals", []):
                st.markdown(f"- {ref}")

        st.divider()
        st.download_button(
            "⬇️ Download TNM Report (JSON)",
            data=stamp_json(result),
            file_name=f"tnm_{result.get('mutation', 'unknown')}_{result.get('stage_group', 'unknown')}.json",
            mime="application/json",
        )
        if st.checkbox("Show FHIR R4 ClinicalImpression resource"):
            st.json(stamp_fhir(result.get("fhir_resource", {})))

# ── TAB 11: African TP53 Atlas ────────────────────────────────────
with tab11:
    st.markdown("## 🌍 African TP53 Atlas")
    st.markdown(
        "Regional TP53 cancer-genomics for African populations — which mutations "
        "dominate which regions, in which cancers, driven by which exposures. "
        "Complements the African Drift bias-detector."
    )

    # ── Command Center: continental decision-support snapshot ──
    with st.expander("🌍 Oncology Command Center (continental snapshot)",
                     expanded=True):
        try:
            from agents.command_center import command_center_snapshot
            from utils.viz import command_center_html
            components.html(command_center_html(_cached_command_center()),
                            height=600, scrolling=True)
        except Exception as e:
            st.caption(f"Command center unavailable: {str(e)[:120]}")

    try:
        from agents.african_atlas import AfricanTP53Atlas
        _atlas = AfricanTP53Atlas()

        # ── Continental map (always shown) ──
        st.plotly_chart(african_atlas_map(_atlas.country_burden()),
                        width="stretch")

        st.divider()
        st.markdown("### 🔎 Explore by mutation, region or cancer")
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            atlas_mut = st.text_input("Mutation:", value="R249S", key="atlas_mut")
        with ac2:
            atlas_region = st.text_input("Region / country:", value="", key="atlas_region")
        with ac3:
            atlas_cancer = st.text_input("Cancer type:", value="", key="atlas_cancer")

        out = _atlas.profile(
            mutation=atlas_mut or None,
            region=atlas_region or None,
            cancer_type=atlas_cancer or None,
        )
        atlas = out["atlas"]
        if out["broadened"]:
            st.info("No exact match — showing the full continental atlas.")

        mcols = st.columns([3, 2])
        with mcols[0]:
            st.plotly_chart(african_burden_bar(atlas["matched_profiles"]),
                            width="stretch")
        with mcols[1]:
            st.metric("Profiles matched", len(atlas["matched_profiles"]))
            if atlas["key_mutations"]:
                st.markdown("**Key mutations:** " + ", ".join(atlas["key_mutations"]))
            if atlas["cancers"]:
                st.markdown("**Cancers:** " + "; ".join(atlas["cancers"]))

        for p in atlas["matched_profiles"]:
            with st.expander(f"📍 {p['title']}  ·  burden {p.get('burden_score', '?')}/100"):
                st.markdown(f"**Regions:** {', '.join(p['regions'])}")
                st.markdown(f"**Countries:** {', '.join(p['countries'])}")
                st.markdown(f"**Dominant cancer:** {p['dominant_cancer']}")
                st.markdown(f"**Key TP53 mutations:** "
                            f"{', '.join(p['key_mutations']) or '— (TP53 typically wild-type)'}")
                st.markdown(f"**Environmental driver:** {p['environmental_driver']}")
                st.markdown(f"**Prevalence (representative):** {p['representative_prevalence']}")
                st.markdown(f"🏥 **Kenya context:** {p['kenya_context']}")
                st.caption("Sources: " + "; ".join(p["sources"]))

        st.caption(f"⚠️ {atlas['disclaimer']}")
    except Exception as e:
        st.error(f"African Atlas unavailable: {str(e)[:200]}")

# ── TAB 12: Clinical Trials ───────────────────────────────────────
with tab12:
    st.markdown("## 🧪 Clinical Trials Matcher")
    st.markdown(
        "Find **recruiting** trials for a TP53 mutation + cancer, with "
        "**Kenyan / African sites prioritised** (then international trials "
        "open to African patients). Live ClinicalTrials.gov data."
    )

    tc1, tc2, tc3 = st.columns([2, 2, 1])
    with tc1:
        ct_mut = st.text_input("Mutation:", value="R175H", key="ct_mut")
    with tc2:
        ct_cancer = st.text_input("Cancer type:", value="breast cancer", key="ct_cancer")
    with tc3:
        ct_live = st.checkbox("Live API", value=True, key="ct_live")

    cta, ctb = st.columns(2)
    _do_find = cta.button("🔎 Find Trials", width="stretch", key="ct_go")
    _do_counter = ctb.button("⚖️ Counterfactual viability", width="stretch",
                             key="ct_counter",
                             help="Also searches for trials that were STOPPED "
                                  "early for this variant — a safety-first "
                                  "counterfactual, not just positive matches.")
    if _do_counter:
        from agents.adversarial_evidence import counterfactual_trials
        with st.spinner("Searching matching AND stopped trials…"):
            cf = counterfactual_trials(ct_mut or "", ct_cancer or "")
        if cf.get("success"):
            v = cf["viability"]
            vc = "🟢" if v >= 0.6 else ("🟠" if v >= 0.35 else "🔴")
            k1, k2, k3 = st.columns(3)
            k1.metric("Matching (recruiting)", cf["positive_count"])
            k2.metric("Stopped early", cf["stopped_count"])
            k3.metric("Viability", f"{vc} {v*100:.0f}%")
            if cf["stopped_trials"]:
                st.markdown("**Trials stopped early for this variant/condition:**")
                for t in cf["stopped_trials"]:
                    st.markdown(f"- `{t.get('nct_id','?')}` — {t.get('status')} "
                                f"· {t.get('title','')[:80]}")
            st.caption(f"⚠️ {cf['disclaimer']}")
        else:
            st.warning(f"Counterfactual search failed: {cf.get('reason','')[:120]}")

    if _do_find:
        try:
            from agents.clinical_trials import ClinicalTrialsMatcher
            with st.spinner("Searching ClinicalTrials.gov..."):
                res = ClinicalTrialsMatcher().search(
                    mutation=ct_mut or None, cancer_type=ct_cancer or None,
                    use_live=ct_live)

            if res["live"]:
                st.success(f"🟢 {res['message']}")
            else:
                st.info(f"⚪ Live API unavailable — showing curated search pointers. "
                        f"{res['message']}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Trials", res["count"])
            m2.metric("🌍 African sites", res["african_count"])
            m3.metric("🇰🇪 Kenyan sites", res["kenya_count"])

            st.plotly_chart(trials_priority_chart(res["trials"]),
                            width="stretch")

            for t in res["trials"]:
                flag = "🇰🇪" if t.get("kenya_site") else ("🌍" if t.get("african_priority") else "🌐")
                header = f"{flag} {t.get('phase', 'N/A')} — {t['title'][:80]}"
                with st.expander(header):
                    if t.get("nct_id"):
                        st.markdown(f"**NCT ID:** `{t['nct_id']}` · **Status:** {t['status']}")
                    if t.get("conditions"):
                        st.markdown(f"**Conditions:** {', '.join(t['conditions'][:5])}")
                    if t.get("countries"):
                        st.markdown(f"**Sites:** {', '.join(t['countries'][:8])}")
                    if t.get("african_sites"):
                        st.markdown(f"🌍 **African sites:** {', '.join(t['african_sites'])}")
                    st.markdown(f"🔗 [Open on ClinicalTrials.gov]({t['url']})")
            st.caption(f"⚠️ {res['disclaimer']}")
        except Exception as e:
            st.error(f"Clinical Trials matcher unavailable: {str(e)[:200]}")

# ── Footer ────────────────────────────────────────────────────────
st.divider()
_fmode = os.getenv("INFERENCE_MODE", "ollama").lower()
_finfo = ("Gemma 4 · Google AI Studio (cloud)" if _fmode == "api"
          else "Gemma 4 · llama.cpp (local)" if _fmode == "llamacpp"
          else "Gemma 4 · Ollama (local)")
st.markdown(
    f"**Precision Onco Africa** | {_finfo} | "
    "Kenya/KEML clinical context | "
    "[GitHub](https://github.com/mbote-droid/tp53_analysis)"
)
st.caption("⚠️ Research & educational use only — not for clinical decisions. "
           "Some visuals (e.g. docking affinities) are illustrative, not measured.")