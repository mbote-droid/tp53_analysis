"""
TP53 RAG Platform — Streamlit Web App
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

st.set_page_config(
    page_title="TP53 RAG Platform",
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

        /* ── Tabs: never let 12 tabs overflow — scroll horizontally ── */
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            scrollbar-width: thin;
        }
        .stTabs [data-baseweb="tab"] { white-space: nowrap; }

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
            "answer": "RAG system offline. Run: python main.py build",
            "agent_used": "offline",
            "sources": [],
            "cache_hit": False,
            "retries": 0,
        }
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
        return result
    except Exception as e:
        return {
            "answer": f"Query error: {str(e)[:300]}",
            "agent_used": agent_type or "error",
            "sources": [],
            "cache_hit": False,
            "retries": 0,
        }

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
    st.markdown("## 🧬 TP53 RAG Platform")
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
(tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11,
 tab12) = st.tabs([
    "🔍 Query",
    "🧬 Analysis",
    "💊 Drug Discovery",
    "📊 Visualization",
    "📋 Report",
    "🔬 Structure",
    "🎤 Voice",
    "🛠 Debug",
    "🔬 Pathology",
    "📍 TNM Staging",
    "🌍 African Atlas",
    "🧪 Clinical Trials",
])

# ── TAB 1: Query ──────────────────────────────────────────────────
with tab1:
    st.markdown("## 🔍 TP53 Knowledge Query")
    st.markdown("Ask any question about TP53 mutations, drug targets, or clinical significance.")

    question = st.text_area(
        "Your question:",
        placeholder="e.g., What are the clinical implications of R175H mutation?",
        height=100,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("🚀 Ask Gemma 4", width="stretch")
    with col2:
        if st.button("🗑 Clear", width="stretch"):
            st.session_state.messages = []
            st.success("Cleared")

    if submit and question:
        with st.spinner("Querying Gemma 4 via RAG..."):
            result = safe_query(question, agent_type=forced_agent)

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

        st.markdown("### Answer")
        st.markdown(result["answer"])

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
        if vcf_bytes is None and use_sample:
            from utils.vcf_parser import sample_vcf
            vcf_bytes = sample_vcf().encode()
        if vcf_bytes:
            try:
                from utils.vcf_parser import parse_vcf_bytes
                from utils.viz import vcf_variant_chart
                parsed = parse_vcf_bytes(vcf_bytes)
                st.caption(f"**{parsed['tp53_count']}** TP53 variant(s) found "
                           f"· {parsed['skipped']} malformed line(s) skipped")
                if parsed["variants"]:
                    st.plotly_chart(vcf_variant_chart(parsed["variants"]),
                                    width="stretch")
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
                json.dumps({"mutation": mutation, "cancer": cancer}, indent=2),
                f"tp53_{mutation}_{datetime.now().strftime('%Y%m%d')}.json",
                "application/json",
            )

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
            result["answer"],
            file_name=f"report_{rep_mutation}_{datetime.now().strftime('%Y%m%d')}.md",
        )

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

    if st.session_state.rag:
        try:
            stats = st.session_state.rag.cache_stats()
            st.markdown("### Cache Stats")
            st.json(stats)
        except Exception:
            pass

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
    st.markdown("## 🔬 Pathology Slide Analysis")
    st.markdown(
        "Upload an H&E stained pathology slide — "
        "tissue classification powered by UNI/ResNet foundation model, "
        "correlated with your TP53 pipeline findings."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload H&E slide image (JPG, PNG, TIFF):",
            type=["jpg", "jpeg", "png", "tiff"]
        )

        if uploaded:
            st.image(uploaded, caption="Uploaded slide", width="stretch")

        if uploaded and st.button("🔬 Analyse Slide", width="stretch"):
            # FIXED: safe load with None check — won't crash on cloud if torch missing
            agent = load_pathology_agent()

            if agent is None:
                st.error(
                    "Pathology agent unavailable in this environment. "
                    "torch/torchvision not installed. Run locally for full functionality."
                )
            else:
                # NamedTemporaryFile avoids writing into the (read-only on
                # Streamlit Cloud) working directory.
                suffix = Path(uploaded.name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as _tf:
                    _tf.write(uploaded.getvalue())
                    tmp = Path(_tf.name)

                try:
                    with st.spinner("Analysing slide with pathology foundation model..."):
                        result = agent.process_slide(
                            str(tmp),
                            mutation_data=st.session_state.pipeline_data
                        )
                finally:
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception as cleanup_err:
                        log.warning(f"Failed to cleanup temp slide: {cleanup_err}")

                if result["success"]:
                    st.session_state.last_pathology_result = result
                    st.success(f"✅ Top tissue: **{result['top_tissue']}**")
                    st.divider()

                    st.markdown("### Tissue Classifications")
                    tissue_df = pd.DataFrame(result["tissue_classifications"])
                    st.dataframe(tissue_df, width="stretch", hide_index=True)

                    if result["mutation_correlations"]:
                        st.markdown("### TP53 Mutation Correlation")
                        for corr in result["mutation_correlations"]:
                            st.markdown(f"**{corr['mutation']}** cancer associations:")
                            for cancer, freq in corr["cancer_correlations"].items():
                                st.progress(freq, text=f"{cancer}: {freq:.0%}")

                    st.divider()
                    st.markdown("### 🧠 Gemma 4 Clinical Narration")
                    st.markdown(result["llm_narration"])

                    st.download_button(
                        "⬇️ Download Report",
                        data=result["llm_narration"],
                        file_name="pathology_report.md",
                        mime="text/markdown"
                    )
                else:
                    st.error(f"Analysis failed: {result.get('error')}")

        if not uploaded:
            st.info("Upload an H&E slide image to begin")
            st.markdown("""
**Free slide datasets:**
- [TCGA Portal](https://portal.gdc.cancer.gov)
- [Kaggle Histopathology](https://www.kaggle.com/datasets?search=histopathology)
""")

    with col2:
        st.markdown("### About This Agent")
        st.markdown("""
**Model:** UNI (Harvard/MGH) or ResNet fallback

**Tissue classes detected:**
- Tumor
- Stroma
- Inflammatory
- Necrosis
- Normal epithelium
- Mucus
- Smooth muscle
- Adipose

**TP53 correlations:**
- R248W → Colorectal (90%)
- R273H → Colorectal (88%)
- R175H → Breast (85%)
- R249S → Liver (95%)
- G245S → Sarcoma (80%)
""")

        st.markdown("### Model Status")
        # FIXED: safe check — no crash if torch missing on cloud
        try:
            agent_check = load_pathology_agent()
            if agent_check is not None and agent_check.model:
                st.success("✅ Model loaded")
            elif agent_check is None:
                st.warning("⚠️ Pathology agent unavailable (cloud mode)")
            else:
                st.error("❌ No model")
        except Exception as e:
            st.error(f"❌ {e}")

        st.markdown("### Get UNI Access")
        st.markdown(
            "For best results request UNI model access: "
            "[HuggingFace MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)\n\n"
            "Set your token in `.env`: `HF_TOKEN=your_token`"
        )

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
            data=json.dumps(result, indent=2),
            file_name=f"tnm_{result.get('mutation', 'unknown')}_{result.get('stage_group', 'unknown')}.json",
            mime="application/json",
        )
        if st.checkbox("Show FHIR R4 ClinicalImpression resource"):
            st.json(result.get("fhir_resource", {}))

# ── TAB 11: African TP53 Atlas ────────────────────────────────────
with tab11:
    st.markdown("## 🌍 African TP53 Atlas")
    st.markdown(
        "Regional TP53 cancer-genomics for African populations — which mutations "
        "dominate which regions, in which cancers, driven by which exposures. "
        "Complements the African Drift bias-detector."
    )

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

    if st.button("🔎 Find Trials", width="stretch", key="ct_go"):
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
    f"**TP53 RAG Platform** | {_finfo} | "
    "Kenya/KEML clinical context | "
    "[GitHub](https://github.com/mbote-droid/tp53_analysis)"
)
st.caption("⚠️ Research & educational use only — not for clinical decisions. "
           "Some visuals (e.g. docking affinities) are illustrative, not measured.")