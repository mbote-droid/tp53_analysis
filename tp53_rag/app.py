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
    domain_legend_chart,
    parse_residues,
    protein_viewer_html,
    dock_candidates,
    docking_affinity_chart,
    docking_pose_html,
    tnm_stage_bar,
)

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
@st.cache_resource
def init_rag_system():
    if not RAG_AVAILABLE:
        return None, None
    try:
        store = TP53VectorStore()
        if not store.is_built():
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
    try:
        # Cap concurrent inferences so parallel sessions can't exhaust the
        # 8GB RAM budget. Acquire/release is handled by the context manager.
        with _inference_lock:
            return st.session_state.rag.query(
                question=question,
                pipeline_data=st.session_state.pipeline_data or None,
                agent_type=agent_type,
            )
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
    st.markdown("""
**Model:** Gemma 4 2B (Q4_K_M)  
**Backend:** llama.cpp CPU  
**RAM:** ~4GB local inference  
**Privacy:** 100% local — no cloud
""")

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
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
        submit = st.button("🚀 Ask Gemma 4", use_container_width=True)
    with col2:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.success("Cleared")

    if submit and question:
        with st.spinner("Querying Gemma 4 via RAG..."):
            result = safe_query(question, agent_type=forced_agent)

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

        st.markdown("### Answer")
        st.markdown(result["answer"])
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

    mutation = st.text_input("TP53 Mutation:", placeholder="e.g., R175H", value="R175H")
    cancer = st.selectbox("Cancer type:", ["Colorectal", "Breast", "Ovarian", "Lung", "Gastric"])
    vaf = st.number_input("Variant Allele Frequency (%):", 0.0, 100.0, 50.0)

    if st.button("🧬 Run Multi-Agent Analysis", use_container_width=True):
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

    if st.button("🔍 Find Therapeutic Targets", use_container_width=True):
        with st.spinner("Searching drug databases..."):
            result = safe_query(
                f"Best drug candidates for {mut_input}? Focus on mechanism and Kenya/KEML availability.",
                agent_type="drug_discovery"
            )
        st.markdown(result["answer"])

    st.markdown("### Known TP53-Targeted Drugs")
    drug_df = pd.DataFrame({
        "Drug":           ["APR-246", "Idasanutlin", "Carboplatin", "Vorinostat"],
        "Mechanism":      ["p53 refolding", "MDM2 inhibitor", "DNA cross-link", "HDAC inhibitor"],
        "Clinical Stage": ["Phase III", "Phase II", "Approved", "Approved"],
        "KEML Available": ["Yes", "Limited", "Yes", "Limited"],
    })
    st.dataframe(drug_df, use_container_width=True, hide_index=True)

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
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    dcol1, dcol2 = st.columns([3, 2])
    with dcol1:
        st.plotly_chart(docking_affinity_chart(candidates), use_container_width=True)
    with dcol2:
        top = candidates[0]
        st.markdown(f"**Top candidate:** {top['name']}  \nΔG ≈ **{top['affinity']} kcal/mol**")
        pocket = parse_residues(mut_input)  # mutation residue + canonical hotspots
        components.html(
            docking_pose_html("2OCJ", pocket, top["name"], top["affinity"]),
            height=480,
        )
        st.caption("Yellow cloud = proposed binding pocket on p53 (illustrative).")

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
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### TP53 Hotspot Frequency")
        st.caption("Press ▶ Play to watch the frequencies build up.")
        fig2 = animated_hotspot_bar(
            codons=["175", "248", "273", "249", "282", "220"],
            freqs=[8.0, 7.5, 7.0, 6.5, 4.0, 3.5],
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### TP53 Protein Domain Map")
    domain_df = pd.DataFrame({
        "Domain":   ["TAD1", "TAD2", "PRD", "DBD", "NLS", "TET", "REG"],
        "Start":    [1, 40, 67, 94, 316, 323, 364],
        "End":      [40, 67, 98, 292, 325, 356, 393],
        "Function": ["Transactivation", "Transactivation", "Proline-rich",
                     "DNA-binding (hotspot region)", "Nuclear signal",
                     "Tetramerization", "Regulatory"],
    })
    st.dataframe(domain_df, use_container_width=True, hide_index=True)

    st.markdown("### Multi-Agent Dispatch Network")
    st.caption("Press ▶ Trace dispatch to watch the orchestrator fan out to each agent.")
    arch_fig = agent_architecture_diagram([
        "mutation_analysis", "drug_discovery", "clinical_interpretation",
        "liquid_biopsy", "gene_expression", "domain_annotation",
        "pathology_vision", "tnm_staging", "variant_curator", "immunogenicity",
    ])
    st.plotly_chart(
        arch_fig, use_container_width=True,
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

    if st.button("📋 Generate Clinical Report", use_container_width=True):
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
            use_container_width=True, hide_index=True,
        )
        st.caption("🔴 Red spheres = hotspot residues (175 / 248 / 273) + your selection.")

        st.markdown("### Hotspot Positions")
        hotspot_ref = pd.DataFrame({
            "Mutation": ["R175H", "R248W", "R248Q", "R273H", "R273C", "R282W"],
            "Residue":  [175, 248, 248, 273, 273, 282],
            "Type":     ["Conform.", "Contact", "Contact",
                         "Contact", "Contact", "Conform."],
        })
        st.dataframe(hotspot_ref, use_container_width=True, hide_index=True)

    # ── Domain map (side chart, colours match the 3D structure) ──
    st.markdown("### Structure Colour Map")
    st.plotly_chart(domain_legend_chart(), use_container_width=True)

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
    if st.button("🚀 Submit", use_container_width=True):
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
            st.image(uploaded, caption="Uploaded slide", use_container_width=True)

        if uploaded and st.button("🔬 Analyse Slide", use_container_width=True):
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
                    st.dataframe(tissue_df, use_container_width=True, hide_index=True)

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

        if st.button("📍 Run TNM Staging", use_container_width=True):
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
        st.plotly_chart(tnm_stage_bar(stage), use_container_width=True)

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

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "**TP53 RAG Platform** | Gemma 4 2B + llama.cpp | "
    "100% local inference | Kenya/KEML clinical context | "
    "[GitHub](https://github.com/mbote-droid/tp53_analysis)"
)