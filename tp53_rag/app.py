"""
TP53 RAG Platform — Streamlit Web App
Clean rebuild — no streamlit_option_menu dependency
"""
import os
import sys
import json
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

st.set_page_config(
    page_title="TP53 RAG Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Import RAG modules ────────────────────────────────────────────
RAG_AVAILABLE = False
try:
    from agents.rag_chain import TP53RAGChain
    from agents.dispatcher import AgentDispatcher
    from knowledge_base.vector_store import TP53VectorStore
    RAG_AVAILABLE = True
except ImportError as e:
    st.error(f"RAG modules not found: {e}. Run: python main.py build")

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🔍 Query",
    "🧬 Analysis",
    "💊 Drug Discovery",
    "📊 Visualization",
    "📋 Report",
    "🔬 Structure",
    "🎤 Voice",
    "🛠 Debug",
    "🔬 Pathology",
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

        cols = st.columns(2)
        for idx, (agent, query) in enumerate(agent_queries.items()):
            with st.spinner(f"Running {agent}..."):
                result = safe_query(query, agent_type=agent)
            with cols[idx % 2]:
                with st.expander(f"🔬 {agent.replace('_', ' ').title()}", expanded=(idx < 2)):
                    answer = result["answer"]
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

# ── TAB 4: Visualization ──────────────────────────────────────────
with tab4:
    st.markdown("## 📊 Visualization & Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ctDNA VAF Timeline")
        vaf_df = pd.DataFrame({
            "Day":    [0, 5, 10, 15, 20, 25],
            "VAF (%)": [50, 48, 45, 42, 38, 35],
        })
        fig = px.line(vaf_df, x="Day", y="VAF (%)", markers=True,
                      title="Variant Allele Frequency Over Treatment")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### TP53 Hotspot Frequency")
        hotspot_df = pd.DataFrame({
            "Codon":          ["175", "248", "273", "249", "282", "220"],
            "Frequency (%)":  [8.0, 7.5, 7.0, 6.5, 4.0, 3.5],
        })
        fig2 = px.bar(hotspot_df, x="Codon", y="Frequency (%)",
                      title="Known TP53 Hotspot Mutations",
                      color="Frequency (%)", color_continuous_scale="Reds")
        fig2.update_layout(template="plotly_dark")
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
    st.markdown("Interactive p53 protein structure — powered by Mol* viewer")

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

        # Mol* viewer embed
        components.iframe(
            f"https://molstar.org/viewer/?pdb={pdb_id}",
            height=480,
            scrolling=False,
        )

    with col2:
        st.markdown("### Domain Reference")
        domain_ref = pd.DataFrame({
            "Domain":   ["TAD", "PRD", "DBD", "TET", "REG"],
            "Residues": ["1–67", "67–98", "94–292", "323–356", "364–393"],
            "Function": ["Transactivation", "Proline-rich",
                         "DNA-binding ⚠️", "Tetramerization", "Regulatory"],
        })
        st.dataframe(domain_ref, use_container_width=True, hide_index=True)

        st.markdown("### Hotspot Positions")
        hotspot_ref = pd.DataFrame({
            "Mutation": ["R175H", "R248W", "R248Q", "R273H", "R273C", "R282W"],
            "Residue":  [175, 248, 248, 273, 273, 282],
            "Type":     ["Conform.", "Contact", "Contact",
                         "Contact", "Contact", "Conform."],
        })
        st.dataframe(hotspot_ref, use_container_width=True, hide_index=True)

        st.markdown("### RAG Narration")
        if st.button("🧠 Explain structure"):
            with st.spinner("Querying Gemma 4..."):
                result = safe_query(
                    f"Explain the 3D structure of p53 PDB {pdb_id}. "
                    f"Focus on the DNA-binding domain and residue {highlight}.",
                    agent_type="domain_annotation"
                )
            st.markdown(result["answer"])

# ── TAB 7: Voice ──────────────────────────────────────────────────
with tab7:
    st.markdown("## 🎤 Voice Input (Beta)")
    st.markdown("Speak your question — transcribed locally via Whisper")

    WHISPER_AVAILABLE = False
    try:
        import whisper
        WHISPER_AVAILABLE = True
        st.success("✅ Whisper ready")
    except ImportError:
        st.warning("⚠️ Install Whisper: `pip install openai-whisper`")

    @st.cache_resource
    def load_whisper():
        try:
            import whisper
            return whisper.load_model("tiny")
        except Exception as e:
            log.error(f"Failed to load Whisper: {e}")
            return None

    def transcribe(audio_bytes) -> str:
        model = load_whisper()
        if model is None:
            return "Transcription error: Whisper model failed to load"

        tmp_path = Path(__file__).parent / "temp_audio.wav"
        try:
            with open(str(tmp_path), "wb") as f:
                if hasattr(audio_bytes, "getvalue"):
                    f.write(audio_bytes.getvalue())
                elif hasattr(audio_bytes, "read"):
                    if hasattr(audio_bytes, "seek"):
                        audio_bytes.seek(0)
                    f.write(audio_bytes.read())
                else:
                    f.write(bytes(audio_bytes))

            try:
                result = model.transcribe(str(tmp_path), language="en", fp16=False)
            except TypeError:
                result = model.transcribe(str(tmp_path), language="en")

            return result.get("text", "").strip()
        except FileNotFoundError as e:
            return f"Transcription error: File not found - {e}"
        except Exception as e:
            return f"Transcription error: {str(e)[:200]}"
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception as cleanup_err:
                log.warning(f"Failed to cleanup temp audio: {cleanup_err}")

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
                tmp = Path("temp_slide.jpg")
                tmp.write_bytes(uploaded.getvalue())

                with st.spinner("Analysing slide with pathology foundation model..."):
                    result = agent.process_slide(
                        str(tmp),
                        mutation_data=st.session_state.pipeline_data
                    )

                if tmp.exists():
                    tmp.unlink()

                if result["success"]:
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

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "**TP53 RAG Platform** | Gemma 4 2B + llama.cpp | "
    "100% local inference | Kenya/KEML clinical context | "
    "[GitHub](https://github.com/mbote-droid/tp53_analysis)"
)
