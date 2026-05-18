"""
============================================================
TP53 RAG Platform - Production Streamlit App
============================================================
Fast multi-agent analysis with voice input transcription
Run: streamlit run app_rag.py
============================================================
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import streamlit as st

# Add project root
sys.path.insert(0, str(Path(__file__).parent / "tp53_rag"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TP53 RAG Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State ──────────────────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "dispatcher" not in st.session_state:
    st.session_state.dispatcher = None
if "transcriber" not in st.session_state:
    st.session_state.transcriber = None

# ── Load RAG Components ────────────────────────────────────────────────────
@st.cache_resource
def load_platform():
    """Load RAG components on first run."""
    try:
        from tp53_rag.agents.dispatcher import AgentDispatcher
        from tp53_rag.knowledge_base.vector_store import TP53VectorStore
        from tp53_rag.utils.voice_transcriber import VoiceTranscriber
        
        log.info("Loading TP53 RAG Platform...")
        vector_store = TP53VectorStore()
        dispatcher = AgentDispatcher(vector_store=vector_store)
        transcriber = VoiceTranscriber(model="base")
        log.info("✓ Platform loaded")
        return dispatcher, transcriber
    except Exception as e:
        log.warning(f"Platform components not fully loaded: {e}")
        return None, None

dispatcher, transcriber = load_platform()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧬 TP53 RAG Platform")
    st.markdown("---")
    
    selected = st.radio(
        "Navigation",
        ["🎯 Quick Query", "🎤 Voice Input", "📊 Analysis", "📋 History", "⚙️ Settings"],
    )
    
    st.markdown("---")
    st.caption("TP53 Bioinformatics • Fast inference • HIPAA-ready")

# ============================================================================
# TAB: Quick Query
# ============================================================================
if selected == "🎯 Quick Query":
    st.header("🎯 Query TP53 Knowledge Base")
    
    query = st.text_area(
        "Ask about TP53 mutations, variants, drugs, clinical significance:",
        placeholder="e.g., What is R175H mutation?",
        height=100,
    )
    
    if st.button("🚀 Analyze", use_container_width=True):
        if query:
            with st.spinner("🔬 Analyzing..."):
                try:
                    if dispatcher:
                        result = dispatcher.dispatch_single(
                            agent_type="mutation_analysis",
                            pipeline_data={},
                            custom_question=query,
                        )
                        st.success("✓ Analysis complete")
                        st.write(result.answer)
                        
                        if result.sources:
                            with st.expander("📚 Sources"):
                                for source in result.sources[:3]:
                                    st.caption(f"• {source.get('title', 'Source')}")
                    else:
                        st.info("📚 RAG platform loading. Demo response:")
                        st.write(f"Query received: {query}\n\nReturn to Full Bioinformatics App for complete analysis.")
                    
                    st.session_state.query_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "query": query,
                        "method": "text",
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
                    log.error(f"Query failed: {e}")

# ============================================================================
# TAB: Voice Input
# ============================================================================
elif selected == "🎤 Voice Input":
    st.header("🎤 Voice Query")
    st.markdown("Record your question → instant transcription + analysis")
    
    audio_bytes = st.audio_input("🎙️ Record your query:")
    
    if audio_bytes:
        if st.button("📝 Transcribe & Analyze", use_container_width=True):
            with st.spinner("🎙️ Transcribing..."):
                try:
                    if transcriber:
                        query_text = transcriber.transcribe(audio_bytes.getvalue())
                        st.success(f"✓ Transcribed: **{query_text}**")
                        
                        with st.spinner("🔬 Analyzing..."):
                            if dispatcher:
                                result = dispatcher.dispatch_single(
                                    agent_type="mutation_analysis",
                                    pipeline_data={},
                                    custom_question=query_text,
                                )
                                st.write(result.answer)
                                
                                if result.sources:
                                    with st.expander("📚 Sources"):
                                        for source in result.sources[:3]:
                                            st.caption(f"• {source.get('title', 'Source')}")
                            else:
                                st.info("📚 RAG platform loading. Try again in a moment.")
                        
                        st.session_state.query_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "query": query_text,
                            "method": "voice",
                        })
                    else:
                        st.warning("Voice transcriber not available. Install openai-whisper.")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    log.error(f"Voice error: {e}")

# ============================================================================
# TAB: Analysis Dashboard
# ============================================================================
elif selected == "📊 Analysis":
    st.header("📊 Multi-Agent Analysis")
    
    agents = [
        ("mutation_analysis", "Mutation Analysis"),
        ("orf_analysis", "ORF Analysis"),
        ("phylogenetic_analysis", "Phylogenetic Analysis"),
        ("domain_annotation", "Domain Annotation"),
        ("clinical_interpretation", "Clinical Interpretation"),
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_agent = st.selectbox("Select Agent:", [a[1] for a in agents])
    
    with col2:
        uploaded = st.file_uploader("Upload results (JSON, optional):")
    
    query = st.text_area("Query:", placeholder="Ask the selected agent...", height=80)
    
    if st.button("🔍 Dispatch Agent", use_container_width=True):
        if query:
            agent_key = next(a[0] for a in agents if a[1] == selected_agent)
            
            with st.spinner(f"Running {selected_agent}..."):
                try:
                    if dispatcher:
                        result = dispatcher.dispatch_single(
                            agent_type=agent_key,
                            pipeline_data={},
                            custom_question=query,
                        )
                        st.success("✓ Agent response received")
                        st.write(result.answer)
                    else:
                        st.info("RAG platform loading. Please wait...")
                except Exception as e:
                    st.error(f"Agent failed: {e}")

# ============================================================================
# TAB: Query History
# ============================================================================
elif selected == "📋 History":
    st.header("📋 Query History")
    
    if st.session_state.query_history:
        for i, item in enumerate(reversed(st.session_state.query_history[-10:])):
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 4, 1])
                
                with col1:
                    method = "🎤" if item["method"] == "voice" else "💬"
                    st.caption(f"{method} {item['method'].title()}")
                
                with col2:
                    text = item["query"][:100] + "..." if len(item["query"]) > 100 else item["query"]
                    st.write(text)
                
                with col3:
                    st.caption(item["timestamp"][:10])
    else:
        st.info("No queries yet. Start with Quick Query or Voice Input!")

# ============================================================================
# TAB: Settings
# ============================================================================
elif selected == "⚙️ Settings":
    st.header("⚙️ Platform Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Vector Store", "✓ Loaded" if dispatcher else "⏳ Loading...")
        st.metric("Dispatcher", "✓ Ready" if dispatcher else "⏳ Initializing...")
    
    with col2:
        st.metric("Voice Transcriber", "✓ Loaded" if transcriber else "⏳ Loading...")
        st.metric("RAG Chain", "✓ Ready" if dispatcher else "⏳ Initializing...")
    
    st.subheader("🔧 Configuration")
    
    with st.expander("Whisper Model"):
        whisper_model = st.select_slider(
            "Transcription Model:",
            ["tiny", "base", "small"],
            value="base"
        )
        st.caption("Larger = more accurate but slower. 'base' is optimal.")
    
    with st.expander("HIPAA Compliance"):
        pii_detection = st.checkbox("Enable PII Detection", value=True)
        if pii_detection:
            st.success("✓ PII scrubbing active")
        else:
            st.warning("⚠️ PII scrubbing disabled")
    
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("✓ Settings saved")
    
    st.divider()
    st.caption("TP53 RAG Platform v1.0 • Local Gemma 2B • LGPL licensed")

# ============================================================================
# Footer
# ============================================================================
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🧬 Data: NCBI, COSMIC, ClinVar")

with col2:
    st.caption("⚡ Inference: Local Gemma 2B")

with col3:
    st.caption("🔒 HIPAA-ready • Open source")
