"""
Streamlit Web App — TP53 Bioinformatics Analysis Pipeline
==========================================================
A browser-based interface for the TP53 genomics pipeline.
Allows users to run the full analysis without any CLI knowledge.

Run locally:
    streamlit run app.py

Deploy free:
    https://streamlit.io/cloud
"""

import os
import sys
import warnings
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio.Seq import Seq

# Suppress BioPython partial codon warning in the UI
warnings.filterwarnings("ignore", category=UserWarning, module="Bio")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialise pipeline logger before importing functions
import main_tp53_analysis as _pipeline
import logging as _logging
_pipeline.logger = _logging.getLogger("TP53Pipeline")
_pipeline.logger.setLevel(_logging.DEBUG)
_pipeline.logger.addHandler(_logging.NullHandler())

from main_tp53_analysis import (
    fetch_sequence,
    analyze_protein,
    find_mutation_positions,
    run_alignment,
    find_orfs,
    codon_usage,
    amino_acid_frequency,
    fetch_multiple_sequences,
    build_multiple_alignment,
    build_phylogenetic_tree,
    save_phylogenetic_tree,
    export_distance_matrix_csv,
    annotate_protein_domains,
    export_domains_csv,
    export_mutations_csv,
    export_orfs_csv,
    plot_all,
    TP53_HOMOLOGS,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TP53 Bioinformatics Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0D1117;
    color: #E6EDF3;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}

code, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
}

.metric-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 6px 0;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #2196F3;
    font-family: 'JetBrains Mono', monospace;
}

.metric-label {
    font-size: 0.75rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.domain-card {
    background: #161B22;
    border-left: 3px solid #4CAF50;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin: 4px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}

.domain-db {
    color: #4CAF50;
    font-weight: 700;
    font-size: 0.7rem;
}

.mutation-badge {
    background: #1C2128;
    border: 1px solid #F44336;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #F44336;
    display: inline-block;
    margin: 2px;
}

.orf-row {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 3px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}

.step-header {
    background: linear-gradient(90deg, #2196F3 0%, #0D47A1 100%);
    border-radius: 6px;
    padding: 8px 16px;
    margin: 12px 0 6px 0;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
}

.success-box {
    background: #0D2818;
    border: 1px solid #4CAF50;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #4CAF50;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

.warning-box {
    background: #1A1200;
    border: 1px solid #FF9800;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #FF9800;
    font-size: 0.85rem;
}

.sidebar-info {
    background: #161B22;
    border-radius: 8px;
    padding: 12px;
    font-size: 0.8rem;
    color: #8B949E;
    margin: 8px 0;
}

div[data-testid="stSidebar"] {
    background: #0D1117;
    border-right: 1px solid #30363D;
}

.stButton > button {
    background: linear-gradient(135deg, #2196F3, #0D47A1);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
}

.stButton > button:hover {
    opacity: 0.85;
}

.stTextInput > div > div > input,
.stSelectbox > div > div > select {
    background: #161B22;
    border: 1px solid #30363D;
    color: #E6EDF3;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Pipeline Config")
    st.markdown("---")

    email = st.text_input(
        "NCBI Email",
        value=os.environ.get("ENTREZ_EMAIL", ""),
        placeholder="your@email.com",
        help="Required by NCBI Entrez API. Never stored or shared."
    )

    accession = st.text_input(
        "NCBI Accession",
        value="NM_000546",
        help="Any NCBI nucleotide accession. Default: human TP53"
    )

    st.markdown("**Analysis Options**")
    run_phylo   = st.checkbox("Multi-species phylogenetic tree", value=True)
    run_domains = st.checkbox("Protein domain annotation", value=True)

    st.markdown("**Parameters**")
    mutation_window = st.slider("Mutation window (bp)", 10, 200, 50)
    gc_window       = st.slider("GC content window (bp)", 50, 300, 100)
    orf_min_length  = st.slider("Min ORF length (bp)", 50, 500, 100)

    st.markdown("---")
    run_button = st.button("🚀 Run Analysis", use_container_width=True)

    st.markdown("""
    <div class="sidebar-info">
    <b>About</b><br>
    Built by Samuel Mbote<br>
    General Surgery Resident<br>
    & Bioinformatics Developer<br><br>
    <a href="https://github.com/mbote-droid/tp53_analysis" 
       style="color:#2196F3;">GitHub Repository</a>
    </div>
    """, unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
# 🧬 TP53 Bioinformatics Pipeline
### Genomic Analysis for Tumour Suppressor Genes
""")
st.markdown("""
<div style="color:#8B949E; font-size:0.9rem; margin-bottom:24px;">
TP53 is mutated in ~50% of all human cancers. This pipeline fetches, analyzes, 
and visualizes the gene across species — from sequence to protein domains.
Enter your NCBI email in the sidebar and click <b>Run Analysis</b>.
</div>
""", unsafe_allow_html=True)

# ── Pre-run info cards ──────────────────────────────────────────────────────
if not run_button:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">9</div>
            <div class="metric-label">Pipeline Steps</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">6</div>
            <div class="metric-label">Reading Frames Scanned</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">5</div>
            <div class="metric-label">Species Compared</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">63</div>
            <div class="metric-label">Passing Unit Tests</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
    ⚡ <b>Quick start:</b> Set your NCBI email in the sidebar, then click 
    <b>Run Analysis</b>. Use <b>--skip phylo/domains</b> checkboxes for a 
    faster ~30 second run. Full analysis takes 2–5 minutes.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Validation ─────────────────────────────────────────────────────────────
if not email or "@" not in email:
    st.error("❌ Please enter a valid email address in the sidebar.")
    st.stop()

if not accession.strip():
    st.error("❌ Please enter an NCBI accession number.")
    st.stop()

clean_accession = accession.strip().replace(" ", "")
os.makedirs("results", exist_ok=True)


# ── Run Pipeline ───────────────────────────────────────────────────────────
st.markdown("---")

# Step 1 — Fetch
st.markdown('<div class="step-header">STEP 1 / 9 — Fetching Sequence from NCBI</div>',
            unsafe_allow_html=True)
with st.spinner(f"Fetching {clean_accession} from NCBI..."):
    record = fetch_sequence(clean_accession, email)

if not record:
    st.error(f"❌ Failed to fetch accession '{clean_accession}'. Check the accession ID and try again.")
    st.stop()

dna_seq = record.seq
st.markdown(f"""
<div class="success-box">
✓ Fetched: <b>{record.id}</b> | Length: <b>{len(dna_seq):,} bp</b>
</div>""", unsafe_allow_html=True)

# Step 2 — Translate
st.markdown('<div class="step-header">STEP 2 / 9 — Translating DNA to Protein</div>',
            unsafe_allow_html=True)
protein_seq = analyze_protein(dna_seq)
if protein_seq:
    st.markdown(f"""
    <div class="success-box">
    ✓ Protein translated: <b>{len(protein_seq)} amino acids</b>
    </div>""", unsafe_allow_html=True)
else:
    st.warning("⚠️ Translation produced no protein. Continuing with remaining steps.")

# Step 3 — Mutations
st.markdown('<div class="step-header">STEP 3 / 9 — Simulating & Detecting Mutations</div>',
            unsafe_allow_html=True)
compare_len = 1000
mutant_dna  = Seq("A" * mutation_window) + dna_seq[mutation_window:compare_len]
mutations   = find_mutation_positions(str(dna_seq[:compare_len]), str(mutant_dna))
export_mutations_csv(mutations)

st.markdown(f"""
<div class="success-box">
✓ Mutations detected: <b>{len(mutations)}</b> 
(first {mutation_window} bases replaced with 'A')
</div>""", unsafe_allow_html=True)

if mutations:
    with st.expander(f"View all {len(mutations)} mutations"):
        cols = st.columns(6)
        for i, m in enumerate(mutations):
            cols[i % 6].markdown(
                f'<span class="mutation-badge">'
                f'pos {m["position"]}: {m["original"]}→{m["mutant"]}'
                f'</span>',
                unsafe_allow_html=True
            )

# Step 4 — Alignment
st.markdown('<div class="step-header">STEP 4 / 9 — Pairwise Alignment</div>',
            unsafe_allow_html=True)
score = run_alignment(dna_seq[:compare_len], mutant_dna)
st.markdown(f"""
<div class="success-box">
✓ Alignment score: <b>{score:.2f}</b> (original vs mutant, {compare_len} bp window)
</div>""", unsafe_allow_html=True)

# Step 5 — ORFs
st.markdown('<div class="step-header">STEP 5 / 9 — ORF Discovery (All 6 Reading Frames)</div>',
            unsafe_allow_html=True)
with st.spinner("Scanning all 6 reading frames..."):
    orfs = find_orfs(dna_seq, min_length=orf_min_length)
export_orfs_csv(orfs)

st.markdown(f"""
<div class="success-box">
✓ ORFs found (≥{orf_min_length} bp): <b>{len(orfs)}</b>
</div>""", unsafe_allow_html=True)

if orfs:
    with st.expander(f"View top 10 ORFs"):
        for orf in orfs[:10]:
            st.markdown(
                f'<div class="orf-row">'
                f'Frame <b>{orf["frame"]}</b> | '
                f'Position {orf["start"]}–{orf["end"]} | '
                f'Length <b>{orf["length"]} nt</b> | '
                f'Protein: {orf["protein"][:30]}...'
                f'</div>',
                unsafe_allow_html=True
            )

# Step 6 — Codon Usage
st.markdown('<div class="step-header">STEP 6 / 9 — Codon Usage Bias</div>',
            unsafe_allow_html=True)
codon_freq = codon_usage(dna_seq)
top5 = sorted(codon_freq.items(), key=lambda x: x[1], reverse=True)[:5]
st.markdown(f"""
<div class="success-box">
✓ Codon usage calculated | Top codon: 
<b>{top5[0][0]}</b> ({top5[0][1]:.4f})
</div>""", unsafe_allow_html=True)

codon_cols = st.columns(5)
for i, (codon, freq) in enumerate(top5):
    codon_cols[i].markdown(f"""
    <div class="metric-card" style="text-align:center">
        <div class="metric-value" style="font-size:1.4rem">{codon}</div>
        <div class="metric-label">{freq:.4f}</div>
    </div>""", unsafe_allow_html=True)

# Step 7 — Phylogenetics
st.markdown('<div class="step-header">STEP 7 / 9 — Multi-Species Phylogenetic Analysis</div>',
            unsafe_allow_html=True)
if run_phylo:
    with st.spinner("Fetching homologs and building phylogenetic tree (this takes ~30s)..."):
        homologs  = fetch_multiple_sequences(TP53_HOMOLOGS, email)
        if len(homologs) >= 3:
            alignment = build_multiple_alignment(homologs)
            tree      = build_phylogenetic_tree(alignment)
            save_phylogenetic_tree(tree)
            export_distance_matrix_csv(alignment)

            st.markdown(f"""
            <div class="success-box">
            ✓ Phylogenetic tree built | 
            {len(homologs)} species × {len(alignment[0].seq):,} bp alignment
            </div>""", unsafe_allow_html=True)

            # Show tree image
            if os.path.exists("results/phylo_tree.png"):
                st.image("results/phylo_tree.png",
                         caption="Neighbour-Joining Phylogenetic Tree — TP53 Orthologs",
                         use_container_width=True)

            # Show distance matrix
            if os.path.exists("results/distance_matrix.csv"):
                import pandas as pd
                with st.expander("View species distance matrix"):
                    df = pd.read_csv("results/distance_matrix.csv", index_col=0)
                    st.dataframe(df.style.background_gradient(cmap="Blues"), 
                                use_container_width=True)
        else:
            st.warning("⚠️ Fewer than 3 homologs fetched. Skipping tree.")
else:
    st.info("⏭️ Phylogenetic analysis skipped.")

# Step 8 — Domain Annotation
st.markdown('<div class="step-header">STEP 8 / 9 — Protein Domain Annotation</div>',
            unsafe_allow_html=True)
if run_domains and protein_seq:
    with st.spinner("Querying EMBL-EBI InterPro API..."):
        domains = annotate_protein_domains(protein_seq, email)
    export_domains_csv(domains)

    if domains:
        st.markdown(f"""
        <div class="success-box">
        ✓ Protein domains annotated: <b>{len(domains)} hits</b>
        </div>""", unsafe_allow_html=True)

        for d in domains[:15]:
            st.markdown(
                f'<div class="domain-card">'
                f'<span class="domain-db">[{d["database"]}]</span> '
                f'<b>{d["accession"]}</b> — {d["name"]} '
                f'<span style="color:#8B949E">| aa {d["start"]}–{d["end"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ No domain annotations returned. EBI may be temporarily unavailable.")
else:
    st.info("⏭️ Domain annotation skipped.")

# Step 9 — Plots & Exports
st.markdown('<div class="step-header">STEP 9 / 9 — Visualisation & Exports</div>',
            unsafe_allow_html=True)

with st.spinner("Generating plots..."):
    if protein_seq:
        plot_all(dna_seq, protein_seq, gc_window=gc_window)

st.markdown("""
<div class="success-box">
✓ All results exported to results/ folder
</div>""", unsafe_allow_html=True)

if os.path.exists("results/tp53_analysis.png"):
    st.image("results/tp53_analysis.png",
             caption="GC Content & Amino Acid Frequency Analysis",
             use_container_width=True)

# ── Summary metrics ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Analysis Summary")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(dna_seq):,}</div>
        <div class="metric-label">Sequence Length (bp)</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(protein_seq) if protein_seq else 0}</div>
        <div class="metric-label">Protein Length (aa)</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(mutations)}</div>
        <div class="metric-label">Mutations Detected</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(orfs)}</div>
        <div class="metric-label">ORFs Discovered</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{score:.0f}</div>
        <div class="metric-label">Alignment Score</div>
    </div>""", unsafe_allow_html=True)

# ── Download buttons ───────────────────────────────────────────────────────
st.markdown("## 📥 Download Results")
dl1, dl2, dl3, dl4 = st.columns(4)

def read_file(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

with dl1:
    data = read_file("results/mutations.csv")
    if data:
        st.download_button("⬇️ Mutations CSV", data,
                          "mutations.csv", "text/csv")
with dl2:
    data = read_file("results/orfs.csv")
    if data:
        st.download_button("⬇️ ORFs CSV", data,
                          "orfs.csv", "text/csv")
with dl3:
    data = read_file("results/protein_domains.csv")
    if data:
        st.download_button("⬇️ Domains CSV", data,
                          "protein_domains.csv", "text/csv")
with dl4:
    data = read_file("results/distance_matrix.csv")
    if data:
        st.download_button("⬇️ Distance Matrix", data,
                          "distance_matrix.csv", "text/csv")

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8B949E; font-size:0.8rem; padding:16px 0">
Built by <b>Samuel Mbote</b> — General Surgery Resident & Bioinformatics Developer<br>
<a href="https://github.com/mbote-droid/tp53_analysis" 
   style="color:#2196F3;">github.com/mbote-droid/tp53_analysis</a>
</div>
""", unsafe_allow_html=True)