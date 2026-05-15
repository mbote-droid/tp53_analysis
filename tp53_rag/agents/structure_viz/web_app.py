"""
============================================================
TP53 Structure Web App — Flask server
============================================================
Serves the 3D visualisation web app locally.
Opens automatically in the browser after analysis.

Run: python -m agents.structure_viz.web_app
  or: python main.py visualise --accession NM_000546
============================================================
"""

import json
import webbrowser
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, render_template_string, jsonify, send_file
from utils.logger import log

app = Flask(__name__)

# Global: current structure data loaded for display
_current_data: Optional[dict] = None
_current_accession: str = "TP53"


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TP53 Structure — {{ accession }}</title>

<!-- Mol* for 3D protein structure -->
<script src="https://cdn.jsdelivr.net/npm/molstar@3.43.0/build/viewer/molstar.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar@3.43.0/build/viewer/molstar.css">

<!-- Plotly for embedding scatter -->
<script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>

<!-- Google Fonts -->
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=JetBrains+Mono:wght@300;400&display=swap" rel="stylesheet">

<style>
  :root {
    --bg: #050a0e;
    --surface: #0d1821;
    --border: #1a2840;
    --accent: #00d4ff;
    --accent2: #ff6b35;
    --hotspot: #ff3366;
    --safe: #00ff88;
    --text: #e8f4f8;
    --muted: #4a7a8a;
    --domain-dbd: #4ECDC4;
    --domain-tet: #A29BFE;
    --domain-tad: #FF6B6B;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated background grid */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .app-container {
    position: relative;
    z-index: 1;
    max-width: 1600px;
    margin: 0 auto;
    padding: 24px;
  }

  /* ── Header ── */
  header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }

  .logo-area h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .logo-area p {
    color: var(--muted);
    font-size: 0.75rem;
    margin-top: 4px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  .badge-row {
    display: flex;
    gap: 8px;
  }

  .badge {
    padding: 4px 12px;
    border-radius: 2px;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
  }
  .badge-gemma { background: rgba(0,212,255,0.12); color: var(--accent); border: 1px solid rgba(0,212,255,0.3); }
  .badge-local { background: rgba(0,255,136,0.1); color: var(--safe); border: 1px solid rgba(0,255,136,0.2); }
  .badge-esmfold { background: rgba(255,107,53,0.1); color: var(--accent2); border: 1px solid rgba(255,107,53,0.2); }

  /* ── Main Grid ── */
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 380px;
    grid-template-rows: auto auto;
    gap: 16px;
  }

  /* ── Panels ── */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }

  .panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
  }

  .panel-tag {
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.05em;
  }

  /* ── 3D Structure Viewer ── */
  .structure-panel {
    grid-column: 1;
    grid-row: 1;
    height: 520px;
  }

  #molstar-container {
    width: 100%;
    height: 460px;
    background: #020609;
  }

  /* ── Embedding Panel ── */
  .embedding-panel {
    grid-column: 1;
    grid-row: 2;
    height: 400px;
  }

  #embedding-plot {
    width: 100%;
    height: 340px;
  }

  /* ── Right Sidebar ── */
  .sidebar {
    grid-column: 2;
    grid-row: 1 / 3;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* ── Mutation List ── */
  .mutation-list {
    flex: 0 0 auto;
  }

  .mutation-item {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    transition: background 0.15s;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .mutation-item:hover { background: rgba(0,212,255,0.05); }
  .mutation-item.active { background: rgba(0,212,255,0.08); border-left: 2px solid var(--accent); }

  .mut-label {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    min-width: 60px;
  }

  .mut-label.hotspot { color: var(--hotspot); }
  .mut-label.non-hotspot { color: var(--accent2); }

  .mut-info { flex: 1; }
  .mut-domain { font-size: 0.7rem; color: var(--muted); }
  .mut-impact { font-size: 0.65rem; margin-top: 2px; }
  .mut-impact.contact { color: #ff9f43; }
  .mut-impact.conformational { color: #a29bfe; }

  .mut-badge {
    font-size: 0.6rem;
    padding: 2px 6px;
    border-radius: 2px;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.06em;
  }
  .mut-badge.hotspot { background: rgba(255,51,102,0.15); color: var(--hotspot); }
  .mut-badge.non-hotspot { background: rgba(255,107,53,0.1); color: var(--accent2); }

  /* ── Domain Legend ── */
  .domain-legend { flex: 0 0 auto; }

  .domain-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 0.7rem;
  }

  .domain-swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .domain-name { flex: 1; color: var(--text); }
  .domain-range { color: var(--muted); font-size: 0.65rem; }

  /* ── Narration Panel ── */
  .narration-panel {
    flex: 1;
    min-height: 160px;
  }

  .narration-text {
    padding: 14px 16px;
    font-size: 0.72rem;
    line-height: 1.7;
    color: #b0c4d0;
    font-family: 'JetBrains Mono', monospace;
    overflow-y: auto;
    max-height: 200px;
  }

  /* ── pLDDT Bar ── */
  .plddt-panel { flex: 0 0 auto; }

  #plddt-chart { width: 100%; height: 80px; }

  /* ── Controls ── */
  .controls {
    display: flex;
    gap: 8px;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }

  .btn {
    padding: 5px 14px;
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    cursor: pointer;
    border: 1px solid;
    transition: all 0.15s;
    text-transform: uppercase;
  }

  .btn-accent {
    background: rgba(0,212,255,0.1);
    color: var(--accent);
    border-color: rgba(0,212,255,0.3);
  }
  .btn-accent:hover { background: rgba(0,212,255,0.2); }

  .btn-danger {
    background: rgba(255,51,102,0.1);
    color: var(--hotspot);
    border-color: rgba(255,51,102,0.3);
  }
  .btn-danger:hover { background: rgba(255,51,102,0.2); }

  /* ── Stats strip ── */
  .stats-strip {
    display: flex;
    gap: 24px;
    padding: 12px 0;
    margin-bottom: 16px;
  }

  .stat { display: flex; flex-direction: column; }
  .stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
  }
  .stat-label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px; }

  .stat-value.danger { color: var(--hotspot); }
  .stat-value.warn { color: var(--accent2); }

  /* ── Loading overlay ── */
  #loading {
    position: fixed; inset: 0;
    background: rgba(5,10,14,0.92);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    z-index: 1000;
    transition: opacity 0.4s;
  }

  #loading.hidden { opacity: 0; pointer-events: none; }

  .loader-ring {
    width: 60px; height: 60px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 20px;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .loader-text {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>

<body>
<div id="loading">
  <div class="loader-ring"></div>
  <div class="loader-text">Initialising 3D Structure Engine...</div>
</div>

<div class="app-container">

  <header>
    <div class="logo-area">
      <h1>TP53 · Structure Atlas</h1>
      <p>Multi-Agent Bioinformatics Platform · Gemma 4 Hackathon</p>
    </div>
    <div class="badge-row">
      <span class="badge badge-gemma">Gemma 4</span>
      <span class="badge badge-local">100% Local</span>
      <span class="badge badge-esmfold">ESMFold + ESM-2</span>
    </div>
  </header>

  <!-- Stats Strip -->
  <div class="stats-strip" id="stats-strip">
    <div class="stat">
      <span class="stat-value" id="stat-length">—</span>
      <span class="stat-label">Residues</span>
    </div>
    <div class="stat">
      <span class="stat-value danger" id="stat-hotspots">—</span>
      <span class="stat-label">Hotspot Mutations</span>
    </div>
    <div class="stat">
      <span class="stat-value warn" id="stat-mutations">—</span>
      <span class="stat-label">Total Mutations</span>
    </div>
    <div class="stat">
      <span class="stat-value" id="stat-domains">7</span>
      <span class="stat-label">Domains Annotated</span>
    </div>
    <div class="stat">
      <span class="stat-value" id="stat-model">ESMFold</span>
      <span class="stat-label">Structure Model</span>
    </div>
  </div>

  <div class="main-grid">

    <!-- 3D Structure Viewer -->
    <div class="panel structure-panel">
      <div class="panel-header">
        <span class="panel-title">3D Structure · Wildtype vs Mutant</span>
        <span class="panel-tag" id="structure-tag">Loading...</span>
      </div>
      <div class="controls">
        <button class="btn btn-accent" onclick="showWildtype()">Wildtype</button>
        <button class="btn btn-danger" onclick="showMutant(0)">Mutant #1</button>
        <button class="btn btn-accent" onclick="highlightMutations()">Highlight Mutations</button>
        <button class="btn btn-accent" onclick="toggleSurface()">Surface</button>
        <button class="btn btn-accent" onclick="resetView()">Reset View</button>
      </div>
      <div id="molstar-container"></div>
    </div>

    <!-- Sidebar -->
    <div class="sidebar">

      <!-- Mutation List -->
      <div class="panel mutation-list">
        <div class="panel-header">
          <span class="panel-title">Mutations Mapped</span>
          <span class="panel-tag">Click to navigate</span>
        </div>
        <div id="mutation-list-body"></div>
      </div>

      <!-- Domain Legend -->
      <div class="panel domain-legend">
        <div class="panel-header">
          <span class="panel-title">Domain Architecture</span>
          <span class="panel-tag">p53 (393 aa)</span>
        </div>
        <div id="domain-legend-body"></div>
      </div>

      <!-- Gemma 4 Narration -->
      <div class="panel narration-panel">
        <div class="panel-header">
          <span class="panel-title">Gemma 4 · Structural Interpretation</span>
          <span class="panel-tag">RAG-grounded</span>
        </div>
        <div class="narration-text" id="narration-text">Loading interpretation...</div>
      </div>

    </div>

    <!-- ESM-2 Embedding Plot -->
    <div class="panel embedding-panel">
      <div class="panel-header">
        <span class="panel-title">ESM-2 Residue Embeddings · 3D Space (UMAP)</span>
        <span class="panel-tag">Wildtype vs Mutant — mutation effect in embedding space</span>
      </div>
      <div id="embedding-plot"></div>
    </div>

  </div>

</div>

<script>
// ── Data injected from Flask ──────────────────────────────
const DATA_URL = '/api/structure-data';

let structureData = null;
let molstarViewer = null;
let currentPDB = 'wildtype';
let showSurface = false;

// ── Load data and initialise ──────────────────────────────
async function init() {
  try {
    const resp = await fetch(DATA_URL);
    structureData = await resp.json();

    updateStats();
    renderMutationList();
    renderDomainLegend();
    renderNarration();

    await initMolstar();
    renderEmbeddingPlot();

    document.getElementById('loading').classList.add('hidden');

  } catch(e) {
    console.error('Init failed:', e);
    document.querySelector('.loader-text').textContent = 'Error loading data — check console';
  }
}

// ── Stats ─────────────────────────────────────────────────
function updateStats() {
  const d = structureData;
  document.getElementById('stat-length').textContent = d.sequence_length || '—';
  const hotspots = (d.mutations || []).filter(m => m.is_hotspot).length;
  document.getElementById('stat-hotspots').textContent = hotspots;
  document.getElementById('stat-mutations').textContent = (d.mutations || []).length;
  document.getElementById('stat-model').textContent = d.model_used || 'ESMFold';
  document.getElementById('structure-tag').textContent = d.accession || 'TP53';
}

// ── Mutation List ─────────────────────────────────────────
function renderMutationList() {
  const container = document.getElementById('mutation-list-body');
  const mutations = structureData.mutations || [];

  if (!mutations.length) {
    container.innerHTML = '<div style="padding:16px;color:var(--muted);font-size:0.75rem;">No mutations detected</div>';
    return;
  }

  container.innerHTML = mutations.map((m, i) => `
    <div class="mutation-item" onclick="focusMutation(${i})" id="mut-item-${i}">
      <span class="mut-label ${m.clinical_class}">${m.label}</span>
      <div class="mut-info">
        <div class="mut-domain">${m.domain} · pos ${m.position}</div>
        <div class="mut-impact ${m.functional_impact.split('_')[0]}">${m.functional_impact.replace(/_/g,' ')}</div>
      </div>
      <span class="mut-badge ${m.clinical_class}">${m.clinical_class}</span>
    </div>
  `).join('');
}

// ── Domain Legend ─────────────────────────────────────────
function renderDomainLegend() {
  const container = document.getElementById('domain-legend-body');
  const domains = structureData.domain_annotations || [];

  container.innerHTML = domains.map(d => `
    <div class="domain-item">
      <div class="domain-swatch" style="background:${d.color}"></div>
      <span class="domain-name">${d.short}</span>
      <span class="domain-range">${d.start}–${d.end}</span>
    </div>
  `).join('');
}

// ── Narration ─────────────────────────────────────────────
function renderNarration() {
  const el = document.getElementById('narration-text');
  el.textContent = structureData.llm_narration || 'No narration available.';
}

// ── Mol* 3D Viewer ────────────────────────────────────────
async function initMolstar() {
  const { createPluginUI } = molstar.Viewer;

  try {
    molstarViewer = await createPluginUI(
      document.getElementById('molstar-container'),
      {
        layoutIsExpanded: false,
        layoutShowControls: false,
        layoutShowRemoteState: false,
        layoutShowSequence: true,
        layoutShowLog: false,
        viewportShowExpand: false,
        collapseLeftPanel: true,
      }
    );

    await loadStructure('wildtype');

  } catch(e) {
    console.error('Mol* init failed:', e);
    document.getElementById('molstar-container').innerHTML =
      '<div style="color:var(--muted);padding:20px;font-size:0.75rem;">3D viewer unavailable — Mol* CDN required</div>';
  }
}

async function loadStructure(type) {
  if (!molstarViewer) return;

  const pdbResp = await fetch(`/api/pdb/${type}`);
  const pdbText = await pdbResp.text();

  await molstarViewer.plugin.clear();

  const data = await molstarViewer.plugin.builders.data.rawData(
    { data: pdbText, label: `TP53 ${type}` }
  );
  const traj = await molstarViewer.plugin.builders.structure.parseTrajectory(data, 'pdb');
  const model = await molstarViewer.plugin.builders.structure.createModel(traj);
  const structure = await molstarViewer.plugin.builders.structure.createStructure(model);

  // Apply cartoon representation
  await molstarViewer.plugin.builders.structure.representation.addRepresentation(structure, {
    type: 'cartoon',
    color: 'residue-name',
  });

  // Highlight mutation sites
  await highlightMutationSites();
}

async function highlightMutationSites() {
  if (!molstarViewer || !structureData.mutations) return;
  // Mutation highlighting via residue selection would go here
  // Mol* supports custom coloring by residue number
}

function showWildtype() {
  currentPDB = 'wildtype';
  loadStructure('wildtype');
}

function showMutant(idx) {
  const mutations = structureData.mutations || [];
  if (mutations[idx]) {
    currentPDB = mutations[idx].label;
    loadStructure(`mutant/${mutations[idx].label}`);
  }
}

function highlightMutations() {
  // Toggle mutation highlighting
  highlightMutationSites();
}

function toggleSurface() {
  showSurface = !showSurface;
  // Mol* surface toggle would go here
}

function resetView() {
  if (molstarViewer) {
    molstarViewer.plugin.managers.camera.reset();
  }
}

function focusMutation(idx) {
  document.querySelectorAll('.mutation-item').forEach(el => el.classList.remove('active'));
  document.getElementById(`mut-item-${idx}`).classList.add('active');
  // Navigate Mol* camera to mutation site
  const mut = structureData.mutations[idx];
  if (mut && molstarViewer) {
    // Camera focus on residue would go here
  }
}

// ── ESM-2 Embedding 3D Scatter ────────────────────────────
function renderEmbeddingPlot() {
  const coords = structureData.embedding_coords || [];
  const mutations = structureData.mutations || [];
  const domains = structureData.domain_annotations || [];

  if (!coords.length) {
    document.getElementById('embedding-plot').innerHTML =
      '<div style="padding:20px;color:var(--muted);font-size:0.75rem;">ESM-2 embeddings not available — install fair-esm</div>';
    return;
  }

  const n = coords.length;

  // Colour by domain
  const domainColors = coords.map((_, i) => {
    const residue = i + 1;
    const domain = domains.find(d => residue >= d.start && residue <= d.end);
    return domain ? domain.color : '#4a7a8a';
  });

  // Mark mutation positions
  const mutPositions = mutations.map(m => m.position - 1);
  const sizes = coords.map((_, i) => mutPositions.includes(i) ? 12 : 4);
  const mutColors = coords.map((_, i) => {
    const mut = mutations.find(m => (m.position - 1) === i);
    if (!mut) return domainColors[i];
    return mut.is_hotspot ? '#ff3366' : '#ff6b35';
  });

  const traces = [];

  // Wildtype residues
  traces.push({
    type: 'scatter3d',
    mode: 'markers',
    name: 'Wildtype residues',
    x: coords.map(c => c[0]),
    y: coords.map(c => c[1]),
    z: coords.map(c => c[2]),
    marker: {
      size: sizes,
      color: mutColors,
      opacity: 0.8,
      line: { width: 0 },
    },
    text: coords.map((_, i) => {
      const mut = mutations.find(m => (m.position - 1) === i);
      const domain = domains.find(d => (i+1) >= d.start && (i+1) <= d.end);
      return `Residue ${i+1}${mut ? ` · ${mut.label} ⚠` : ''} · ${domain ? domain.short : ''}`;
    }),
    hovertemplate: '%{text}<extra></extra>',
  });

  // Add mutant traces
  const mutantCoords = structureData.mutant_embedding_coords || {};
  const mutantColors = ['#00d4ff', '#a29bfe', '#00ff88'];
  Object.entries(mutantCoords).forEach(([label, mCoords], idx) => {
    if (!mCoords.length) return;
    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      name: `Mutant: ${label}`,
      x: mCoords.map(c => c[0]),
      y: mCoords.map(c => c[1]),
      z: mCoords.map(c => c[2]),
      marker: {
        size: 3,
        color: mutantColors[idx % mutantColors.length],
        opacity: 0.5,
        symbol: 'diamond',
        line: { width: 0 },
      },
    });
  });

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'JetBrains Mono', color: '#4a7a8a', size: 10 },
    scene: {
      bgcolor: '#050a0e',
      xaxis: { gridcolor: '#1a2840', zerolinecolor: '#1a2840', title: '' },
      yaxis: { gridcolor: '#1a2840', zerolinecolor: '#1a2840', title: '' },
      zaxis: { gridcolor: '#1a2840', zerolinecolor: '#1a2840', title: '' },
    },
    legend: {
      font: { color: '#4a7a8a', size: 9 },
      bgcolor: 'transparent',
      x: 0.01, y: 0.99,
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
  };

  Plotly.newPlot('embedding-plot', traces, layout, {
    responsive: true,
    displayModeBar: false,
  });
}

// ── Boot ──────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', init);
</script>

</body>
</html>'''


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, accession=_current_accession)


@app.route("/api/structure-data")
def get_structure_data():
    if _current_data is None:
        return jsonify({"error": "No structure data loaded"}), 404
    return jsonify(_current_data)


@app.route("/api/pdb/wildtype")
def get_wildtype_pdb():
    if _current_data and _current_data.get("wildtype_pdb"):
        return _current_data["wildtype_pdb"], 200, {"Content-Type": "text/plain"}
    # Try loading from file
    pdb_path = Path(f"data/structure_outputs/{_current_accession}_wildtype.pdb")
    if pdb_path.exists():
        return pdb_path.read_text(), 200, {"Content-Type": "text/plain"}
    return "REMARK No structure available", 200, {"Content-Type": "text/plain"}


@app.route("/api/pdb/mutant/<label>")
def get_mutant_pdb(label):
    if _current_data and label in (_current_data.get("mutant_pdbs") or {}):
        return _current_data["mutant_pdbs"][label], 200, {"Content-Type": "text/plain"}
    pdb_path = Path(f"data/structure_outputs/{_current_accession}_{label}.pdb")
    if pdb_path.exists():
        return pdb_path.read_text(), 200, {"Content-Type": "text/plain"}
    return "REMARK No mutant structure available", 200, {"Content-Type": "text/plain"}


def launch(structure_output: dict, accession: str = "TP53", port: int = 5001):
    """
    Launch the web app with the given structure output data.
    Called automatically after structure analysis completes.
    """
    global _current_data, _current_accession
    _current_data = structure_output
    _current_accession = accession

    url = f"http://localhost:{port}"
    log.info(f"Opening 3D visualisation at {url}")

    # Open browser after short delay
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host="localhost", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    # Demo mode: load existing structure data if available
    demo_path = Path("data/structure_outputs/NM_000546_structure_data.json")
    if demo_path.exists():
        import json
        data = json.loads(demo_path.read_text())
        launch(data, accession="NM_000546")
    else:
        log.warning("No structure data found. Run analysis first: python main.py visualise")
