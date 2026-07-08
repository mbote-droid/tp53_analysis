"""
Precision Onco Africa — Visualization helpers.

Pure, Streamlit-free functions so they can be unit-tested without a running
app. app.py imports these; tests import these. Everything returns a non-empty
result (honours the platform's zero-empty-output rule) and degrades
gracefully on bad input.
"""
from __future__ import annotations

import html
import json
import math
from typing import Optional

import plotly.graph_objects as go

# ── Domain knowledge ──────────────────────────────────────────────
CANONICAL_HOTSPOTS = (175, 248, 273)

# p53 domain map — name, residue range, colour, function. Shared by the
# 3D viewer colouring and the side legend chart so they always agree.
P53_DOMAINS = [
    {"name": "TAD", "start": 1,   "end": 66,  "color": "#00d4ff", "function": "Transactivation"},
    {"name": "PRD", "start": 67,  "end": 93,  "color": "#a29bfe", "function": "Proline-rich"},
    {"name": "DBD", "start": 94,  "end": 292, "color": "#2ecc71", "function": "DNA-binding (hotspots)"},
    {"name": "NLS", "start": 293, "end": 322, "color": "#f1c40f", "function": "Nuclear localization"},
    {"name": "TET", "start": 323, "end": 356, "color": "#f39c12", "function": "Tetramerization"},
    {"name": "REG", "start": 357, "end": 393, "color": "#ff6b9d", "function": "Regulatory"},
]

# How many full revolutions the dispatch network spins across one play-through.
# Set to 0.0 to stop the connectors spinning entirely.
DISPATCH_SPIN_REVOLUTIONS = 1.0


def agent_status_badge(name: str, state: str = "running",
                       elapsed: Optional[float] = None) -> str:
    """Return HTML for a single live agent status badge.

    state: 'running' | 'complete' | 'failed'. Always returns a non-empty
    string; unknown states render as a neutral badge.
    """
    safe_state = state if state in ("running", "complete", "failed") else "complete"
    label = str(name).replace("_", " ").title()
    icon = {"running": "🔄", "complete": "✅", "failed": "❌"}.get(safe_state, "•")
    time_txt = f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
    return (
        f'<div class="tp53-badge {safe_state}">'
        f'<span class="dot"></span>'
        f'<span class="name">{icon} {label}</span>'
        f'<span class="time">{time_txt}</span>'
        f'</div>'
    )


def _empty_fig(message: str) -> go.Figure:
    """A never-empty placeholder figure (honours the zero-results rule)."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(color="#8b98a5", family="JetBrains Mono"))
    fig.update_layout(template="plotly_dark", height=320,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def animated_vaf_timeline(days, vafs, mrd_threshold: float = 5.0) -> go.Figure:
    """Animated ctDNA VAF reveal with an MRD threshold line.

    Markers turn green when VAF falls (responding) and red when it rises
    (progressing). Plays by progressively revealing each timepoint.
    """
    try:
        days = list(days)
        vafs = [float(v) for v in vafs]
    except (TypeError, ValueError):
        return _empty_fig("VAF data unavailable")
    if not days or len(days) != len(vafs):
        return _empty_fig("No VAF timepoints to plot")

    colours = ["#2ecc71"]
    for i in range(1, len(vafs)):
        colours.append("#ff4b4b" if vafs[i] > vafs[i - 1] else "#2ecc71")

    def _trace(n):
        return go.Scatter(
            x=days[:n], y=vafs[:n], mode="lines+markers",
            line=dict(color="#00d4ff", width=2),
            marker=dict(color=colours[:n], size=10,
                        line=dict(color="#0d1117", width=1)),
            hovertemplate="Day %{x}: %{y:.1f}%<extra></extra>",
        )

    frames = [go.Frame(data=[_trace(n)], name=str(n))
              for n in range(1, len(days) + 1)]
    fig = go.Figure(data=[_trace(1)], frames=frames)
    fig.add_hline(y=mrd_threshold, line_dash="dash", line_color="#f39c12",
                  annotation_text=f"MRD threshold {mrd_threshold:.0f}%",
                  annotation_font_color="#f39c12")
    fig.update_layout(
        template="plotly_dark", height=340,
        title="Variant Allele Frequency Over Treatment",
        xaxis_title="Day", yaxis_title="VAF (%)",
        yaxis=dict(range=[0, max(vafs) * 1.15 + 1]),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=1.18, xanchor="left",
            buttons=[dict(label="▶ Play", method="animate",
                          args=[None, dict(frame=dict(duration=400, redraw=True),
                                           fromcurrent=True)])],
        )],
    )
    return fig


def animated_hotspot_bar(codons, freqs) -> go.Figure:
    """Animated 'build-up' bar chart of TP53 hotspot frequencies (Reds scale)."""
    try:
        codons = [str(c) for c in codons]
        freqs = [float(f) for f in freqs]
    except (TypeError, ValueError):
        return _empty_fig("Hotspot data unavailable")
    if not codons or len(codons) != len(freqs):
        return _empty_fig("No hotspot data to plot")

    steps = 12
    frames = [
        go.Frame(data=[go.Bar(
            x=codons, y=[f * (s / steps) for f in freqs],
            marker=dict(color=freqs, colorscale="Reds",
                        cmin=0, cmax=max(freqs) or 1),
            hovertemplate="Codon %{x}: %{y:.1f}%<extra></extra>",
        )], name=str(s))
        for s in range(1, steps + 1)
    ]
    fig = go.Figure(
        data=[go.Bar(x=codons, y=[0] * len(codons),
                     marker=dict(color=freqs, colorscale="Reds",
                                 cmin=0, cmax=max(freqs) or 1))],
        frames=frames,
    )
    fig.update_layout(
        template="plotly_dark", height=340,
        title="Known TP53 Hotspot Mutations",
        xaxis_title="Codon", yaxis_title="Frequency (%)",
        yaxis=dict(range=[0, max(freqs) * 1.15 + 0.5]),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=1.18, xanchor="left",
            buttons=[dict(label="▶ Play", method="animate",
                          args=[None, dict(frame=dict(duration=90, redraw=True),
                                           fromcurrent=True)])],
        )],
    )
    return fig


# ── 3D agent graph (Obsidian-style WebGL force-directed network) ──────────
# Curated architecture: agents grouped by domain, plus semantic edges between
# agents that actually pass data/context to each other.
_AGENT_GRAPH_GROUPS = {
    "Genomics":          (["variant_curator", "mutation_analysis", "domain_annotation",
                           "gene_expression", "vcf_parser"], "#2ecc71"),
    "Clinical":          (["clinical_interpretation", "tnm_staging", "surgical_brief",
                           "immunogenicity", "liquid_biopsy"], "#00d4ff"),
    "Drug discovery":    (["drug_discovery", "molecular_docking", "synthetic_lethality",
                           "structural_analyzer", "ind_generator"], "#ff6b9d"),
    "Imaging":           (["pathology_vision", "structure_viz"], "#f39c12"),
    "Equity & evidence": (["african_atlas", "clinvar_conflict_checker",
                           "clinical_trials", "pubmed_citations"], "#a29bfe"),
}
_AGENT_GRAPH_EDGES = [
    ("variant_curator", "clinvar_conflict_checker"),
    ("variant_curator", "mutation_analysis"),
    ("mutation_analysis", "drug_discovery"),
    ("drug_discovery", "molecular_docking"),
    ("molecular_docking", "structural_analyzer"),
    ("drug_discovery", "synthetic_lethality"),
    ("structural_analyzer", "ind_generator"),
    ("pathology_vision", "tnm_staging"),
    ("clinical_interpretation", "tnm_staging"),
    ("clinical_interpretation", "clinical_trials"),
    ("gene_expression", "immunogenicity"),
    ("african_atlas", "variant_curator"),
]


def build_agent_graph_data(agent_names: Optional[list] = None) -> dict:
    """Build {nodes, links} for the 3D multi-agent graph. Pure data, never empty.

    Central `Dispatcher` → `RAG Core` → every agent, plus curated agent-to-agent
    semantic edges. If `agent_names` is given, only those agents are included
    (others are dropped, but the graph is always at least the Dispatcher node).
    """
    keep = set(agent_names) if agent_names else None

    def _pretty(s: str) -> str:
        return str(s).replace("_", " ").title()

    nodes = [
        {"id": "dispatcher", "name": "Dispatcher", "group": "Core", "val": 26, "color": "#ffffff"},
        {"id": "rag_core", "name": "RAG Core", "group": "Core", "val": 18, "color": "#00d4ff"},
    ]
    links = [{"source": "dispatcher", "target": "rag_core"}]

    present = {"dispatcher", "rag_core"}
    for group, (members, color) in _AGENT_GRAPH_GROUPS.items():
        for agent in members:
            if keep is not None and agent not in keep:
                continue
            nodes.append({"id": agent, "name": _pretty(agent), "group": group,
                          "val": 7, "color": color})
            links.append({"source": "rag_core", "target": agent})
            present.add(agent)

    for a, b in _AGENT_GRAPH_EDGES:
        if a in present and b in present:
            links.append({"source": a, "target": b})

    return {"nodes": nodes, "links": links}


def agent_graph_3d_html(graph_data: Optional[dict] = None, height: int = 560) -> str:
    """Self-contained HTML for an interactive 3D force-directed agent graph.

    Uses the `3d-force-graph` WebGL library (drag to rotate, scroll to zoom,
    click a node to focus). Obsidian-style organic network. Always returns a
    non-empty HTML string; if the CDN library can't load (offline/edge), it
    shows a graceful fallback message instead of a blank box.
    """
    if not graph_data or not graph_data.get("nodes"):
        graph_data = build_agent_graph_data()
    data_json = json.dumps(graph_data)
    inner_h = max(int(height) - 8, 200)

    template = """
<div style="width:100%;background:#0d1117;border-radius:10px;overflow:hidden;">
  <div id="tp53-agraph" style="width:100%;height:__H__px;"></div>
  <div id="tp53-agraph-fallback" style="display:none;padding:18px;color:#8b98a5;
       font-family:sans-serif;font-size:0.85rem;">
    3D agent graph needs internet access once to load the WebGL library.
    Offline? The 2D dispatch diagram below still works.
  </div>
</div>
<script>
(function(){
  var DATA = __DATA__;
  function boot(){
    var el = document.getElementById('tp53-agraph');
    var Graph = ForceGraph3D()(el)
      .graphData(DATA)
      .backgroundColor('#0d1117')
      .nodeLabel(function(n){ return n.name; })
      .nodeVal(function(n){ return n.val; })
      .nodeColor(function(n){ return n.color; })
      .nodeOpacity(0.95)
      .nodeResolution(16)
      .linkColor(function(){ return 'rgba(0,212,255,0.22)'; })
      .linkWidth(0.6)
      .linkDirectionalParticles(2)
      .linkDirectionalParticleWidth(1.4)
      .linkDirectionalParticleSpeed(0.006)
      .warmupTicks(60)
      .cooldownTicks(160)
      .onNodeClick(function(node){
        var ratio = 1 + 90 / Math.max(Math.hypot(node.x, node.y, node.z || 1), 1);
        Graph.cameraPosition(
          {x: node.x * ratio, y: node.y * ratio, z: (node.z || 1) * ratio},
          node, 1400);
      });
  }
  function showFallback(){
    document.getElementById('tp53-agraph-fallback').style.display = 'block';
  }
  var s = document.createElement('script');
  s.src = 'https://unpkg.com/3d-force-graph';
  s.onload = boot;
  s.onerror = function(){
    var s2 = document.createElement('script');
    s2.src = 'https://cdn.jsdelivr.net/npm/3d-force-graph';
    s2.onload = boot;
    s2.onerror = showFallback;
    document.head.appendChild(s2);
  };
  document.head.appendChild(s);
})();
</script>
"""
    return (template
            .replace("__DATA__", data_json)
            .replace("__H__", str(inner_h)))


def variant_annotation_table(result: Optional[dict]) -> go.Figure:
    """Render a real-variant annotation dict (utils.variant_annotation) as a
    clean two-column table. Pure + never-empty.
    """
    result = result or {}
    rows = [
        ("Protein change", result.get("protein_change") or "—"),
        ("Gene", result.get("gene") or "TP53"),
        ("rsID", result.get("rsid") or "—"),
        ("HGVS (c.)", result.get("hgvs_c") or "—"),
        ("Consequence", result.get("consequence") or "unknown"),
        ("Impact", result.get("impact") or "unknown"),
        ("SIFT", result.get("sift") or "unknown"),
        ("PolyPhen", result.get("polyphen") or "unknown"),
        ("CADD (phred)", result.get("cadd_phred") if result.get("cadd_phred") is not None else "—"),
        ("ClinVar", result.get("clinvar_significance") or "not_provided"),
        ("gnomAD AF", result.get("gnomad_af") or "unknown"),
        ("Structural class", result.get("structural_class") or "—"),
        ("Source", result.get("method") or "curated_fallback"),
    ]
    fields = [r[0] for r in rows]
    values = [str(r[1]) for r in rows]
    fig = go.Figure(data=[go.Table(
        columnwidth=[34, 66],
        header=dict(values=["<b>Field</b>", "<b>Value</b>"],
                    fill_color="#0b3d63", font=dict(color="white", size=13),
                    align="left", height=28),
        cells=dict(values=[fields, values],
                   fill_color=[["#161b22", "#0d1117"] * len(rows)],
                   font=dict(color="#e6edf3", size=12),
                   align="left", height=24),
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                      height=30 * len(rows) + 40,
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def variant_effect_gauge(result: Optional[dict]) -> go.Figure:
    """Gauge for an ESM-2 variant-effect result (utils.variant_effect).

    Shows the masked-marginal log-likelihood ratio (more negative = more
    deleterious). If the score is unavailable (matrix not precomputed), returns
    a clear 'pending' indicator instead of a fabricated value. Never empty.
    """
    result = result or {}
    score = result.get("esm2_score")
    if score is None:
        fig = go.Figure(go.Indicator(
            mode="number",
            value=0,
            number={"prefix": "", "suffix": "", "font": {"size": 1, "color": "rgba(0,0,0,0)"}},
            title={"text": "ESM-2 score<br><span style='font-size:0.8em;color:#8b98a5'>"
                           "not precomputed — run tools/precompute_esm2.py</span>"},
        ))
        fig.update_layout(height=180, margin=dict(l=20, r=20, t=50, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6edf3"})
        return fig

    label = result.get("interpretation", "")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        number={"font": {"size": 34, "color": "#e6edf3"}, "valueformat": ".2f"},
        title={"text": f"ESM-2 effect (LLR)<br><span style='font-size:0.8em;color:#8b98a5'>"
                       f"{label}</span>"},
        gauge={
            "axis": {"range": [-15, 5], "tickcolor": "#8b98a5"},
            "bar": {"color": "#00d4ff"},
            "steps": [
                {"range": [-15, -7.5], "color": "#5b1a1a"},   # likely deleterious
                {"range": [-7.5, -4], "color": "#6b4a16"},    # possibly
                {"range": [-4, 0], "color": "#2a3340"},       # uncertain
                {"range": [0, 5], "color": "#1e3a2a"},        # likely tolerated
            ],
            "threshold": {"line": {"color": "#ff4b4b", "width": 3},
                          "thickness": 0.75, "value": -7.5},
        },
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=60, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6edf3"})
    return fig


def agent_architecture_diagram(agent_names,
                               spin_revolutions: float = DISPATCH_SPIN_REVOLUTIONS
                               ) -> go.Figure:
    """Radial agent network with an animated 'data flow' dispatch.

    The dispatcher sits at the centre; each agent is a node on the ring.
    Pressing Play fires a glowing packet out along each edge in turn, lights
    up the receiving agent, and slowly rotates the whole ring so the
    connectors visibly spin. Pure Plotly — fully offline, no assets.

    `spin_revolutions` controls how many full turns the ring makes across one
    play-through (0.0 = no spin). Fixed 5-trace layout, identical trace order
    in every frame so Plotly can tween cleanly: dim edges, active edge, glow
    halo, nodes, travelling packet.
    """
    names = [str(n).replace("_", " ").title() for n in (agent_names or [])]
    if not names:
        return _empty_fig("No agents registered")

    n = len(names)
    cx, cy = 0.0, 0.0
    base_angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    labels = ["Dispatcher"] + names

    def coords(rot):
        xs = [cx] + [math.cos(a + rot) for a in base_angles]
        ys = [cy] + [math.sin(a + rot) for a in base_angles]
        return xs, ys

    def dim_edges(xs, ys):
        ex, ey = [], []
        for i in range(1, n + 1):
            ex += [cx, xs[i], None]
            ey += [cy, ys[i], None]
        return go.Scatter(x=ex, y=ey, mode="lines",
                          line=dict(color="#222b36", width=1), hoverinfo="skip")

    def active_edge(xs, ys, target):
        return go.Scatter(x=[cx, xs[target]], y=[cy, ys[target]], mode="lines",
                          line=dict(color="#00d4ff", width=3), hoverinfo="skip")

    def glow(xs, ys, target, on):
        return go.Scatter(
            x=[xs[target]] if on else [None], y=[ys[target]] if on else [None],
            mode="markers", hoverinfo="skip",
            marker=dict(color="rgba(0,255,156,0.25)", size=46),
        )

    def nodes(xs, ys, active):
        colors = ["#00d4ff"] + ["#3a4654"] * n
        sizes = [36] + [22] * n
        if active is not None:
            colors[active] = "#00ff9c"
            sizes[active] = 32
        return go.Scatter(
            x=xs, y=ys, mode="markers+text",
            text=labels, textposition="top center",
            textfont=dict(color="#c2ccd6", size=9, family="JetBrains Mono"),
            marker=dict(color=colors, size=sizes,
                        line=dict(color="#0d1117", width=1.5)),
            hovertemplate="%{text}<extra></extra>",
        )

    def packet(xs, ys, target, t):
        return go.Scatter(
            x=[cx + (xs[target] - cx) * t], y=[cy + (ys[target] - cy) * t],
            mode="markers", hoverinfo="skip",
            marker=dict(color="#00ff9c", size=12,
                        line=dict(color="#0d1117", width=1)),
        )

    substeps = (0.35, 0.7, 1.0)
    total_steps = n * len(substeps)
    step_idx = 0
    frames = []
    for active in range(1, n + 1):
        for t in substeps:
            rot = (2 * math.pi * spin_revolutions) * (step_idx / max(total_steps, 1))
            xs, ys = coords(rot)
            frames.append(go.Frame(
                name=f"{active}-{t}",
                data=[dim_edges(xs, ys), active_edge(xs, ys, active),
                      glow(xs, ys, active, t >= 1.0),
                      nodes(xs, ys, active if t >= 1.0 else None),
                      packet(xs, ys, active, t)],
            ))
            step_idx += 1

    xs0, ys0 = coords(0.0)
    fig = go.Figure(
        data=[dim_edges(xs0, ys0), active_edge(xs0, ys0, 1),
              glow(xs0, ys0, 1, False), nodes(xs0, ys0, None),
              packet(xs0, ys0, 1, 0.0)],
        frames=frames,
    )
    fig.update_layout(
        template="plotly_dark", height=480, showlegend=False,
        title=dict(text="Multi-Agent Dispatch Network",
                   font=dict(color="#e6edf3", size=16)),
        margin=dict(l=10, r=10, t=60, b=10),
        # fixedrange disables zoom/pan so a stray click/scroll can't jump the view
        xaxis=dict(visible=False, range=[-1.6, 1.6], fixedrange=True),
        yaxis=dict(visible=False, range=[-1.6, 1.6], scaleanchor="x", fixedrange=True),
        annotations=[dict(
            text="🟢 active agent &nbsp; 🔵 dispatcher &nbsp; — packet = data flow",
            x=0.5, y=-0.05, xref="paper", yref="paper", showarrow=False,
            font=dict(color="#8b98a5", size=10, family="JetBrains Mono"),
        )],
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=1.12, xanchor="left",
            buttons=[dict(label="▶ Trace dispatch", method="animate",
                          args=[None, dict(frame=dict(duration=180, redraw=True),
                                           fromcurrent=True,
                                           transition=dict(duration=120))])],
        )],
    )
    return fig


def domain_legend_chart() -> go.Figure:
    """Horizontal domain map that doubles as the colour legend for the 3D view."""
    fig = go.Figure()
    for d in P53_DOMAINS:
        fig.add_trace(go.Bar(
            y=["p53"], x=[d["end"] - d["start"] + 1], base=d["start"],
            orientation="h", name=d["name"], marker_color=d["color"],
            hovertemplate=f"{d['name']} ({d['start']}–{d['end']})<br>{d['function']}<extra></extra>",
        ))
    for r in CANONICAL_HOTSPOTS:
        fig.add_vline(x=r, line_color="#ff3b3b", line_width=2,
                      annotation_text=str(r), annotation_font_color="#ff3b3b",
                      annotation_font_size=10)
    fig.update_layout(
        template="plotly_dark", height=230, barmode="stack",
        title="p53 Domain Map (colours match the 3D structure)",
        xaxis_title="Residue position", yaxis=dict(visible=False),
        legend=dict(orientation="h", y=-0.45, font=dict(size=10)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# Significance → lollipop-head colour. Matches the platform's pathogenicity
# palette elsewhere (red = pathogenic, green = benign, amber = uncertain).
_NEEDLE_COLORS = {
    "pathogenic": "#ff3b3b",
    "likely_pathogenic": "#ff7b3b",
    "uncertain": "#f1c40f",
    "vus": "#f1c40f",
    "likely_benign": "#5dade2",
    "benign": "#2ecc71",
}
_NEEDLE_DEFAULT_COLOR = "#9b9b9b"
TP53_LENGTH = 393


def _domain_for_position(pos: int) -> Optional[dict]:
    """Return the P53_DOMAINS entry containing *pos*, or None if outside all."""
    for d in P53_DOMAINS:
        if d["start"] <= pos <= d["end"]:
            return d
    return None


def needle_plot(variants, title: str = "TP53 mutation needle plot") -> go.Figure:
    """Lollipop / needle plot of mutations along the p53 protein (1–393).

    *variants* is an iterable of dicts; each needs at least a residue position
    under any of the keys ``position`` / ``pos`` / ``residue`` / ``codon``.
    Optional per-variant keys:
      * ``count`` (int)        — stem height (recurrence); defaults to 1.
      * ``label`` (str)        — hover label, e.g. "R175H".
      * ``significance`` (str) — colours the head via _NEEDLE_COLORS.

    A domain track is drawn underneath (colours match the 3D viewer and the
    domain legend). Recurrent positions are merged: counts sum, the highest
    significance label wins. Pure + never-empty (graceful placeholder on no
    valid input). Out-of-range / non-numeric positions are dropped, not raised.
    """
    # ── Parse + merge by position ────────────────────────────────
    merged: dict = {}
    for v in variants or []:
        if not isinstance(v, dict):
            continue
        raw_pos = (v.get("position") if v.get("position") is not None
                   else v.get("pos") if v.get("pos") is not None
                   else v.get("residue") if v.get("residue") is not None
                   else v.get("codon"))
        try:
            pos = int(raw_pos)
        except (TypeError, ValueError):
            continue
        if not (1 <= pos <= TP53_LENGTH):
            continue
        try:
            count = int(v.get("count", 1))
        except (TypeError, ValueError):
            count = 1
        count = max(count, 1)

        entry = merged.setdefault(
            pos, {"count": 0, "labels": [], "significance": ""})
        entry["count"] += count
        label = str(v.get("label") or "").strip()
        if label and label not in entry["labels"]:
            entry["labels"].append(label)
        sig = str(v.get("significance") or "").strip().lower().replace(" ", "_")
        # Keep the most clinically severe significance seen at this position.
        severity = ["benign", "likely_benign", "vus", "uncertain",
                    "likely_pathogenic", "pathogenic"]
        if sig in severity and (
            entry["significance"] not in severity
            or severity.index(sig) > severity.index(entry["significance"])
        ):
            entry["significance"] = sig

    if not merged:
        return _empty_fig("No mutations to plot — enter variants to see the needle map")

    max_count = max(e["count"] for e in merged.values())
    fig = go.Figure()

    # ── Domain track (thin bars along the baseline at y=0) ───────
    for d in P53_DOMAINS:
        fig.add_trace(go.Bar(
            x=[d["end"] - d["start"] + 1], base=d["start"], y=[-max_count * 0.12],
            width=max_count * 0.10, orientation="h", marker_color=d["color"],
            name=d["name"], hovertemplate=(
                f"{d['name']} ({d['start']}–{d['end']})<br>{d['function']}"
                "<extra></extra>"),
            showlegend=True,
        ))

    # ── Stems + heads (one trace for stems, one for heads) ───────
    for pos in sorted(merged):
        e = merged[pos]
        fig.add_trace(go.Scatter(
            x=[pos, pos], y=[0, e["count"]], mode="lines",
            line=dict(color="#5a6570", width=1.4),
            hoverinfo="skip", showlegend=False,
        ))

    head_x, head_y, head_color, head_text = [], [], [], []
    for pos in sorted(merged):
        e = merged[pos]
        dom = _domain_for_position(pos)
        color = _NEEDLE_COLORS.get(e["significance"]) or (
            "#ff3b3b" if pos in CANONICAL_HOTSPOTS else _NEEDLE_DEFAULT_COLOR)
        head_x.append(pos)
        head_y.append(e["count"])
        head_color.append(color)
        # Escape user-derived labels — they flow into the hover label markup.
        lab = html.escape(", ".join(e["labels"]) or f"residue {pos}")
        sig_txt = html.escape(e["significance"].replace("_", " ") or "unclassified")
        dom_txt = dom["name"] if dom else "—"
        head_text.append(
            f"<b>{lab}</b><br>position {pos} · {dom_txt}"
            f"<br>recurrence {e['count']}<br>{sig_txt}")

    # Head size scales gently with recurrence (8–22 px).
    sizes = [8 + 14 * (merged[p]["count"] / max_count) for p in sorted(merged)]
    fig.add_trace(go.Scatter(
        x=head_x, y=head_y, mode="markers",
        marker=dict(size=sizes, color=head_color,
                    line=dict(color="#0d1117", width=1)),
        text=head_text, hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly_dark", height=380, barmode="overlay",
        title=str(title),
        xaxis=dict(title="Residue position", range=[0, TP53_LENGTH + 5],
                   showgrid=False),
        yaxis=dict(title="Recurrence (count)", rangemode="tozero",
                   zeroline=True, zerolinecolor="#2a2f37"),
        legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
        margin=dict(l=10, r=10, t=50, b=10),
        bargap=0,
    )
    return fig


def parse_residues(raw: str, extra=CANONICAL_HOTSPOTS) -> list:
    """Parse a residue string into a sorted list of unique ints (1–393).

    Non-numeric tokens are dropped — this also sanitises the value before
    it is embedded in the viewer markup (injection-safe).
    """
    found = set()
    for tok in str(raw or "").replace(",", " ").split():
        digits = "".join(ch for ch in tok if ch.isdigit())
        if digits:
            val = int(digits)
            if 1 <= val <= 393:
                found.add(val)
    for r in extra:
        if 1 <= int(r) <= 393:
            found.add(int(r))
    return sorted(found)


def protein_viewer_html(pdb_id: str, residues) -> str:
    """Self-contained 3Dmol.js viewer: domain-coloured cartoon, auto-rotating,
    hotspot residues shown as labelled red spheres.

    Loads the structure + library from CDN (same online assumption as the
    existing Mol* iframe). Residue list is integer-sanitised upstream;
    domain ranges/colours are injected from the trusted P53_DOMAINS constant.
    """
    safe_pdb = "".join(ch for ch in str(pdb_id) if ch.isalnum())[:8] or "2OCJ"
    resi_js = ",".join(str(int(r)) for r in residues) if residues else ""
    domains_js = ",".join(
        f"{{s:{int(d['start'])},e:{int(d['end'])},c:'{d['color']}'}}"
        for d in P53_DOMAINS
    )
    return f"""
    <div id="viewer3d" style="width:100%;height:480px;position:relative;
         background:#0d1117;border-radius:8px;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
    <script>
      (function() {{
        try {{
          var el = document.getElementById('viewer3d');
          var viewer = $3Dmol.createViewer(el, {{backgroundColor: '#0d1117'}});
          $3Dmol.download('pdb:{safe_pdb}', viewer, {{}}, function() {{
            viewer.setStyle({{}}, {{cartoon: {{color: 'lightgray', opacity: 0.6}}}});
            var domains = [{domains_js}];
            domains.forEach(function(d) {{
              viewer.setStyle({{resi: d.s + '-' + d.e}},
                {{cartoon: {{color: d.c, opacity: 0.9}}}});
            }});
            var resi = [{resi_js}];
            resi.forEach(function(r) {{
              viewer.setStyle({{resi: r}},
                {{stick: {{colorscheme: 'redCarbon', radius: 0.3}},
                 sphere: {{color: '#ff3b3b', radius: 1.0}}}});
              viewer.addLabel('Res ' + r, {{
                inFront: true, fontSize: 11, fontColor: '#ff6b6b',
                backgroundColor: '#0d1117', backgroundOpacity: 0.7,
                position: {{resi: r}}
              }});
            }});
            viewer.zoomTo();
            viewer.render();
            viewer.spin('y', 0.6);
          }});
        }} catch (e) {{
          document.getElementById('viewer3d').innerHTML =
            '<p style=\"color:#8b98a5;font-family:monospace;padding:1rem\">'
            + '3D viewer needs an internet connection.</p>';
        }}
      }})();
    </script>
    """


def alphafold_viewer_html(pdb_text: Optional[str], residues=None,
                          mean_plddt: Optional[float] = None, height: int = 480) -> str:
    """Self-contained 3Dmol.js viewer for a real AlphaFold model, coloured by the
    standard pLDDT confidence scheme (blue=very high ... orange=very low).

    The PDB text is embedded directly (fetched server-side by AlphaFoldClient) so
    there is no browser CORS issue. Hotspot residues are highlighted. Always
    returns a non-empty HTML string; shows a clear message if no model/library.
    """
    if not pdb_text:
        return ("<div style='padding:1rem;color:#8b98a5;font-family:monospace;"
                "background:#0d1117;border-radius:8px;'>AlphaFold model not loaded "
                "(offline or fetch failed). Use the experimental-structure viewer.</div>")
    resi_js = ",".join(str(int(r)) for r in (residues or []))
    pdb_js = json.dumps(pdb_text)
    mean_txt = (f"mean pLDDT {mean_plddt}" if mean_plddt is not None
                else "pLDDT confidence")
    template = """
    <div style="position:relative;">
      <div id="afviewer" style="width:100%;height:__H__px;background:#0d1117;
           border-radius:8px;"></div>
      <div style="position:absolute;top:8px;right:10px;background:rgba(13,17,23,0.85);
           padding:6px 9px;border-radius:6px;font-family:sans-serif;font-size:11px;
           color:#c9d1d9;line-height:1.5;">
        <b>__MEAN__</b><br>
        <span style="color:#0053D6;">&#9608;</span> very high (&gt;90)<br>
        <span style="color:#65CBF3;">&#9608;</span> confident (70-90)<br>
        <span style="color:#FFDB13;">&#9608;</span> low (50-70)<br>
        <span style="color:#FF7D45;">&#9608;</span> very low (&lt;50)
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
    <script>
      (function() {
        try {
          var el = document.getElementById('afviewer');
          var viewer = $3Dmol.createViewer(el, {backgroundColor: '#0d1117'});
          viewer.addModel(__PDB__, 'pdb');
          function plddt(atom) {
            var b = atom.b;
            if (b > 90) return 0x0053D6;
            if (b > 70) return 0x65CBF3;
            if (b > 50) return 0xFFDB13;
            return 0xFF7D45;
          }
          viewer.setStyle({}, {cartoon: {colorfunc: plddt}});
          var resi = [__RESI__];
          resi.forEach(function(r) {
            viewer.setStyle({resi: r}, {
              cartoon: {colorfunc: plddt},
              stick: {colorscheme: 'redCarbon', radius: 0.3},
              sphere: {color: '#ff3b3b', radius: 0.9}
            });
            viewer.addLabel('Res ' + r, {
              inFront: true, fontSize: 11, fontColor: '#ff6b6b',
              backgroundColor: '#0d1117', backgroundOpacity: 0.7,
              position: {resi: r}
            });
          });
          viewer.zoomTo();
          viewer.render();
          viewer.spin('y', 0.5);
        } catch (e) {
          document.getElementById('afviewer').innerHTML =
            '<p style=\"color:#8b98a5;font-family:monospace;padding:1rem\">'
            + '3D viewer needs an internet connection to load the library.</p>';
        }
      })();
    </script>
    """
    return (template
            .replace("__PDB__", pdb_js)
            .replace("__RESI__", resi_js)
            .replace("__MEAN__", mean_txt)
            .replace("__H__", str(int(height))))


# Functionally important TP53 residues (curated, for the mutation-aware viewer).
_DRUGGABLE_RESIDUES = {
    124: "Cys124 — APR-246/eprenetapopt reactivation site",
    175: "Arg175 — structural hotspot",
    220: "Tyr220 — PK11007/PhiKan cryptic pocket",
}
_ZINC_RESIDUES = {176: "Cys176", 179: "His179", 238: "Cys238", 242: "Cys242"}


def mutation_structure_html(pdb_text: Optional[str], mutation: str,
                            height: int = 480) -> str:
    """Mutation-aware 3D structure viewer. Colours the p53 fold by domain,
    highlights the *patient's* mutated residue in gold with a label, and marks
    the druggable reactivation/zinc sites in cyan. Shows the mutation's
    structural class in an overlay. Pure, never-empty, injection-safe.
    """
    from agents.tumor_board import parse_variant   # local import avoids cycle
    if not pdb_text:
        return ("<div style='padding:1rem;color:#8b98a5;font-family:monospace;"
                "background:#0d1117;border-radius:8px;'>Structure not loaded "
                "(offline or fetch failed).</div>")

    vp = parse_variant(mutation)
    mut_resi = vp.codon if vp.codon else 0
    mut_label = html.escape(str(vp.raw or mutation or "—"))
    klass = {"contact": "DNA-contact mutant", "conformational": "conformational mutant",
             "truncating": "truncating", "other_hotspot": "recurrent hotspot",
             "non_hotspot_missense": "non-hotspot missense",
             "unknown": "unclassified"}.get(vp.klass, vp.klass)
    dom = _domain_for_position(mut_resi) if mut_resi else None
    dom_name = dom["name"] if dom else "—"

    domains_js = ",".join(
        "{s:%d,e:%d,c:'%s'}" % (d["start"], d["end"], d["color"])
        for d in P53_DOMAINS)
    drug_js = ",".join(str(r) for r in _DRUGGABLE_RESIDUES)
    zinc_js = ",".join(str(r) for r in _ZINC_RESIDUES)
    pdb_js = json.dumps(pdb_text)

    template = """
    <style>
      .mv-ctrls{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 6px 0;}
      .mv-ctrls button{background:#161b22;color:#c9d1d9;border:1px solid #30363d;
        border-radius:6px;padding:4px 10px;font-family:sans-serif;font-size:11px;
        cursor:pointer;}
      .mv-ctrls button:hover{border-color:#22d3ee;color:#22d3ee;}
      .mv-ctrls button.on{background:#22d3ee;color:#03121a;border-color:#22d3ee;}
    </style>
    <div class="mv-ctrls">
      <button id="mv-cartoon" class="on" onclick="window.__mv&&window.__mv.rep('cartoon',this)">Cartoon</button>
      <button id="mv-surface" onclick="window.__mv&&window.__mv.rep('surface',this)">Surface</button>
      <button id="mv-stick" onclick="window.__mv&&window.__mv.rep('stick',this)">Sticks</button>
      <button onclick="window.__mv&&window.__mv.spin(this)">Spin ⏯</button>
      <button onclick="window.__mv&&window.__mv.reset()">Reset view</button>
    </div>
    <div style="position:relative;">
      <div id="mutview" style="width:100%;height:__H__px;background:#0d1117;
           border-radius:8px;"></div>
      <div style="position:absolute;top:8px;left:10px;background:rgba(13,17,23,0.88);
           padding:8px 11px;border-radius:6px;font-family:sans-serif;font-size:11px;
           color:#c9d1d9;line-height:1.6;">
        <b style="color:#ffd54a;">&#9679;</b> mutation <b>__MUT__</b><br>
        class: __KLASS__ · domain __DOM__<br>
        <b style="color:#22d3ee;">&#9679;</b> druggable / reactivation site<br>
        <b style="color:#a78bfa;">&#9679;</b> zinc-binding cluster
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
    <script>
      (function() {
        try {
          var el = document.getElementById('mutview');
          var viewer = $3Dmol.createViewer(el, {backgroundColor: '#0d1117'});
          viewer.addModel(__PDB__, 'pdb');
          var domains = [__DOMAINS__];
          var drug = [__DRUG__];
          var zinc = [__ZINC__];
          var mut = __MUTRESI__;
          var spinning = true;

          function base(cartoonColor){
            return cartoonColor ? {cartoon:{color:cartoonColor}} : {};
          }
          function apply(rep){
            viewer.removeAllSurfaces();
            viewer.removeAllLabels();
            // base representation
            if(rep === 'stick'){
              viewer.setStyle({}, {stick:{colorscheme:'default', radius:0.15}});
            } else {
              viewer.setStyle({}, {cartoon:{color:'#3a4250'}});
              domains.forEach(function(d){
                viewer.setStyle({resi:d.s+'-'+d.e}, {cartoon:{color:d.c}}); });
            }
            // druggable + zinc always shown as sticks for orientation
            drug.forEach(function(r){ viewer.addStyle({resi:r},
              {stick:{color:'#22d3ee', radius:0.25}}); });
            zinc.forEach(function(r){ viewer.addStyle({resi:r},
              {stick:{color:'#a78bfa', radius:0.25}}); });
            // patient mutation — gold + sphere + label
            if(mut > 0){
              viewer.addStyle({resi:mut}, {stick:{color:'#ffd54a', radius:0.4},
                sphere:{color:'#ffd54a', radius:1.0}});
              if(rep !== 'stick'){
                viewer.setStyle({resi:mut}, {cartoon:{color:'#ffd54a'},
                  stick:{color:'#ffd54a', radius:0.4},
                  sphere:{color:'#ffd54a', radius:1.0}}); }
              viewer.addLabel('__MUT__', {inFront:true, fontSize:13,
                fontColor:'#1a1a1a', backgroundColor:'#ffd54a',
                backgroundOpacity:0.95, position:{resi:mut}});
            }
            if(rep === 'surface'){
              viewer.addSurface($3Dmol.SurfaceType.VDW,
                {opacity:0.72, colorscheme:'whiteCarbon'});
            }
            viewer.render();
          }
          window.__mv = {
            rep: function(rep, btn){
              ['mv-cartoon','mv-surface','mv-stick'].forEach(function(id){
                var b=document.getElementById(id); if(b) b.className=''; });
              if(btn) btn.className='on';
              apply(rep);
            },
            spin: function(btn){ spinning=!spinning;
              viewer.spin(spinning?'y':false, 0.4);
              if(btn) btn.className = spinning ? '' : 'on'; },
            reset: function(){ viewer.zoomTo(mut>0?{resi:mut}:{}); viewer.render(); }
          };
          apply('cartoon');
          viewer.zoomTo(mut>0?{resi:mut}:{});
          viewer.render();
          viewer.spin('y', 0.4);
        } catch (e) {
          document.getElementById('mutview').innerHTML =
            '<p style=\"color:#8b98a5;font-family:monospace;padding:1rem\">'
            + '3D viewer needs an internet connection to load the library.</p>';
        }
      })();
    </script>
    """
    return (template
            .replace("__PDB__", pdb_js)
            .replace("__DOMAINS__", domains_js)
            .replace("__DRUG__", drug_js)
            .replace("__ZINC__", zinc_js)
            .replace("__MUTRESI__", str(int(mut_resi)))
            .replace("__MUT__", mut_label)
            .replace("__KLASS__", html.escape(klass))
            .replace("__DOM__", html.escape(dom_name))
            .replace("__H__", str(int(height))))


def plddt_profile_chart(per_residue: Optional[dict],
                        mean_plddt: Optional[float] = None) -> go.Figure:
    """Per-residue AlphaFold pLDDT confidence profile. Pure + never-empty."""
    if not per_residue:
        return _empty_fig("pLDDT profile — load the AlphaFold model")
    items = sorted(((int(k), float(v)) for k, v in per_residue.items()), key=lambda x: x[0])
    xs = [i for i, _ in items]
    ys = [v for _, v in items]

    def _col(v):
        if v > 90: return "#0053D6"
        if v > 70: return "#65CBF3"
        if v > 50: return "#FFDB13"
        return "#FF7D45"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color="#3a4756", width=1),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=4, color=[_col(v) for v in ys]),
        text=[f"residue {i}: pLDDT {v:.0f}" for i, v in items],
        hoverinfo="text", showlegend=False))
    for thr in (50, 70, 90):
        fig.add_hline(y=thr, line=dict(color="#2a3340", width=1, dash="dot"))
    title = "AlphaFold per-residue confidence (pLDDT)"
    if mean_plddt is not None:
        title += f" — mean {mean_plddt}"
    fig.update_layout(
        title=title, height=300, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Residue", yaxis_title="pLDDT",
        yaxis=dict(range=[0, 100]), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3"))
    return fig


def pathogenicity_gauge(significance: str, confidence: Optional[float] = None) -> go.Figure:
    """Benign -> Pathogenic gauge for a variant classification. Never empty."""
    pos = {
        "benign": 8, "likely_benign": 28, "vus": 50,
        "likely_pathogenic": 72, "pathogenic": 92,
    }
    s = str(significance or "").strip().lower().replace(" ", "_").replace("-", "_")
    val = pos.get(s, 50)
    label = s.replace("_", " ").title() if s else "Unknown"
    title = f"{label}" + (f" · conf {confidence:.0%}" if isinstance(confidence, (int, float)) else "")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"suffix": "/100", "font": {"size": 22}},
        title={"text": f"Pathogenicity — {title}", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [10, 30, 50, 72, 92],
                     "ticktext": ["Benign", "Likely<br>benign", "VUS",
                                  "Likely<br>path.", "Path."]},
            "bar": {"color": "#e6edf3", "thickness": 0.25},
            "steps": [
                {"range": [0, 20], "color": "#2ecc71"},
                {"range": [20, 40], "color": "#a9dfbf"},
                {"range": [40, 60], "color": "#f1c40f"},
                {"range": [60, 80], "color": "#e67e22"},
                {"range": [80, 100], "color": "#e74c3c"},
            ],
            "threshold": {"line": {"color": "#00d4ff", "width": 4},
                          "thickness": 0.8, "value": val},
        },
    ))
    fig.update_layout(template="plotly_dark", height=240,
                      margin=dict(l=20, r=20, t=50, b=10))
    return fig


def tme_donut(t_cell_fraction: float, immune_status: str = "") -> go.Figure:
    """Tumour-microenvironment composition donut (T-cell vs other). Never empty."""
    try:
        t = float(t_cell_fraction)
    except (TypeError, ValueError):
        t = 0.0
    t = max(0.0, min(1.0, t))
    status = str(immune_status or "").lower()
    hot = "#e74c3c" if "hot" in status else "#f39c12" if "inter" in status else "#3498db"
    fig = go.Figure(go.Pie(
        labels=["T-cell infiltrate", "Other / stroma"],
        values=[round(t * 100, 1), round((1 - t) * 100, 1)],
        hole=0.55, sort=False,
        marker=dict(colors=[hot, "#2a3340"]),
        textinfo="label+percent", textfont=dict(size=11),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    label = status.replace("-", " ").title() or "—"
    fig.update_layout(
        template="plotly_dark", height=300, showlegend=False,
        title=dict(text=f"Tumour Microenvironment — {label}", font=dict(size=14)),
        annotations=[dict(text=f"{t*100:.0f}%<br>T-cell", x=0.5, y=0.5,
                          font=dict(size=14, color="#e6edf3"), showarrow=False)],
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def vaf_gauge(vaf: float, mrd_threshold: float = 5.0) -> go.Figure:
    """Current ctDNA VAF as a burden gauge against the MRD threshold. Never empty."""
    try:
        v = float(vaf)
    except (TypeError, ValueError):
        v = 0.0
    v = max(0.0, min(100.0, v))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(v, 1),
        number={"suffix": "%", "font": {"size": 22}},
        title={"text": "ctDNA Tumour Burden (VAF)", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 60]},
            "bar": {"color": "#00d4ff", "thickness": 0.3},
            "steps": [
                {"range": [0, mrd_threshold], "color": "#1e3a2a"},
                {"range": [mrd_threshold, 20], "color": "#3a3320"},
                {"range": [20, 60], "color": "#3a1e1e"},
            ],
            "threshold": {"line": {"color": "#f39c12", "width": 4},
                          "thickness": 0.85, "value": mrd_threshold},
        },
    ))
    fig.update_layout(template="plotly_dark", height=240,
                      margin=dict(l=20, r=20, t=50, b=10))
    return fig


def pathway_diverging_bar(activated, suppressed) -> go.Figure:
    """Diverging bar of activated (red, +) vs suppressed (blue, -) p53 pathways."""
    act = [str(p) for p in (activated or [])][:8]
    sup = [str(p) for p in (suppressed or [])][:8]
    if not act and not sup:
        return _empty_fig("No pathway data")
    fig = go.Figure()
    if act:
        fig.add_trace(go.Bar(
            y=act, x=[1] * len(act), orientation="h", name="Activated",
            marker_color="#e74c3c", hovertemplate="%{y} (activated)<extra></extra>"))
    if sup:
        fig.add_trace(go.Bar(
            y=sup, x=[-1] * len(sup), orientation="h", name="Suppressed",
            marker_color="#3498db", hovertemplate="%{y} (suppressed)<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", height=max(240, 36 * (len(act) + len(sup)) + 60),
        barmode="overlay", title=dict(text="p53 Pathway Dysregulation", font=dict(size=14)),
        xaxis=dict(title="suppressed ← → activated", tickvals=[-1, 0, 1],
                   ticktext=["Suppressed", "", "Activated"], range=[-1.4, 1.4]),
        legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# Stable ISO-3 codes (the "country names" locationmode is deprecated in Plotly).
_ISO3 = {
    "Nigeria": "NGA", "Kenya": "KEN", "Senegal": "SEN", "Gambia": "GMB",
    "Mozambique": "MOZ", "Ghana": "GHA", "Tanzania": "TZA", "Malawi": "MWI",
    "Ethiopia": "ETH", "Uganda": "UGA", "South Africa": "ZAF", "Zambia": "ZMB",
    "Guinea": "GIN", "Mali": "MLI", "Zimbabwe": "ZWE", "Egypt": "EGY",
}


def synthetic_lethal_network(sl_result) -> go.Figure:
    """Radial TP53-centric synthetic-lethal network: TP53 at the centre, SL
    target genes around it; edge/node colour by evidence, node size by score.
    Never empty.
    """
    import math
    targets = (sl_result or {}).get("targets", []) if isinstance(sl_result, dict) else []
    targets = [t for t in targets if isinstance(t, dict)][:12]
    if not targets:
        return _empty_fig("No synthetic-lethal targets")

    ev_col = {"high": "#e74c3c", "medium": "#f39c12", "emerging": "#3498db"}
    n = len(targets)
    cx, cy = 0.0, 0.0
    xs = [math.cos(2 * math.pi * i / n - math.pi / 2) for i in range(n)]
    ys = [math.sin(2 * math.pi * i / n - math.pi / 2) for i in range(n)]

    fig = go.Figure()
    # edges TP53 -> each target
    for i, t in enumerate(targets):
        fig.add_trace(go.Scatter(
            x=[cx, xs[i]], y=[cy, ys[i]], mode="lines",
            line=dict(color=ev_col.get(t.get("evidence"), "#5a6b7a"),
                      width=1 + t.get("sl_score", 1) / 2),
            hoverinfo="skip", showlegend=False))
    # target nodes
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=[t.get("gene", "?") for t in targets], textposition="top center",
        textfont=dict(color="#c2ccd6", size=10, family="JetBrains Mono"),
        marker=dict(
            size=[16 + 3 * t.get("sl_score", 1) for t in targets],
            color=[ev_col.get(t.get("evidence"), "#5a6b7a") for t in targets],
            line=dict(color="#0d1117", width=1.5)),
        customdata=[[t.get("drug", "—"), t.get("evidence", "?")] for t in targets],
        hovertemplate="%{text} · %{customdata[1]} evidence<br>%{customdata[0]}<extra></extra>",
    ))
    # TP53 hub
    fig.add_trace(go.Scatter(
        x=[cx], y=[cy], mode="markers+text", text=["TP53"],
        textposition="middle center", textfont=dict(color="#0d1117", size=11),
        marker=dict(size=44, color="#00d4ff", line=dict(color="#0d1117", width=2)),
        hovertemplate="TP53 (mutant)<extra></extra>", showlegend=False))
    fig.update_layout(
        template="plotly_dark", height=460, showlegend=False,
        title=dict(text="Synthetic-lethal network — 🔴 high · 🟠 medium · 🔵 emerging",
                   font=dict(size=14)),
        xaxis=dict(visible=False, range=[-1.5, 1.5], fixedrange=True),
        yaxis=dict(visible=False, range=[-1.5, 1.5], scaleanchor="x", fixedrange=True),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def ind_section_chart(ind_result) -> go.Figure:
    """IND draft section coverage — each section green (drafted) or amber
    (placeholder / to-be-completed). Never empty.
    """
    draft = (ind_result or {}).get("draft", {}) if isinstance(ind_result, dict) else {}
    sections = draft.get("sections", []) if isinstance(draft, dict) else []
    if not sections:
        return _empty_fig("No IND sections")
    titles, colors, status = [], [], []
    for s in sections:
        content = str(s.get("content", ""))
        placeholder = "[TO BE COMPLETED" in content or "[Populate" in content
        titles.append(f"{s.get('number','?')}. {str(s.get('title',''))[:34]}")
        colors.append("#f39c12" if placeholder else "#2ecc71")
        status.append("placeholder" if placeholder else "drafted")
    titles = titles[::-1]; colors = colors[::-1]; status = status[::-1]
    fig = go.Figure(go.Bar(
        x=[1] * len(titles), y=titles, orientation="h",
        marker=dict(color=colors), customdata=status,
        hovertemplate="%{y}: %{customdata}<extra></extra>",
        text=status, textposition="inside", insidetextanchor="start",
    ))
    fig.update_layout(
        template="plotly_dark", height=max(220, 34 * len(titles) + 70),
        title=dict(text="IND draft coverage — 🟢 drafted · 🟠 to complete",
                   font=dict(size=14)),
        xaxis=dict(visible=False, range=[0, 1]), margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def structural_profile_radar(struct_result) -> go.Figure:
    """Radar of a mutation's structural-mechanics profile: destabilisation
    (ΔΔG), druggability, cavity volume and pocket hydrophobicity (each
    normalised 0–1). Never empty.
    """
    r = struct_result if isinstance(struct_result, dict) else {}
    if not r:
        return _empty_fig("No structural profile")

    def _n(v, lo, hi):
        try:
            return max(0.0, min(1.0, (float(v) - lo) / (hi - lo)))
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0

    axes = ["Destabilisation<br>(ΔΔG)", "Druggability", "Cavity volume", "Hydrophobicity"]
    vals = [
        _n(r.get("ddG_kcal_mol"), 0, 5),
        _n(r.get("druggability"), 0, 1),
        _n(r.get("cavity_volume_A3"), 0, 300),
        _n(r.get("hydrophobicity"), 0, 1),
    ]
    mut = r.get("mutation", "?")
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=axes + [axes[0]], fill="toself",
        line=dict(color="#00d4ff"), fillcolor="rgba(0,212,255,0.25)",
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", height=340,
        title=dict(text=f"Structural profile — {mut} ({r.get('stability_class','?')})",
                   font=dict(size=14)),
        polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False,
                                   gridcolor="#2a3340"),
                   angularaxis=dict(gridcolor="#2a3340")),
        margin=dict(l=40, r=40, t=60, b=30), showlegend=False,
    )
    return fig


def docking_affinity_gauge(docking_result) -> go.Figure:
    """Binding-affinity gauge (kcal/mol) for a docking result. More negative =
    stronger binding (green); the docking method is shown in the title so a
    real Vina run vs a heuristic estimate is never ambiguous. Never empty.
    """
    r = docking_result if isinstance(docking_result, dict) else {}
    aff = r.get("binding_affinity")
    try:
        val = float(aff)
    except (TypeError, ValueError):
        return _empty_fig("No binding affinity")
    method = r.get("method", "")
    label = "AutoDock Vina" if method == "autodock_vina" else "Heuristic estimate"
    drug = r.get("drug", "ligand")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(val, 2),
        number={"suffix": " kcal/mol", "font": {"size": 20}},
        title={"text": f"{drug} — binding affinity<br><span style='font-size:11px'>"
                       f"{label}</span>", "font": {"size": 14}},
        gauge={
            "axis": {"range": [-12, 0]},
            "bar": {"color": "#00d4ff", "thickness": 0.3},
            "steps": [
                {"range": [-12, -9], "color": "#1e3a2a"},   # very strong
                {"range": [-9, -7], "color": "#274d35"},
                {"range": [-7, -5], "color": "#3a3320"},
                {"range": [-5, 0], "color": "#3a1e1e"},      # weak
            ],
            "threshold": {"line": {"color": "#2ecc71", "width": 4},
                          "thickness": 0.85, "value": val},
        },
    ))
    fig.update_layout(template="plotly_dark", height=260,
                      margin=dict(l=20, r=20, t=60, b=10))
    return fig


def vcf_variant_chart(variants) -> go.Figure:
    """TP53 variants from a VCF by QUAL, coloured by significance
    (hotspot = red, annotated = blue, unannotated = grey). Never empty.
    """
    rows = [v for v in (variants or []) if isinstance(v, dict)]
    if not rows:
        return _empty_fig("No TP53 variants in VCF")

    def _qual(v):
        try:
            return float(v.get("qual"))
        except (TypeError, ValueError):
            return 0.0

    def _col(v):
        if v.get("is_hotspot"):
            return "#e74c3c"
        return "#00d4ff" if v.get("annotated") else "#5a6b7a"

    rows = sorted(rows, key=_qual)[-16:]
    labels = []
    seen: dict = {}
    for v in rows:
        base = v.get("amino_acid_change") or f"{v.get('chrom','?')}:{v.get('pos','?')}"
        seen[base] = seen.get(base, 0) + 1
        labels.append(base if seen[base] == 1 else f"{base}·{seen[base]}")

    fig = go.Figure(go.Bar(
        x=[_qual(v) for v in rows], y=labels, orientation="h",
        marker=dict(color=[_col(v) for v in rows]),
        customdata=[[v.get("ref", "?") + ">" + v.get("alt", "?"),
                     v.get("filter", "?"),
                     "hotspot" if v.get("is_hotspot") else
                     ("annotated" if v.get("annotated") else "unannotated")]
                    for v in rows],
        hovertemplate="%{y} (%{customdata[0]})<br>QUAL %{x} · %{customdata[1]} · %{customdata[2]}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", height=max(220, 32 * len(rows) + 70),
        title=dict(text="TP53 variants — hotspot 🔴 · annotated 🔵 · unannotated ⚪",
                   font=dict(size=14)),
        xaxis_title="QUAL", margin=dict(l=10, r=20, t=50, b=10),
    )
    return fig


def trials_priority_chart(trials) -> go.Figure:
    """Recruiting trials by clinical phase, coloured by geographic priority
    (Kenya = gold, other African = teal, international = grey). Never empty.
    """
    rows = [t for t in (trials or []) if isinstance(t, dict)]
    if not rows:
        return _empty_fig("No trials to plot")
    rows = sorted(rows, key=lambda t: t.get("phase_rank", 0))[-14:]

    def _col(t):
        if t.get("kenya_site"):
            return "#f1c40f"
        if t.get("african_priority"):
            return "#1abc9c"
        return "#5a6b7a"

    labels = [(t.get("nct_id") or "search")[:14] for t in rows]
    # de-duplicate y labels (Plotly needs distinct categories)
    seen: dict = {}
    ylabels = []
    for lbl in labels:
        seen[lbl] = seen.get(lbl, 0) + 1
        ylabels.append(lbl if seen[lbl] == 1 else f"{lbl}·{seen[lbl]}")

    fig = go.Figure(go.Bar(
        x=[t.get("phase_rank", 0) for t in rows], y=ylabels, orientation="h",
        marker=dict(color=[_col(t) for t in rows]),
        customdata=[[t.get("title", "")[:60], t.get("phase", "?"),
                     ", ".join(t.get("countries", [])[:3]) or "—"] for t in rows],
        hovertemplate="%{customdata[0]}<br>%{customdata[1]} · %{customdata[2]}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", height=max(240, 34 * len(rows) + 80),
        title=dict(text="Recruiting trials — Kenya 🟡 · Africa 🟢 · International ⚪",
                   font=dict(size=14)),
        xaxis=dict(title="Clinical phase", tickvals=[0, 1, 2, 3, 4],
                   ticktext=["—", "I", "II", "III", "IV"], range=[0, 4.6]),
        margin=dict(l=10, r=20, t=50, b=10),
    )
    return fig


def chembl_phase_chart(compounds) -> go.Figure:
    """Horizontal bar of TP53-pathway compounds by clinical phase (0-4).

    Coloured green→red by maturity (Approved = green). Never empty.
    """
    rows = [c for c in (compounds or []) if isinstance(c, dict)]
    rows = [c for c in rows if isinstance(c.get("max_phase"), (int, float))]
    if not rows:
        return _empty_fig("No clinical-phase data")
    rows = sorted(rows, key=lambda c: c.get("max_phase", 0))[-14:]
    names = [str(c.get("name", "?"))[:34] for c in rows]
    phases = [c.get("max_phase", 0) for c in rows]
    phase_colour = {4: "#2ecc71", 3: "#7ac943", 2: "#f1c40f",
                    1: "#e67e22", 0: "#8b98a5"}
    fig = go.Figure(go.Bar(
        x=phases, y=names, orientation="h",
        marker=dict(color=[phase_colour.get(p, "#8b98a5") for p in phases]),
        customdata=[c.get("phase_label", "?") for c in rows],
        hovertemplate="%{y}: %{customdata}<extra></extra>",
        text=[c.get("phase_label", "") for c in rows], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", height=max(240, 34 * len(rows) + 70),
        title=dict(text="TP53-pathway drugs by clinical phase (ChEMBL)",
                   font=dict(size=14)),
        xaxis=dict(title="Clinical phase", tickvals=[0, 1, 2, 3, 4],
                   ticktext=["Preclin.", "I", "II", "III", "Approved"],
                   range=[0, 4.6]),
        margin=dict(l=10, r=40, t=50, b=10),
    )
    return fig


def clinvar_conflict_chart(findings) -> go.Figure:
    """Dumbbell chart: AI vs ClinVar classification per mutation.

    Each row shows the AI's call (◆) and ClinVar's call (●) on a
    Benign → Uncertain → Pathogenic axis; a red connector = conflict,
    green = concordant. Never empty.
    """
    rows = [f for f in (findings or []) if isinstance(f, dict)]
    if not rows:
        return _empty_fig("No classifications to compare")

    bucket = {"pathogenic": 2, "likely_pathogenic": 2,
              "benign": 0, "likely_benign": 0,
              "vus": 1, "uncertain": 1}

    def _pos(label):
        s = str(label or "").lower().replace(" ", "_").replace("-", "_")
        for k, v in bucket.items():
            if k in s:
                return v
        return 1  # unknown / not-stated -> uncertain column

    muts = [f.get("mutation", "?") for f in rows]
    ai_x = [_pos(f.get("ai_classification")) for f in rows]
    cv_x = [_pos(f.get("clinvar_classification")) for f in rows]

    fig = go.Figure()
    for i, f in enumerate(rows):
        conflict = bool(f.get("conflict"))
        fig.add_trace(go.Scatter(
            x=[ai_x[i], cv_x[i]], y=[muts[i], muts[i]], mode="lines",
            line=dict(color="#e74c3c" if conflict else "#2ecc71", width=4),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=cv_x, y=muts, mode="markers", name="ClinVar",
        marker=dict(symbol="circle", size=15, color="#00d4ff",
                    line=dict(color="#0d1117", width=1)),
        hovertemplate="%{y} · ClinVar<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ai_x, y=muts, mode="markers", name="AI",
        marker=dict(symbol="diamond", size=14, color="#f39c12",
                    line=dict(color="#0d1117", width=1)),
        hovertemplate="%{y} · AI<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", height=max(220, 52 * len(rows) + 80),
        title=dict(text="AI vs ClinVar — classification concordance", font=dict(size=14)),
        xaxis=dict(tickvals=[0, 1, 2], ticktext=["Benign", "Uncertain", "Pathogenic"],
                   range=[-0.4, 2.4]),
        legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def african_atlas_map(country_burden: dict) -> go.Figure:
    """Africa choropleth coloured by representative TP53-cancer burden (0-100).

    `country_burden`: {country_name: score}. Never empty.
    """
    cb = {k: v for k, v in dict(country_burden or {}).items() if k in _ISO3}
    if not cb:
        return _empty_fig("No regional data")
    names = list(cb.keys())
    fig = go.Figure(go.Choropleth(
        locations=[_ISO3[n] for n in names],
        text=names,
        z=list(cb.values()),
        colorscale="YlOrRd",
        zmin=0, zmax=100,
        marker_line_color="#0d1117", marker_line_width=0.5,
        colorbar=dict(title="Burden", thickness=12, len=0.7),
        hovertemplate="%{text}: %{z}/100<extra></extra>",
    ))
    fig.update_geos(
        scope="africa", bgcolor="#0d1117",
        showcountries=True, countrycolor="#2a3340",
        showframe=False, showcoastlines=False, landcolor="#161b22",
    )
    fig.update_layout(
        template="plotly_dark", height=460,
        title=dict(text="African TP53 / Cancer Burden (representative)",
                   font=dict(color="#e6edf3", size=15)),
        margin=dict(l=0, r=0, t=46, b=0),
        annotations=[dict(text="Curated/illustrative — not measured incidence",
                          x=0.5, y=-0.02, xref="paper", yref="paper",
                          showarrow=False, font=dict(color="#8b98a5", size=9))],
    )
    return fig


def african_burden_bar(matched_profiles) -> go.Figure:
    """Horizontal bar of relative burden per matched African TP53 profile."""
    rows = [p for p in (matched_profiles or []) if isinstance(p, dict)]
    if not rows:
        return _empty_fig("No profiles to plot")
    rows = sorted(rows, key=lambda p: p.get("burden_score", 0))
    titles = [str(p.get("title", p.get("id", "?")))[:42] for p in rows]
    scores = [p.get("burden_score", 0) for p in rows]
    lo, hi = (min(scores), max(scores)) if scores else (0, 1)

    def _col(v):
        t = 0.0 if hi == lo else (v - lo) / (hi - lo)
        return f"rgb({int(120 + 135 * t)},{int(180 - 140 * t)},60)"

    fig = go.Figure(go.Bar(
        x=scores, y=titles, orientation="h",
        marker=dict(color=[_col(v) for v in scores]),
        text=scores, textposition="outside",
        hovertemplate="%{y}: %{x}/100<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", height=max(240, 60 * len(rows) + 60),
        title=dict(text="Relative regional burden", font=dict(size=14)),
        xaxis=dict(title="Burden (0-100)", range=[0, 100]),
        margin=dict(l=10, r=30, t=50, b=10),
    )
    return fig


def tnm_stage_bar(stage_group: str) -> go.Figure:
    """Horizontal stage-progression gauge (I → IV) with the patient's stage
    highlighted. Colour runs green (early) → red (advanced). Never empty.
    """
    stages = ["I", "II", "III", "IV"]
    colours = {"I": "#2ecc71", "II": "#f1c40f", "III": "#e67e22", "IV": "#e74c3c"}
    # Map full group (e.g. "IIIB") to its coarse Roman stage for positioning.
    sg = str(stage_group or "").upper().strip()
    coarse = "IV" if sg.startswith("IV") else \
             "III" if sg.startswith("III") else \
             "II" if sg.startswith("II") else \
             "I" if sg.startswith("I") else ""
    active = stages.index(coarse) if coarse in stages else -1

    xs = list(range(len(stages)))
    fig = go.Figure()
    # connecting track
    fig.add_trace(go.Scatter(x=xs, y=[0] * len(stages), mode="lines",
                             line=dict(color="#2a3340", width=4), hoverinfo="skip"))
    # stage nodes
    fig.add_trace(go.Scatter(
        x=xs, y=[0] * len(stages), mode="markers+text",
        marker=dict(
            size=[44 if i == active else 26 for i in range(len(stages))],
            color=[colours[s] for s in stages],
            line=dict(color=["#e6edf3" if i == active else "#0d1117"
                             for i in range(len(stages))],
                      width=[3 if i == active else 1 for i in range(len(stages))]),
        ),
        text=[f"<b>{s}</b>" for s in stages], textposition="middle center",
        textfont=dict(color="#0d1117", size=12, family="JetBrains Mono"),
        hovertemplate="Stage %{text}<extra></extra>",
    ))
    label = f"Stage {sg}" if sg else "Stage —"
    fig.update_layout(
        template="plotly_dark", height=150, showlegend=False,
        title=dict(text=f"Stage progression — {label}",
                   font=dict(color="#e6edf3", size=14)),
        margin=dict(l=20, r=20, t=46, b=10),
        xaxis=dict(visible=False, range=[-0.5, len(stages) - 0.5], fixedrange=True),
        yaxis=dict(visible=False, range=[-1, 1], fixedrange=True),
    )
    return fig


# ── Drug docking (illustrative) ───────────────────────────────────
# NOTE: the binding affinities below are ILLUSTRATIVE / heuristic, not the
# output of a real docking engine (AutoDock Vina is a separate agent). They
# are deterministic so they can be unit-tested and reproduced.

# Residue-class buckets used to weight which mechanism fits which mutation.
_ZINC_CODONS = {176, 179, 238, 242}
_CONTACT_CODONS = {248, 273, 249}
_CONFORMATIONAL_CODONS = {143, 175, 245, 282}
_LOF_CLASSES = {"y220c", "zinc", "contact", "conformational"}

TP53_DRUGS = [
    {"name": "PC14586 (Rezatapopt)", "mechanism": "Y220C pocket stabiliser",
     "base": -6.0, "fit": {"y220c": -3.4}},
    {"name": "APR-246 (Eprenetapopt)", "mechanism": "Mutant p53 reactivator (thiol refolding)",
     "base": -7.0, "fit": {"conformational": -1.8, "zinc": -1.2, "y220c": -1.0}},
    {"name": "COTI-2", "mechanism": "Zinc-metallochaperone reactivator",
     "base": -6.6, "fit": {"zinc": -2.0, "conformational": -1.0}},
    {"name": "Idasanutlin", "mechanism": "MDM2 inhibitor (requires functional p53)",
     "base": -6.4, "fit": {}, "lof_penalty": 2.5},
    {"name": "Carboplatin", "mechanism": "DNA cross-linker (mutation-agnostic chemo)",
     "base": -5.4, "fit": {}},
]


def mutation_class(mutation: str) -> str:
    """Classify a TP53 variant into a mechanism-relevant bucket.

    Returns one of: y220c, zinc, contact, conformational, other. Never raises.
    """
    digits = "".join(ch for ch in str(mutation or "") if ch.isdigit())
    codon = int(digits) if digits else 0
    if codon == 220:
        return "y220c"
    if codon in _ZINC_CODONS:
        return "zinc"
    if codon in _CONTACT_CODONS:
        return "contact"
    if codon in _CONFORMATIONAL_CODONS:
        return "conformational"
    return "other"


def dock_candidates(mutation: str) -> list:
    """Rank candidate drugs for a mutation by illustrative binding affinity.

    More-negative affinity = stronger predicted binding = better rank. The
    ranking shifts with the mutation class so the *why* is visible (e.g. the
    Y220C stabiliser wins for Y220C; the MDM2 inhibitor is penalised on any
    loss-of-function mutant). Always returns a non-empty, sorted list.
    """
    cls = mutation_class(mutation)
    out = []
    for d in TP53_DRUGS:
        affinity = d["base"] + d.get("fit", {}).get(cls, 0.0)
        penalty = d.get("lof_penalty", 0.0) if cls in _LOF_CLASSES else 0.0
        affinity += penalty
        if d.get("fit", {}).get(cls):
            rationale = f"Mechanism fits a {cls} mutant → stronger predicted binding."
        elif penalty:
            rationale = "Needs functional p53; penalised on a loss-of-function mutant."
        else:
            rationale = "Mutation-agnostic baseline."
        out.append({
            "name": d["name"], "mechanism": d["mechanism"],
            "affinity": round(affinity, 1), "fit_class": cls, "rationale": rationale,
        })
    out.sort(key=lambda r: r["affinity"])  # most negative first
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


def docking_affinity_chart(candidates) -> go.Figure:
    """Horizontal bar of illustrative binding affinity (kcal/mol).

    Lower (more negative) = stronger binding; the top bar is the best
    candidate. Greens = stronger, reds = weaker. Never empty.
    """
    candidates = list(candidates or [])
    if not candidates:
        return _empty_fig("No drug candidates")
    ordered = sorted(candidates, key=lambda r: r.get("affinity", 0), reverse=True)
    names = [r.get("name", "?") for r in ordered]
    vals = [r.get("affinity", 0.0) for r in ordered]
    # Strongest (most negative) -> green; weakest -> red.
    lo, hi = min(vals), max(vals)
    def _col(v):
        t = 0.0 if hi == lo else (v - lo) / (hi - lo)  # 0 = strongest
        return f"rgb({int(40 + 200 * t)},{int(200 - 150 * t)},90)"
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=[_col(v) for v in vals]),
        hovertemplate="%{y}: %{x:.1f} kcal/mol<extra></extra>",
        text=[f"{v:.1f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", height=300,
        title="Illustrative Binding Affinity (kcal/mol) — lower = stronger",
        xaxis_title="Predicted ΔG (kcal/mol)", margin=dict(l=10, r=10, t=50, b=10),
        annotations=[dict(text="Heuristic estimate — not real docking",
                          x=1.0, y=1.12, xref="paper", yref="paper",
                          showarrow=False, font=dict(color="#8b98a5", size=9))],
    )
    return fig


def docking_pose_html(pdb_id: str, residues, drug_name: str = "",
                      affinity: float = 0.0) -> str:
    """Illustrative 3Dmol docking pose: protein cartoon + highlighted binding
    pocket + a translucent ligand cloud over the pocket residues.

    Not a real docked pose — it visualises *where* the drug is proposed to act.
    All injected values are sanitised (pdb alnum, residues ints, name charset).
    """
    safe_pdb = "".join(ch for ch in str(pdb_id) if ch.isalnum())[:8] or "2OCJ"
    resi = [int(r) for r in (residues or [])]
    resi_js = ",".join(str(r) for r in resi)
    safe_name = "".join(ch for ch in str(drug_name) if ch.isalnum() or ch in " ()-+.")[:40]
    label = f"{safe_name} (~{float(affinity):.1f} kcal/mol)" if safe_name else "ligand"
    return f"""
    <div id="dockview" style="width:100%;height:520px;background:#0b1622;
         border-radius:8px;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
    <script>
      (function() {{
        try {{
          var el = document.getElementById('dockview');
          // Slightly lighter background for contrast against the structure.
          var viewer = $3Dmol.createViewer(el, {{backgroundColor: '#0b1622'}});
          $3Dmol.download('pdb:{safe_pdb}', viewer, {{}}, function() {{
            // Brighter spectrum cartoon (was a dim flat grey at 0.55 opacity).
            viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum', opacity: 0.95}}}});
            var resi = [{resi_js}];
            resi.forEach(function(r) {{
              // Bold cyan binding-site sticks.
              viewer.setStyle({{resi: r}},
                {{cartoon: {{color: '#00e0ff'}},
                 stick: {{colorscheme: 'cyanCarbon', radius: 0.45}}}});
              // Bright, more opaque 'ligand cloud' over the binding pocket.
              viewer.addStyle({{resi: r}},
                {{sphere: {{color: '#ffd23f', radius: 1.9, opacity: 0.6}}}});
            }});
            if (resi.length) {{
              viewer.addLabel('{label}', {{
                inFront: true, fontSize: 13, fontColor: '#ffd23f',
                backgroundColor: '#0b1622', backgroundOpacity: 0.85,
                borderColor: '#ffd23f', borderThickness: 0.5,
                position: {{resi: resi[0]}}
              }});
            }}
            // Fit the whole structure so it opens at normal size (not a
            // close-up of the pocket). Users can scroll to zoom in.
            viewer.zoomTo();
            viewer.render();
            viewer.spin('y', 0.5);
          }});
        }} catch (e) {{
          document.getElementById('dockview').innerHTML =
            '<p style=\"color:#8b98a5;font-family:monospace;padding:1rem\">'
            + '3D docking view needs an internet connection.</p>';
        }}
      }})();
    </script>
    """


# ── Live AI Tumour Board (the hero visual) ────────────────────────
_THEME_TAG_COLOR = {
    "p53-reactivation pathway (trial / molecular board referral)": "#00d4ff",
    "stage-directed standard of care with TP53-aware prognosis": "#2ecc71",
    "reclassify the variant before acting (insufficient evidence)": "#f1c40f",
}


def _conf_color(c: float) -> str:
    """Confidence → colour ramp (red→amber→green)."""
    try:
        c = float(c)
    except (TypeError, ValueError):
        c = 0.0
    if c >= 0.75:
        return "#2ecc71"
    if c >= 0.5:
        return "#f1c40f"
    return "#ff6b6b"


def tumor_board_html(board: Optional[dict], height: int = 720) -> str:
    """Self-contained HTML for the Live AI Tumour Board — specialist cards with
    confidence meters, a sequentially-revealed debate stream, and a consensus
    banner. Year-2100 aesthetic. Pure + never-empty; injection-safe (all
    case-derived text is HTML-escaped).
    """
    board = board or {}
    members = board.get("members") or []
    debate = board.get("debate") or []
    consensus = board.get("consensus") or {}
    mutation = html.escape(str(board.get("mutation") or "—"))

    if not members:
        return ("<div style='padding:24px;color:#8b98a5;font-family:sans-serif;"
                "background:#0d1117;border-radius:12px'>No case convened yet — "
                "enter a TP53 mutation to assemble the tumour board.</div>")

    # ── Member cards ─────────────────────────────────────────────
    cards = []
    for i, m in enumerate(members):
        conf = float(m.get("confidence", 0.0))
        conf_pct = max(0, min(100, round(conf * 100)))
        ccol = _conf_color(conf)
        rec = m.get("recommendation", "")
        tag_col = _THEME_TAG_COLOR.get(rec, "#8b98a5")
        concerns = "".join(
            f"<li>{html.escape(str(c))}</li>" for c in (m.get("concerns") or [])
        )
        concerns_block = (f"<ul class='tb-concerns'>{concerns}</ul>"
                          if concerns else "")
        cards.append(
            "<div class='tb-card' style='animation-delay:%dms'>" % (i * 180)
            + f"<div class='tb-card-head'><span class='tb-icon'>"
              f"{html.escape(str(m.get('icon','•')))}</span>"
              f"<div><div class='tb-name'>{html.escape(str(m.get('member','')))}</div>"
              f"<div class='tb-spec'>{html.escape(str(m.get('specialty','')))}</div>"
              f"</div></div>"
            + f"<div class='tb-stance'>{html.escape(str(m.get('stance','')))}</div>"
            + f"<div class='tb-rationale'>{html.escape(str(m.get('rationale','')))}</div>"
            + "<div class='tb-conf-row'><span class='tb-conf-label'>confidence</span>"
              f"<div class='tb-conf-track'><div class='tb-conf-fill' "
              f"style='width:{conf_pct}%;background:{ccol}'></div></div>"
              f"<span class='tb-conf-val' style='color:{ccol}'>{conf_pct}%</span></div>"
            + f"<div class='tb-tag' style='border-color:{tag_col};color:{tag_col}'>"
              f"{html.escape(str(rec))}</div>"
            + concerns_block
            + "</div>"
        )
    cards_html = "\n".join(cards)

    # ── Debate stream ────────────────────────────────────────────
    base_delay = len(members) * 180 + 200
    bubbles = []
    for j, d in enumerate(debate):
        kind = d.get("type", "note")
        icon = {"agreement": "🤝", "challenge": "⚔️", "note": "📋"}.get(kind, "💬")
        col = {"agreement": "#2ecc71", "challenge": "#ff6b6b"}.get(kind, "#8b98a5")
        bubbles.append(
            "<div class='tb-bubble' style='animation-delay:%dms;border-left-color:%s'>"
            % (base_delay + j * 320, col)
            + f"<span class='tb-bubble-ic'>{icon}</span>"
            + f"<span>{html.escape(str(d.get('text','')))}</span></div>"
        )
    bubbles_html = "\n".join(bubbles)

    # ── Consensus banner ─────────────────────────────────────────
    cc = float(consensus.get("confidence", 0.0))
    cc_pct = max(0, min(100, round(cc * 100)))
    cc_col = _conf_color(cc)
    agree_pct = round(float(consensus.get("agreement_ratio", 0.0)) * 100)
    rec_text = html.escape(str(consensus.get("recommendation", "—")))
    rationale = html.escape(str(consensus.get("rationale", "")))
    dissents = consensus.get("dissents") or []
    dissent_html = ("<div class='tb-dissent'>Dissent: "
                    + html.escape("; ".join(str(x) for x in dissents)) + "</div>"
                    ) if dissents else ""
    consensus_delay = base_delay + len(debate) * 320 + 200

    template = """
<div class="tb-root">
  <style>
    .tb-root{font-family:'Inter',system-ui,sans-serif;background:radial-gradient(
        circle at 20% 0%,#16203a 0%,#0d1117 60%);border-radius:14px;padding:18px;
        color:#e6edf3;border:1px solid #1f2937;}
    .tb-title{font-size:1.05rem;font-weight:700;letter-spacing:.3px;margin-bottom:2px;}
    .tb-sub{font-size:.78rem;color:#8b98a5;margin-bottom:14px;}
    .tb-mut{color:#00d4ff;font-family:'JetBrains Mono',monospace;}
    .tb-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));
        gap:12px;margin-bottom:16px;}
    .tb-card{background:rgba(22,32,58,.55);border:1px solid #25304a;border-radius:11px;
        padding:13px;opacity:0;transform:translateY(10px);
        animation:tbIn .5s ease forwards;backdrop-filter:blur(4px);}
    .tb-card-head{display:flex;gap:9px;align-items:center;margin-bottom:8px;}
    .tb-icon{font-size:1.4rem;}
    .tb-name{font-weight:650;font-size:.9rem;}
    .tb-spec{font-size:.7rem;color:#8b98a5;}
    .tb-stance{font-size:.82rem;font-weight:600;color:#cdd9e5;margin-bottom:5px;}
    .tb-rationale{font-size:.74rem;color:#9aa7b4;line-height:1.4;margin-bottom:9px;}
    .tb-conf-row{display:flex;align-items:center;gap:7px;margin-bottom:9px;}
    .tb-conf-label{font-size:.62rem;text-transform:uppercase;color:#6b7685;
        letter-spacing:.5px;}
    .tb-conf-track{flex:1;height:6px;background:#1b2435;border-radius:4px;overflow:hidden;}
    .tb-conf-fill{height:100%;border-radius:4px;animation:tbGrow 1s ease forwards;}
    .tb-conf-val{font-size:.72rem;font-weight:700;font-family:'JetBrains Mono',monospace;}
    .tb-tag{font-size:.66rem;border:1px solid;border-radius:20px;padding:3px 9px;
        display:inline-block;margin-bottom:6px;}
    .tb-concerns{margin:4px 0 0;padding-left:16px;font-size:.68rem;color:#c9a24a;}
    .tb-stream{margin:6px 0 16px;}
    .tb-bubble{background:rgba(13,17,23,.6);border-left:3px solid;border-radius:7px;
        padding:8px 11px;margin-bottom:7px;font-size:.78rem;color:#cdd9e5;
        display:flex;gap:8px;opacity:0;animation:tbIn .45s ease forwards;}
    .tb-bubble-ic{flex-shrink:0;}
    .tb-consensus{background:linear-gradient(135deg,rgba(0,212,255,.08),
        rgba(46,204,113,.06));border:1px solid #2a3a52;border-radius:12px;padding:16px;
        opacity:0;animation:tbIn .6s ease forwards;}
    .tb-consensus-h{font-size:.7rem;text-transform:uppercase;letter-spacing:1px;
        color:#8b98a5;margin-bottom:6px;}
    .tb-rec{font-size:1rem;font-weight:700;margin-bottom:8px;}
    .tb-meters{display:flex;gap:22px;margin:10px 0;flex-wrap:wrap;}
    .tb-meter-v{font-size:1.5rem;font-weight:800;font-family:'JetBrains Mono',monospace;}
    .tb-meter-l{font-size:.66rem;color:#8b98a5;text-transform:uppercase;}
    .tb-crationale{font-size:.8rem;color:#b6c2cf;line-height:1.5;}
    .tb-dissent{font-size:.72rem;color:#c9a24a;margin-top:8px;}
    .tb-foot{font-size:.66rem;color:#6b7685;margin-top:14px;border-top:1px solid #1f2937;
        padding-top:9px;}
    @keyframes tbIn{to{opacity:1;transform:translateY(0);}}
    @keyframes tbGrow{from{width:0;}}
  </style>
  <div class="tb-title">🧑‍⚕️ Live AI Tumour Board</div>
  <div class="tb-sub">Case: <span class="tb-mut">__MUT__</span> · six specialists deliberate, then vote toward consensus</div>
  <div class="tb-grid">__CARDS__</div>
  <div class="tb-stream">__BUBBLES__</div>
  <div class="tb-consensus" style="animation-delay:__CDELAY__ms">
    <div class="tb-consensus-h">⚖️ Consensus recommendation</div>
    <div class="tb-rec" style="color:__CCOL__">__REC__</div>
    <div class="tb-meters">
      <div><div class="tb-meter-v" style="color:__CCOL__">__CPCT__%</div>
           <div class="tb-meter-l">confidence</div></div>
      <div><div class="tb-meter-v">__APCT__%</div>
           <div class="tb-meter-l">panel agreement</div></div>
    </div>
    <div class="tb-crationale">__RATIONALE__</div>
    __DISSENT__
  </div>
  <div class="tb-foot">__DISCLAIMER__</div>
</div>
"""
    return (template
            .replace("__MUT__", mutation)
            .replace("__CARDS__", cards_html)
            .replace("__BUBBLES__", bubbles_html)
            .replace("__CDELAY__", str(consensus_delay))
            .replace("__CCOL__", cc_col)
            .replace("__REC__", rec_text)
            .replace("__CPCT__", str(cc_pct))
            .replace("__APCT__", str(agree_pct))
            .replace("__RATIONALE__", rationale)
            .replace("__DISSENT__", dissent_html)
            .replace("__DISCLAIMER__",
                     html.escape(str(board.get("disclaimer", "")))))


# ── Explainability ("Why?") panel ─────────────────────────────────
_STRENGTH_STYLE = {
    "strong":     ("#2ecc71", "STRONG"),
    "moderate":   ("#00d4ff", "MODERATE"),
    "supporting": ("#f1c40f", "SUPPORTING"),
    "uncertain":  ("#ff8c6b", "UNCERTAIN"),
}


def explainability_panel_html(exp: Optional[dict], height: int = 640) -> str:
    """Self-contained HTML for the Explainability 'Why?' trace — headline
    classification with confidence, evidence lines tagged by strength, the
    perturbed pathways, citations and an explicit uncertainty list. Pure +
    never-empty + injection-safe.
    """
    exp = exp or {}
    evidence = exp.get("evidence") or []
    if not exp.get("classification") and not evidence:
        return ("<div style='padding:22px;color:#8b98a5;font-family:sans-serif;"
                "background:#0d1117;border-radius:12px'>No explanation yet — "
                "assess a variant to see why.</div>")

    mutation = html.escape(str(exp.get("mutation") or "—"))
    headline = html.escape(str(exp.get("headline") or "—"))
    conf = float(exp.get("confidence", 0.0))
    conf_pct = max(0, min(100, round(conf * 100)))
    ccol = _conf_color(conf)

    rows = []
    for e in evidence:
        col, lbl = _STRENGTH_STYLE.get(e.get("strength", "uncertain"),
                                       ("#8b98a5", "?"))
        rows.append(
            "<div class='ex-row'>"
            f"<span class='ex-badge' style='background:{col}1a;color:{col};"
            f"border-color:{col}'>{lbl}</span>"
            f"<div class='ex-body'><span class='ex-src'>"
            f"{html.escape(str(e.get('source','')))}</span>"
            f"<span class='ex-cat'>{html.escape(str(e.get('category','')))}</span>"
            f"<div class='ex-stmt'>{html.escape(str(e.get('statement','')))}</div>"
            "</div></div>"
        )
    rows_html = "\n".join(rows)

    paths = "".join(
        f"<li><b>{html.escape(str(p.get('pathway','')))}</b> "
        f"({html.escape(str(p.get('effector','')))}) — "
        f"{html.escape(str(p.get('consequence','')))}</li>"
        for p in (exp.get("pathways") or [])
    )
    paths_block = f"<ul class='ex-list'>{paths}</ul>" if paths else \
        "<div class='ex-empty'>No pathway mapping for this variant.</div>"

    cites = "".join(
        f"<li>{html.escape(str(c.get('ref','')))} — "
        f"<i>{html.escape(str(c.get('topic','')))}</i></li>"
        for c in (exp.get("citations") or [])
    )
    unc = "".join(f"<li>{html.escape(str(u))}</li>"
                  for u in (exp.get("uncertainty") or []))

    plain = html.escape(str(exp.get("plain_language") or ""))
    disclaimer = html.escape(str(exp.get("disclaimer") or ""))

    template = """
<div class="ex-root">
  <style>
    .ex-root{font-family:'Inter',system-ui,sans-serif;background:radial-gradient(
        circle at 80% 0%,#15233a 0%,#0d1117 60%);border:1px solid #1f2937;
        border-radius:14px;padding:18px;color:#e6edf3;}
    .ex-head{display:flex;justify-content:space-between;align-items:center;
        flex-wrap:wrap;gap:10px;margin-bottom:6px;}
    .ex-title{font-size:1.02rem;font-weight:700;}
    .ex-mut{color:#00d4ff;font-family:'JetBrains Mono',monospace;}
    .ex-conf{text-align:right;}
    .ex-conf-v{font-size:1.5rem;font-weight:800;font-family:'JetBrains Mono',monospace;}
    .ex-conf-l{font-size:.62rem;color:#8b98a5;text-transform:uppercase;letter-spacing:.6px;}
    .ex-headline{font-size:.85rem;color:#cdd9e5;margin-bottom:12px;}
    .ex-plain{background:rgba(0,212,255,.06);border-left:3px solid #00d4ff;
        border-radius:7px;padding:10px 12px;font-size:.82rem;color:#cdd9e5;
        line-height:1.5;margin-bottom:14px;}
    .ex-sec{font-size:.66rem;text-transform:uppercase;letter-spacing:1px;
        color:#8b98a5;margin:14px 0 7px;}
    .ex-row{display:flex;gap:10px;align-items:flex-start;padding:8px 0;
        border-bottom:1px solid #1a2230;}
    .ex-badge{font-size:.58rem;font-weight:700;border:1px solid;border-radius:5px;
        padding:2px 6px;flex-shrink:0;letter-spacing:.5px;min-width:78px;
        text-align:center;}
    .ex-body{flex:1;}
    .ex-src{font-weight:650;font-size:.82rem;}
    .ex-cat{font-size:.66rem;color:#6b7685;margin-left:8px;}
    .ex-stmt{font-size:.78rem;color:#a9b6c2;line-height:1.45;margin-top:2px;}
    .ex-list{margin:4px 0;padding-left:18px;font-size:.78rem;color:#b6c2cf;
        line-height:1.6;}
    .ex-unc{margin:4px 0;padding-left:18px;font-size:.76rem;color:#c9a24a;
        line-height:1.55;}
    .ex-empty{font-size:.76rem;color:#6b7685;}
    .ex-foot{font-size:.65rem;color:#6b7685;margin-top:14px;border-top:1px solid
        #1f2937;padding-top:9px;}
  </style>
  <div class="ex-head">
    <div><div class="ex-title">🔎 Why this assessment?</div>
         <div class="ex-headline">Case <span class="ex-mut">__MUT__</span> · __HEADLINE__</div></div>
    <div class="ex-conf"><div class="ex-conf-v" style="color:__CCOL__">__CPCT__%</div>
         <div class="ex-conf-l">confidence</div></div>
  </div>
  <div class="ex-plain">__PLAIN__</div>
  <div class="ex-sec">📚 Evidence (strongest first)</div>
  __ROWS__
  <div class="ex-sec">🧭 Pathways perturbed</div>
  __PATHS__
  <div class="ex-sec">📖 Citations</div>
  <ul class="ex-list">__CITES__</ul>
  <div class="ex-sec">⚠️ What we don't know</div>
  <ul class="ex-unc">__UNC__</ul>
  <div class="ex-foot">__DISCLAIMER__</div>
</div>
"""
    return (template
            .replace("__MUT__", mutation)
            .replace("__HEADLINE__", headline)
            .replace("__CCOL__", ccol)
            .replace("__CPCT__", str(conf_pct))
            .replace("__PLAIN__", plain)
            .replace("__ROWS__", rows_html)
            .replace("__PATHS__", paths_block)
            .replace("__CITES__", cites)
            .replace("__UNC__", unc)
            .replace("__DISCLAIMER__", disclaimer))


# ── AMD deployment + benchmark visuals ────────────────────────────
def amd_benchmark_chart(bench: Optional[dict]) -> go.Figure:
    """Bar chart of real AMD-hardware benchmark runs (TFLOP/s and latency).
    Honest placeholder when the benchmark has not been run. Never empty."""
    bench = bench or {}
    if not bench.get("available"):
        return _empty_fig(bench.get("reason",
                          "AMD benchmark not yet run — execute "
                          "tools/benchmark_amd.py on an AMD GPU host."))
    runs = [r for r in bench.get("runs", []) if r.get("ran")]
    if not runs:
        return _empty_fig("Benchmark file present but no completed runs.")

    names, values, texts = [], [], []
    for r in runs:
        if r.get("tflops") is not None:
            names.append(f"{r.get('name','run')} ({r.get('device','?')})")
            values.append(r["tflops"])
            texts.append(f"{r['tflops']} TFLOP/s")
        elif r.get("tokens_per_s") is not None:
            names.append(f"{r.get('name','run')}")
            values.append(r["tokens_per_s"])
            texts.append(f"{r['tokens_per_s']} tok/s")
        elif r.get("seconds") is not None:
            names.append(r.get("name", "run"))
            values.append(r["seconds"])
            texts.append(f"{r['seconds']} s")
    if not names:
        return _empty_fig("Benchmark runs contained no comparable metrics.")

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", text=texts, textposition="auto",
        marker_color="#ed1c24",   # AMD red
    ))
    dev = bench.get("device", {})
    subtitle = dev.get("device_name") or ("ROCm" if dev.get("is_rocm") else "host")
    fig.update_layout(
        template="plotly_dark", height=260,
        title=f"Measured on AMD hardware · {subtitle}",
        margin=dict(l=10, r=10, t=46, b=10), xaxis_title="value",
    )
    return fig


def deployment_panel_html(tiers: Optional[dict] = None) -> str:
    """Honest 'Current ✓ / Future •' deployment map. Current = runnable today;
    future = roadmap targets, never claimed as functioning. Never-empty."""
    tiers = tiers or {}
    current = tiers.get("current") or []
    future = tiers.get("future") or []

    def _rows(items, marker, col):
        if not items:
            return "<div class='dp-empty'>—</div>"
        out = []
        for it in items:
            out.append(
                f"<div class='dp-row'><span class='dp-mark' style='color:{col}'>"
                f"{marker}</span><div><div class='dp-t'>"
                f"{html.escape(str(it.get('target','')))}</div>"
                f"<div class='dp-v'>{html.escape(str(it.get('via','')))} · "
                f"{html.escape(str(it.get('use','')))}</div></div></div>")
        return "\n".join(out)

    template = """
<div class="dp-root">
  <style>
    .dp-root{font-family:'Inter',system-ui,sans-serif;background:#0d1117;
        border:1px solid #1f2937;border-radius:14px;padding:16px;color:#e6edf3;}
    .dp-cols{display:grid;grid-template-columns:1fr 1fr;gap:18px;}
    @media(max-width:640px){.dp-cols{grid-template-columns:1fr;}}
    .dp-h{font-size:.72rem;text-transform:uppercase;letter-spacing:1px;
        margin-bottom:9px;font-weight:700;}
    .dp-row{display:flex;gap:9px;align-items:flex-start;padding:7px 0;
        border-bottom:1px solid #1a2230;}
    .dp-mark{font-weight:800;flex-shrink:0;}
    .dp-t{font-size:.84rem;font-weight:600;}
    .dp-v{font-size:.72rem;color:#8b98a5;line-height:1.4;}
    .dp-empty{color:#6b7685;font-size:.78rem;}
    .dp-note{font-size:.66rem;color:#6b7685;margin-top:12px;}
  </style>
  <div class="dp-cols">
    <div><div class="dp-h" style="color:#2ecc71">✓ Current deployment</div>__CURRENT__</div>
    <div><div class="dp-h" style="color:#00d4ff">• Future deployment (roadmap)</div>__FUTURE__</div>
  </div>
  <div class="dp-note">Current targets run today. Future targets are credible
    roadmap directions on AMD hardware — shown as aspiration, not as functioning
    features.</div>
</div>
"""
    return (template
            .replace("__CURRENT__", _rows(current, "✓", "#2ecc71"))
            .replace("__FUTURE__", _rows(future, "•", "#00d4ff")))


# ── African Oncology Command Center ───────────────────────────────
def command_center_html(snapshot: Optional[dict], height: int = 560) -> str:
    """Decision-support dashboard: continental KPI tiles + per-region analytics
    cards with dominant cancers, key mutations, drivers and access notes. Pure,
    never-empty, injection-safe."""
    snapshot = snapshot or {}
    kpis = snapshot.get("kpis") or {}
    regions = snapshot.get("regions") or []
    if not kpis and not regions:
        return ("<div style='padding:22px;color:#8b98a5;font-family:sans-serif;"
                "background:#0d1117;border-radius:12px'>Command center has no "
                "data loaded.</div>")

    kpi_defs = [
        ("regions", "Regions", "#00d4ff"),
        ("countries", "Countries", "#2ecc71"),
        ("cancers", "Cancer types", "#f1c40f"),
        ("key_mutations", "Key mutations", "#ff6b9d"),
        ("drivers", "Env. drivers", "#a29bfe"),
    ]
    tiles = "".join(
        f"<div class='cc-tile'><div class='cc-num' style='color:{col}'>"
        f"{html.escape(str(kpis.get(k, 0)))}</div>"
        f"<div class='cc-lab'>{lab}</div></div>"
        for k, lab, col in kpi_defs
    )

    cards = []
    for r in regions:
        muts = ", ".join(html.escape(str(m)) for m in (r.get("key_mutations") or [])[:6])
        cancers = ", ".join(html.escape(str(c)) for c in (r.get("cancers") or [])[:4])
        drivers = ", ".join(html.escape(str(d)) for d in (r.get("drivers") or [])[:4])
        cards.append(
            "<div class='cc-card'>"
            f"<div class='cc-region'>📍 {html.escape(str(r.get('region','')))}</div>"
            f"<div class='cc-line'><b>Cancers:</b> {cancers or '—'}</div>"
            f"<div class='cc-line'><b>Key mutations:</b> "
            f"<span class='cc-mono'>{muts or '—'}</span></div>"
            f"<div class='cc-line'><b>Drivers:</b> {drivers or '—'}</div>"
            f"<div class='cc-access'>🏥 {html.escape(str(r.get('access_note','')))}</div>"
            "</div>"
        )
    cards_html = "\n".join(cards)
    disclaimer = html.escape(str(snapshot.get("disclaimer", "")))

    template = """
<div class="cc-root">
  <style>
    .cc-root{font-family:'Inter',system-ui,sans-serif;background:radial-gradient(
        circle at 50% -10%,#16243d 0%,#0d1117 55%);border:1px solid #1f2937;
        border-radius:14px;padding:18px;color:#e6edf3;}
    .cc-title{font-size:1.05rem;font-weight:700;margin-bottom:12px;}
    .cc-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));
        gap:10px;margin-bottom:16px;}
    .cc-tile{background:rgba(22,32,58,.55);border:1px solid #25304a;
        border-radius:10px;padding:12px;text-align:center;}
    .cc-num{font-size:1.7rem;font-weight:800;font-family:'JetBrains Mono',monospace;}
    .cc-lab{font-size:.64rem;color:#8b98a5;text-transform:uppercase;
        letter-spacing:.5px;margin-top:3px;}
    .cc-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
        gap:12px;}
    .cc-card{background:rgba(13,17,23,.6);border:1px solid #25304a;
        border-radius:11px;padding:13px;}
    .cc-region{font-weight:700;font-size:.92rem;margin-bottom:7px;color:#cdd9e5;}
    .cc-line{font-size:.76rem;color:#a9b6c2;line-height:1.5;margin-bottom:3px;}
    .cc-mono{font-family:'JetBrains Mono',monospace;color:#00d4ff;}
    .cc-access{font-size:.72rem;color:#c9a24a;margin-top:7px;line-height:1.45;
        border-top:1px solid #1a2230;padding-top:6px;}
    .cc-foot{font-size:.65rem;color:#6b7685;margin-top:14px;border-top:1px solid
        #1f2937;padding-top:9px;}
  </style>
  <div class="cc-title">🌍 African Oncology Command Center</div>
  <div class="cc-kpis">__TILES__</div>
  <div class="cc-grid">__CARDS__</div>
  <div class="cc-foot">__DISCLAIMER__</div>
</div>
"""
    return (template
            .replace("__TILES__", tiles)
            .replace("__CARDS__", cards_html)
            .replace("__DISCLAIMER__", disclaimer))


# ── Offline Cancer Copilot — readiness map ────────────────────────
def offline_readiness_html(status: Optional[dict]) -> str:
    """Render the offline-capability map: each capability tagged offline (green)
    or needs-network (amber). Honest, never-empty, injection-safe."""
    status = status or {}
    caps = status.get("capabilities") or []
    if not caps:
        return ("<div style='padding:18px;color:#8b98a5;font-family:sans-serif;"
                "background:#0d1117;border-radius:12px'>No capability data.</div>")
    rows = []
    for c in caps:
        off = bool(c.get("offline"))
        col = "#2ecc71" if off else "#f1c40f"
        tag = "OFFLINE" if off else "NEEDS NET"
        rows.append(
            "<div class='of-row'>"
            f"<span class='of-tag' style='color:{col};border-color:{col}'>{tag}</span>"
            f"<div><div class='of-n'>{html.escape(str(c.get('name','')))}</div>"
            f"<div class='of-d'>{html.escape(str(c.get('detail','')))}</div></div></div>"
        )
    summary = html.escape(str(status.get("summary", "")))
    template = """
<div class="of-root">
  <style>
    .of-root{font-family:'Inter',system-ui,sans-serif;background:#0d1117;
        border:1px solid #1f2937;border-radius:14px;padding:16px;color:#e6edf3;}
    .of-h{font-size:.98rem;font-weight:700;margin-bottom:4px;}
    .of-sum{font-size:.76rem;color:#8b98a5;margin-bottom:12px;line-height:1.5;}
    .of-row{display:flex;gap:9px;align-items:flex-start;padding:7px 0;
        border-bottom:1px solid #1a2230;}
    .of-tag{font-size:.58rem;font-weight:700;border:1px solid;border-radius:5px;
        padding:2px 6px;flex-shrink:0;min-width:74px;text-align:center;}
    .of-n{font-size:.82rem;font-weight:600;}
    .of-d{font-size:.72rem;color:#8b98a5;line-height:1.4;}
  </style>
  <div class="of-h">📡 Offline Cancer Copilot — readiness</div>
  <div class="of-sum">__SUMMARY__</div>
  __ROWS__
</div>
"""
    return template.replace("__SUMMARY__", summary).replace("__ROWS__", "\n".join(rows))


# ── DNA double-helix codebase knowledge graph (signature visual) ──
def codegraph_helix_html(graph: Optional[dict], height: int = 620) -> str:
    """Self-contained WebGL render of the codebase as a DNA double helix:
    modules are glowing spheres on two spiralling strands, base-pair rungs
    connect the strands, and internal imports arc between modules. Three.js
    via CDN with a graceful offline fallback. Pure + never-empty.
    """
    graph = graph or {}
    nodes = graph.get("nodes") or []
    if not nodes:
        return ("<div style='padding:22px;color:#8b98a5;font-family:sans-serif;"
                "background:#05080f;border-radius:12px'>No code graph to render."
                "</div>")
    data_json = json.dumps({
        "nodes": nodes,
        "links": graph.get("links", []),
        "rungs": graph.get("rungs", []),
    })
    inner_h = max(int(height) - 8, 240)
    module_count = graph.get("module_count", len(nodes))
    edge_count = graph.get("edge_count", len(graph.get("links", [])))

    template = """
<div style="width:100%;background:#05080f;border-radius:12px;overflow:hidden;
     position:relative;">
  <div style="position:absolute;top:10px;left:14px;z-index:5;font-family:
       'JetBrains Mono',monospace;color:#8b98a5;font-size:.72rem;">
    🧬 codebase as DNA · <span style="color:#00d4ff">__MC__ modules</span> ·
    <span style="color:#2ecc71">__EC__ imports</span> · drag to rotate ·
    scroll to zoom · double-click to pause
  </div>
  <div id="dna-graph" style="width:100%;height:__H__px;"></div>
  <div id="dna-fallback" style="display:none;padding:18px;color:#8b98a5;
       font-family:sans-serif;font-size:.85rem;">
    The DNA codebase graph needs internet once to load the WebGL library.
  </div>
</div>
<script>
(function(){
  var DATA = __DATA__;
  function boot(){
    var THREE = window.THREE;
    var el = document.getElementById('dna-graph');
    var W = el.clientWidth || 700, H = __H__;
    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(60, W/H, 0.1, 4000);
    camera.position.set(0, 0, 180);
    var renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
    renderer.setSize(W, H); renderer.setPixelRatio(window.devicePixelRatio||1);
    el.appendChild(renderer.domElement);
    var group = new THREE.Group(); scene.add(group);

    var byId = {};
    DATA.nodes.forEach(function(n){ byId[n.id] = n; });

    // Nodes as glowing spheres.
    DATA.nodes.forEach(function(n){
      var r = 1.6 + Math.min((n.loc||0)/180, 3.2);
      var geo = new THREE.SphereGeometry(r, 16, 16);
      var col = new THREE.Color(n.color || '#9aa7b4');
      var mat = new THREE.MeshBasicMaterial({color: col});
      var m = new THREE.Mesh(geo, mat);
      m.position.set(n.x||0, n.y||0, n.z||0);
      group.add(m);
      var halo = new THREE.Mesh(new THREE.SphereGeometry(r*1.7,16,16),
        new THREE.MeshBasicMaterial({color:col, transparent:true, opacity:0.14}));
      halo.position.copy(m.position); group.add(halo);
    });

    // Zoom-responsive module-name labels (canvas-texture sprites). They stay
    // hidden while zoomed out (avoids clutter) and fade in as you zoom in.
    var labels = [];
    function makeLabel(text, color){
      var c = document.createElement('canvas'); var ctx = c.getContext('2d');
      var fs = 44; ctx.font = 'bold ' + fs + 'px monospace';
      var w = ctx.measureText(text).width + 20;
      c.width = w; c.height = fs + 20;
      ctx.font = 'bold ' + fs + 'px monospace';
      ctx.fillStyle = 'rgba(5,8,15,0.72)';
      ctx.fillRect(0,0,c.width,c.height);
      ctx.fillStyle = color || '#e6edf3';
      ctx.textBaseline = 'middle'; ctx.fillText(text, 10, c.height/2);
      var tex = new THREE.CanvasTexture(c);
      var spr = new THREE.Sprite(new THREE.SpriteMaterial(
        {map:tex, transparent:true, depthTest:false}));
      spr.scale.set(w*0.09, (fs+20)*0.09, 1);
      return spr;
    }
    DATA.nodes.forEach(function(n){
      var lbl = makeLabel(n.label || n.id, n.color || '#e6edf3');
      lbl.position.set((n.x||0), (n.y||0)+3.2, (n.z||0));
      lbl.visible = false;
      group.add(lbl); labels.push(lbl);
    });

    function strandCurve(parity, color){
      var pts = DATA.nodes.filter(function(n){return (n.strand||0)===parity;})
        .map(function(n){return new THREE.Vector3(n.x||0,n.y||0,n.z||0);});
      if(pts.length<2) return;
      var curve = new THREE.CatmullRomCurve3(pts);
      var geo = new THREE.TubeGeometry(curve, pts.length*6, 0.5, 8, false);
      group.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial(
        {color:color, transparent:true, opacity:0.55})));
    }
    strandCurve(0, 0x00d4ff); strandCurve(1, 0x2ecc71);

    // Base-pair rungs.
    DATA.rungs.forEach(function(e){
      var a=byId[e.source], b=byId[e.target]; if(!a||!b) return;
      var g=new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(a.x,a.y,a.z), new THREE.Vector3(b.x,b.y,b.z)]);
      group.add(new THREE.Line(g, new THREE.LineBasicMaterial(
        {color:0x9aa7b4, transparent:true, opacity:0.35})));
    });
    // Import edges as faint arcs.
    DATA.links.forEach(function(e){
      var a=byId[e.source], b=byId[e.target]; if(!a||!b) return;
      var g=new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(a.x,a.y,a.z), new THREE.Vector3(b.x,b.y,b.z)]);
      group.add(new THREE.Line(g, new THREE.LineBasicMaterial(
        {color:0x00d4ff, transparent:true, opacity:0.12})));
    });

    // Drag to rotate · scroll to zoom · double-click toggles auto-spin.
    var rx=0, ry=0, down=false, px=0, py=0, paused=false;
    renderer.domElement.addEventListener('mousedown',function(e){
      down=true;px=e.clientX;py=e.clientY;});
    window.addEventListener('mouseup',function(){down=false;});
    window.addEventListener('mousemove',function(e){
      if(!down)return; ry+=(e.clientX-px)*0.01; rx+=(e.clientY-py)*0.01;
      px=e.clientX; py=e.clientY;});
    // Double-click properly TOGGLES pause/resume (previously a click paused
    // forever with no way back).
    renderer.domElement.addEventListener('dblclick',function(){paused=!paused;});
    // Scroll to zoom (clamped); labels reveal as you get closer.
    renderer.domElement.addEventListener('wheel',function(e){
      e.preventDefault();
      camera.position.z = Math.max(60, Math.min(360,
        camera.position.z + (e.deltaY>0?12:-12)));
    }, {passive:false});
    function animate(){
      requestAnimationFrame(animate);
      if(!paused) ry+=0.0025;
      group.rotation.y=ry; group.rotation.x=rx;
      // Zoom-responsive labels: show the module names once zoomed in enough.
      var showLabels = camera.position.z < 150;
      for(var i=0;i<labels.length;i++){ labels[i].visible = showLabels; }
      renderer.render(scene,camera);
    }
    animate();
  }
  function fail(){document.getElementById('dna-fallback').style.display='block';}
  if(window.THREE){boot();return;}
  var s=document.createElement('script');
  s.src='https://unpkg.com/three@0.160.0/build/three.min.js';
  s.onload=boot;
  s.onerror=function(){
    var s2=document.createElement('script');
    s2.src='https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js';
    s2.onload=boot; s2.onerror=fail; document.head.appendChild(s2);
  };
  document.head.appendChild(s);
})();
</script>
"""
    return (template
            .replace("__DATA__", data_json)
            .replace("__H__", str(inner_h))
            .replace("__MC__", str(module_count))
            .replace("__EC__", str(edge_count)))


# ── Agent evaluation harness table ────────────────────────────────
def agent_eval_table(report: Optional[dict]) -> go.Figure:
    """Render per-agent evaluation metrics as a table. Never empty."""
    report = report or {}
    agents = report.get("agents") or []
    if not agents:
        return _empty_fig("No agent-evaluation results — run the harness.")
    headers = ["Agent", "Cases", "Latency (ms)", "Success", "Quality signal"]
    rows = [[], [], [], [], []]
    for a in agents:
        rows[0].append(a.get("agent", "?"))
        rows[1].append(a.get("cases", "—"))
        rows[2].append(a.get("mean_latency_ms", "—"))
        rows[3].append(f"{round(a.get('success_rate', 0) * 100)}%")
        if "calibrated" in a:
            rows[4].append("✓ calibrated" if a["calibrated"] else "✗ miscalibrated")
        elif "citation_rate" in a:
            rows[4].append(f"cite {round(a.get('citation_rate', 0) * 100)}%")
        else:
            rows[4].append("—")
    fig = go.Figure(go.Table(
        header=dict(values=headers, fill_color="#1b2435",
                    font=dict(color="#e6edf3"), align="left"),
        cells=dict(values=rows, fill_color="#0d1117",
                   font=dict(color="#cdd9e5"), align="left", height=28),
    ))
    fig.update_layout(template="plotly_dark", height=160,
                      margin=dict(l=4, r=4, t=4, b=4))
    return fig


# ── Token-efficient router savings ────────────────────────────────
def token_router_chart(report: Optional[dict]) -> go.Figure:
    """Donut of how queries were routed (cache / deterministic / LLM), with the
    avoided-call share. Never empty."""
    report = report or {}
    if not report.get("queries"):
        return _empty_fig("No routed queries yet — the router logs savings live.")
    labels = ["Cache (0 tokens)", "Deterministic (0 tokens)", "LLM (full cost)"]
    values = [report.get("cache", 0), report.get("deterministic", 0),
              report.get("llm", 0)]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=["#2ecc71", "#00d4ff", "#ff6b6b"]),
        textinfo="value",
    ))
    saved = report.get("tokens_saved", 0)
    usd = report.get("usd_saved_est", 0)
    fig.update_layout(
        template="plotly_dark", height=300,
        title=f"{report.get('pct_avoided', 0)}% of queries avoided the LLM",
        annotations=[dict(text=f"{saved:,}<br>tokens saved<br>≈ ${usd}",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=13, color="#e6edf3"))],
        margin=dict(l=10, r=10, t=46, b=10),
        legend=dict(orientation="h", y=-0.1, font=dict(size=10)),
    )
    return fig


# ── Dual guardrails verdict strip ─────────────────────────────────
def guardrails_html(verdict: Optional[dict]) -> str:
    """Compact verdict strip: the two gates (form + fact) with pass/fail, plus
    the overall gate and confidence. Pure, never-empty, injection-safe."""
    verdict = verdict or {}
    gates = verdict.get("gates") or []
    if not gates:
        return ("<div style='padding:14px;color:#8b98a5;font-family:sans-serif;"
                "background:#0d1117;border-radius:10px'>No guardrail check run."
                "</div>")
    gate = str(verdict.get("gate", "pass"))
    gate_col = {"pass": "#2ecc71", "flag": "#f1c40f", "block": "#ff6b6b"}.get(
        gate, "#8b98a5")
    conf = float(verdict.get("confidence", 0.0))
    rows = []
    for g in gates:
        sev = g.get("severity", "ok")
        col = {"ok": "#2ecc71", "warn": "#f1c40f", "fail": "#ff6b6b"}.get(sev, "#8b98a5")
        icon = {"ok": "✓", "warn": "!", "fail": "✗"}.get(sev, "•")
        name = "Form (syntactic)" if g.get("name") == "syntactic" else "Fact (ClinVar)"
        rows.append(
            f"<div class='gr-row'><span class='gr-ic' style='color:{col}'>{icon}</span>"
            f"<b>{html.escape(name)}</b>"
            f"<span class='gr-d'>{html.escape(str(g.get('detail','')))}</span></div>")
    template = """
<div class="gr-root">
  <style>
    .gr-root{font-family:'Inter',system-ui,sans-serif;background:#0d1117;
        border:1px solid #1f2937;border-radius:10px;padding:13px;color:#e6edf3;}
    .gr-head{display:flex;justify-content:space-between;align-items:center;
        margin-bottom:9px;}
    .gr-gate{font-weight:800;font-size:.9rem;letter-spacing:.5px;}
    .gr-conf{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#8b98a5;}
    .gr-row{display:flex;gap:8px;align-items:baseline;font-size:.78rem;
        padding:4px 0;border-top:1px solid #1a2230;}
    .gr-ic{font-weight:800;}
    .gr-d{color:#9aa7b4;}
  </style>
  <div class="gr-head"><span class="gr-gate" style="color:__COL__">⛉ __GATE__</span>
    <span class="gr-conf">confidence __CONF__%</span></div>
  __ROWS__
</div>
"""
    return (template
            .replace("__COL__", gate_col)
            .replace("__GATE__", gate.upper())
            .replace("__CONF__", str(round(conf * 100)))
            .replace("__ROWS__", "\n".join(rows)))


# ── Mock edge-device control panel ────────────────────────────────
def mock_device_html(demo: Optional[dict]) -> str:
    """Edge-sequencer control panel: lifecycle stages lighting up + telemetry.
    Clearly labelled a simulated device interface. Never-empty, injection-safe."""
    demo = demo or {}
    pipeline = demo.get("pipeline") or []
    final = demo.get("final") or {}
    if not pipeline:
        return ("<div style='padding:18px;color:#8b98a5;font-family:sans-serif;"
                "background:#05080f;border-radius:12px'>Mock device idle.</div>")

    steps = []
    for p in pipeline:
        reached = p.get("reached")
        active = p.get("active")
        col = "#00d4ff" if active else ("#2ecc71" if reached else "#2a3240")
        glow = "box-shadow:0 0 12px #00d4ff;" if active else ""
        steps.append(
            f"<div class='md-step'><div class='md-dot' style='background:{col};"
            f"{glow}'></div><div class='md-lab' style='color:"
            f"{'#e6edf3' if reached else '#6b7685'}'>"
            f"{html.escape(str(p.get('label','')))}</div></div>")
    steps_html = "\n".join(steps)

    focus = final.get("focus_score", 0)
    temp = final.get("temperature_c", 0)
    prog = round(float(final.get("run_progress", 0)) * 100)
    barcode = html.escape(str(final.get("barcode", "—") or "—"))

    template = """
<div class="md-root">
  <style>
    .md-root{font-family:'Inter',system-ui,sans-serif;background:radial-gradient(
        circle at 50% 0%,#0b1320 0%,#05080f 60%);border:1px solid #1f2937;
        border-radius:14px;padding:16px;color:#e6edf3;}
    .md-head{display:flex;justify-content:space-between;align-items:center;
        margin-bottom:14px;}
    .md-title{font-weight:700;font-size:.98rem;}
    .md-badge{font-size:.6rem;color:#f1c40f;border:1px solid #f1c40f;
        border-radius:5px;padding:2px 7px;letter-spacing:.5px;}
    .md-rail{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;}
    .md-step{display:flex;flex-direction:column;align-items:center;width:84px;
        text-align:center;}
    .md-dot{width:14px;height:14px;border-radius:50%;margin-bottom:5px;
        transition:all .4s;}
    .md-lab{font-size:.6rem;line-height:1.25;}
    .md-tele{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;}
    .md-cell{background:rgba(13,17,23,.6);border:1px solid #25304a;
        border-radius:9px;padding:9px;text-align:center;}
    .md-v{font-size:1.05rem;font-weight:800;font-family:'JetBrains Mono',monospace;}
    .md-k{font-size:.58rem;color:#8b98a5;text-transform:uppercase;}
    .md-foot{font-size:.64rem;color:#6b7685;margin-top:12px;}
  </style>
  <div class="md-head"><span class="md-title">🧪 Portable Sequencer — control panel</span>
    <span class="md-badge">SIMULATED DEVICE</span></div>
  <div class="md-rail">__STEPS__</div>
  <div class="md-tele">
    <div class="md-cell"><div class="md-v" style="color:#00d4ff">__BC__</div><div class="md-k">barcode</div></div>
    <div class="md-cell"><div class="md-v" style="color:#2ecc71">__FOCUS__</div><div class="md-k">focus</div></div>
    <div class="md-cell"><div class="md-v">__TEMP__°C</div><div class="md-k">flowcell temp</div></div>
    <div class="md-cell"><div class="md-v" style="color:#f1c40f">__PROG__%</div><div class="md-k">run progress</div></div>
  </div>
  <div class="md-foot">Software mock of a device API — develop-before-hardware
    pattern. No physical instrument is attached; drop in a real driver to go live.</div>
</div>
"""
    return (template
            .replace("__STEPS__", steps_html)
            .replace("__BC__", barcode)
            .replace("__FOCUS__", str(focus))
            .replace("__TEMP__", str(temp))
            .replace("__PROG__", str(prog)))


# ── Microfluidic QC decision viz ──────────────────────────────────
def microfluidic_html(run: Optional[dict]) -> str:
    """Per-frame channel quality strip with the abort/continue decision and the
    sequencing compute saved. Labelled a simulated QC workflow. Never-empty."""
    run = run or {}
    verdicts = run.get("verdicts") or []
    if not verdicts:
        return ("<div style='padding:18px;color:#8b98a5;font-family:sans-serif;"
                "background:#05080f;border-radius:12px'>No QC run to show.</div>")
    cells = []
    for v in verdicts:
        fault = v.get("fault")
        col = "#ff6b6b" if fault else "#2ecc71"
        title = html.escape(str(fault)) if fault else f"q={v.get('quality')}"
        cells.append(
            f"<div class='mf-cell' style='background:{col}'>"
            f"<span class='mf-i'>{html.escape(str(v.get('index')))}</span>"
            f"<span class='mf-t'>{title}</span></div>")
    cells_html = "\n".join(cells)
    decision = str(run.get("decision", "completed"))
    dcol = "#ff6b6b" if decision == "aborted" else "#2ecc71"
    saved = run.get("compute_saved_s", 0)
    msg = html.escape(str(run.get("message", "")))

    template = """
<div class="mf-root">
  <style>
    .mf-root{font-family:'Inter',system-ui,sans-serif;background:#05080f;
        border:1px solid #1f2937;border-radius:12px;padding:14px;color:#e6edf3;}
    .mf-head{display:flex;justify-content:space-between;align-items:center;
        margin-bottom:10px;}
    .mf-title{font-weight:700;font-size:.9rem;}
    .mf-badge{font-size:.58rem;color:#f1c40f;border:1px solid #f1c40f;
        border-radius:5px;padding:2px 7px;}
    .mf-strip{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:10px;}
    .mf-cell{width:54px;height:46px;border-radius:7px;display:flex;
        flex-direction:column;align-items:center;justify-content:center;
        color:#05080f;font-weight:700;}
    .mf-i{font-size:.66rem;}
    .mf-t{font-size:.5rem;text-align:center;line-height:1.1;padding:0 2px;}
    .mf-dec{font-size:.95rem;font-weight:800;}
    .mf-msg{font-size:.74rem;color:#a9b6c2;margin-top:5px;line-height:1.4;}
    .mf-foot{font-size:.62rem;color:#6b7685;margin-top:9px;}
  </style>
  <div class="mf-head"><span class="mf-title">💧 Microfluidic channel QC</span>
    <span class="mf-badge">SIMULATED QC</span></div>
  <div class="mf-strip">__CELLS__</div>
  <div class="mf-dec" style="color:__DCOL__">Decision: __DECISION__</div>
  <div class="mf-msg">__MSG__ · ≈ __SAVED__s of sequencing compute saved.</div>
  <div class="mf-foot">Illustrates the abort/recollect policy on simulated
    fluidics telemetry — not real imaging or diagnosis.</div>
</div>
"""
    return (template
            .replace("__CELLS__", cells_html)
            .replace("__DCOL__", dcol)
            .replace("__DECISION__", decision.upper())
            .replace("__MSG__", msg)
            .replace("__SAVED__", str(saved)))


# ── Evidence scenario explorer ("digital twin") ───────────────────
_CONF_BADGE = {
    "high": "#2ecc71", "moderate": "#00d4ff", "low": "#f1c40f",
    "investigational": "#a78bfa",
}


def scenario_explorer_html(exploration: Optional[dict]) -> str:
    """Render illustrative management scenarios as cards. Each carries its
    evidence basis, confidence and caveat — and the whole panel is clearly
    marked illustrative (not an individual prediction). Never-empty, safe."""
    exploration = exploration or {}
    scenarios = exploration.get("scenarios") or []
    if not scenarios:
        return ("<div style='padding:20px;color:#8b98a5;font-family:sans-serif;"
                "background:#0d1117;border-radius:12px'>No scenarios to explore.</div>")
    mut = html.escape(str(exploration.get("mutation", "—")))
    cancer = html.escape(str(exploration.get("cancer", "")))
    stage = html.escape(str(exploration.get("stage", "")))
    cards = []
    for s in scenarios:
        col = _CONF_BADGE.get(s.get("confidence", "moderate"), "#8b98a5")
        cards.append(
            "<div class='sc-card'>"
            f"<div class='sc-name'>{html.escape(str(s.get('name','')))}</div>"
            f"<div class='sc-int'>{html.escape(str(s.get('intervention','')))}</div>"
            f"<div class='sc-out'>{html.escape(str(s.get('illustrative_outcome','')))}</div>"
            f"<div class='sc-meta'><span class='sc-badge' style='color:{col};"
            f"border-color:{col}'>{html.escape(str(s.get('confidence','')))}</span>"
            f"<span class='sc-ev'>{html.escape(str(s.get('evidence_basis','')))}</span></div>"
            f"<div class='sc-cav'>⚠ {html.escape(str(s.get('caveat','')))}</div>"
            "</div>")
    template = """
<div class="sc-root">
  <style>
    .sc-root{font-family:'Inter',system-ui,sans-serif;background:radial-gradient(
        circle at 30% -10%,#1a1830 0%,#0d1117 55%);border:1px solid #1f2937;
        border-radius:14px;padding:18px;color:#e6edf3;}
    .sc-title{font-size:1.02rem;font-weight:700;}
    .sc-sub{font-size:.76rem;color:#8b98a5;margin-bottom:6px;}
    .sc-warn{font-size:.72rem;color:#f1c40f;background:rgba(241,196,15,.08);
        border:1px solid #4a3f1a;border-radius:7px;padding:7px 10px;margin-bottom:13px;}
    .sc-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(255px,1fr));
        gap:12px;}
    .sc-card{background:rgba(13,17,23,.6);border:1px solid #2a2546;
        border-radius:11px;padding:13px;}
    .sc-name{font-weight:700;font-size:.9rem;color:#cdd9e5;margin-bottom:3px;}
    .sc-int{font-size:.74rem;color:#a78bfa;margin-bottom:7px;}
    .sc-out{font-size:.78rem;color:#a9b6c2;line-height:1.45;margin-bottom:9px;}
    .sc-meta{display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap;}
    .sc-badge{font-size:.62rem;border:1px solid;border-radius:20px;padding:2px 8px;
        text-transform:uppercase;letter-spacing:.4px;}
    .sc-ev{font-size:.68rem;color:#6b7685;}
    .sc-cav{font-size:.7rem;color:#c9a24a;line-height:1.4;}
  </style>
  <div class="sc-title">🧪 Evidence Scenario Explorer</div>
  <div class="sc-sub">Case <span style="color:#00d4ff;font-family:monospace">__MUT__</span> · __CANCER__ · stage __STAGE__</div>
  <div class="sc-warn">⚠ Illustrative scenarios from published cohort patterns —
    <b>not a prediction or prognosis for this patient.</b> Explore options; do not act on these alone.</div>
  <div class="sc-grid">__CARDS__</div>
</div>
"""
    return (template
            .replace("__MUT__", mut)
            .replace("__CANCER__", cancer)
            .replace("__STAGE__", stage)
            .replace("__CARDS__", "\n".join(cards)))
