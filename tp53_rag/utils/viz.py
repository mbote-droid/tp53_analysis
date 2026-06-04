"""
TP53 RAG Platform — Visualization helpers.

Pure, Streamlit-free functions so they can be unit-tested without a running
app. app.py imports these; tests import these. Everything returns a non-empty
result (honours the platform's zero-empty-output rule) and degrades
gracefully on bad input.
"""
from __future__ import annotations

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
