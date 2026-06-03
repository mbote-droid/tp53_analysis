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
