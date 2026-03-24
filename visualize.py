"""visualize.py — Experiment tracking dashboard for autoresearch-tabular.

Two-tab dashboard:
  PROGRESS — stats, metric chart, experiment log, SHAP, hypotheses
  GRAPH    — interactive pipeline graph (Columns→Transforms→Features→FeatureSets→Experiments)

Run with:
    uv run python visualize.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import dash
    from dash import Input, Output, State, dcc, html, callback_context, ALL
    import dash_cytoscape as cyto
    import plotly.graph_objects as go
except ImportError:
    print("Missing deps. Run: uv run pip install dash dash-cytoscape networkx plotly")
    sys.exit(1)

from autoresearch_tabular.config import load_config
from autoresearch_tabular.memory_graph import DEFAULT_GRAPH_PATH, load_graph
from autoresearch_tabular.inspect_graph import (
    get_diminishing_returns_signal,
    get_load_bearing_features,
    get_shap_ranking,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    _cfg = load_config()
    METRIC       = _cfg.metric.upper()
    IS_HB        = _cfg.is_higher_better
    MIN_DELTA    = _cfg.min_delta
except Exception:
    METRIC, IS_HB, MIN_DELTA = "SCORE", False, 0.001

DIRECTION = "↑ higher = better" if IS_HB else "↓ lower = better"

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG      = "#0a0a0a"
SURFACE = "#0c0c0c"
CARD    = "#111111"
BORDER  = "rgba(255,255,255,0.08)"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"
DIM     = "#3f3f46"

GREEN   = "#34d399"
RED     = "#f87171"
AMBER   = "#fbbf24"
INDIGO  = "#818cf8"
EMERALD = "#10b981"
TEAL    = "#2dd4bf"
CYAN    = "#22d3ee"

FONT = "Inter, system-ui, sans-serif"
MONO = "JetBrains Mono, Fira Code, monospace"

# Node type → color + shape mapping for cytoscape
NODE_STYLES = {
    "Column":         {"color": MUTED,   "shape": "ellipse"},
    "Transformation": {"color": TEAL,    "shape": "triangle"},
    "Feature":        {"color": GREEN,   "shape": "diamond"},
    "FeatureSet":     {"color": INDIGO,  "shape": "hexagon"},
    "Experiment":     {"color": TEAL,    "shape": "rectangle"},
    "Hypothesis":     {"color": AMBER,   "shape": "diamond"},
}

# Short labels for filter pills
NODE_TYPE_SHORT = {
    "Column": "col", "Transformation": "tra", "Feature": "fea",
    "FeatureSet": "fset", "Experiment": "exp", "Hypothesis": "hyp",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_desc(desc: str) -> str:
    if not desc:
        return "—"
    parts: dict[str, str] = {}
    for seg in desc.split(";"):
        seg = seg.strip()
        if "=" in seg:
            k, _, v = seg.partition("=")
            parts[k.strip()] = v.strip().replace("_", " ")
    col    = parts.get("col", "")
    op     = parts.get("op", "")
    reason = parts.get("reason", "")
    if col and op:
        return f"{col} → {op}"
    if reason:
        return reason[:80]
    return desc[:80]


def _short_name(n: str) -> str:
    return (n.replace("geo_target_mean4", "geo_enc_4")
             .replace("geo_target_mean3", "geo_enc_3")
             .replace("geo_target_mean2", "geo_enc_2")
             .replace("geo_target_mean",  "geo_enc_1"))[:24]


def _momentum(dr: float, n_kept: int, n_total: int) -> tuple[str, str, str]:
    if dr < 0.35:
        return "Active",    f"{n_kept}/{n_total} kept · improving steadily",    GREEN
    if dr < 0.68:
        return "Slowing",   "Recent experiments struggling to clear the bar",    AMBER
    return     "Stalled",   "Most recent attempts failed — try a new direction", RED


# ---------------------------------------------------------------------------
# Progress chart
# ---------------------------------------------------------------------------
def _progress_chart(exps: list[dict]) -> go.Figure:
    by_id = sorted(exps, key=lambda e: e.get("exp_id", 0))

    kept_pts    = [(e["exp_id"], e.get("cv_score", 0), _parse_desc(e.get("description", "")))
                   for e in by_id if e.get("kept")]
    dropped_pts = [(e["exp_id"], e.get("cv_score", 0), _parse_desc(e.get("description", "")))
                   for e in by_id if not e.get("kept")]

    ratchet_x, ratchet_y = [], []
    best = None
    for e in by_id:
        s = e.get("cv_score", 0)
        if e.get("kept"):
            if best is None or (not IS_HB and s < best) or (IS_HB and s > best):
                best = s
        if best is not None:
            ratchet_x.append(e["exp_id"])
            ratchet_y.append(best)

    fig = go.Figure()

    if dropped_pts:
        xs, ys, ds = zip(*dropped_pts)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=9, color="rgba(248,113,113,0.6)", symbol="x",
                        line=dict(width=2, color="rgba(248,113,113,0.6)")),
            name="Dropped", text=ds,
            hovertemplate="<b>Exp %{x}</b>  dropped<br>" + METRIC + ": %{y:.4f}<br>%{text}<extra></extra>",
        ))

    if ratchet_x:
        fig.add_trace(go.Scatter(
            x=ratchet_x, y=ratchet_y, mode="lines",
            line=dict(color="rgba(129,140,248,0.3)", width=1.5, dash="dash"),
            name="Best So Far", hoverinfo="skip",
        ))

    if kept_pts:
        xs, ys, ds = zip(*kept_pts)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=INDIGO, width=2),
            marker=dict(size=8, color=INDIGO, line=dict(color=BG, width=2)),
            name="Kept", text=ds,
            hovertemplate="<b>Exp %{x}</b>  kept<br>" + METRIC + ": %{y:.4f}<br>%{text}<extra></extra>",
            connectgaps=True,
        ))

    all_scores = [p[1] for p in kept_pts] + [p[1] for p in dropped_pts]
    if all_scores:
        ymin, ymax = min(all_scores), max(all_scores)
        pad = (ymax - ymin) * 0.15 if ymax > ymin else 0.001
    else:
        ymin, ymax, pad = 0, 1, 0.1

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(l=50, r=16, t=8, b=28),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=MUTED, family=MONO),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(size=10, color=MUTED, family=MONO), title=None, dtick=1),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
                   tickfont=dict(size=10, color=MUTED, family=MONO),
                   tickformat=".4f", range=[ymin - pad, ymax + pad], title=None),
        hoverlabel=dict(bgcolor="#111111", bordercolor="rgba(255,255,255,0.1)",
                        font=dict(size=11, color=TEXT, family=MONO)),
    )
    return fig


# ---------------------------------------------------------------------------
# SHAP HTML bar chart
# ---------------------------------------------------------------------------
def _shap_bar_row(name: str, value: float, std: float, max_val: float,
                  is_load_bearing: bool) -> html.Div:
    pct = (value / max_val * 100) if max_val > 0 else 0
    std_pct = (std / max_val * 100) if max_val > 0 else 0
    bar_color = AMBER if is_load_bearing else EMERALD
    display_name = _short_name(name)

    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "8px",
               "marginBottom": "6px", "height": "18px"},
        children=[
            html.Span(display_name, style={
                "width": "130px", "flexShrink": "0", "textAlign": "right",
                "fontSize": "11px", "fontFamily": MONO, "color": TEXT,
                "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap",
            }),
            html.Div(style={
                "flex": "1", "position": "relative", "height": "12px",
                "background": "rgba(255,255,255,0.04)", "borderRadius": "1px",
            }, children=[
                html.Div(style={
                    "position": "absolute", "top": "0", "left": "0",
                    "height": "100%", "width": f"{pct:.1f}%",
                    "background": bar_color, "borderRadius": "1px", "opacity": "0.85",
                }),
                html.Div(style={
                    "position": "absolute", "top": "1px",
                    "left": f"{max(0, pct - std_pct):.1f}%",
                    "width": f"{min(std_pct * 2, 100 - max(0, pct - std_pct)):.1f}%",
                    "height": "10px",
                    "borderTop": "1.5px solid rgba(255,255,255,0.45)",
                    "borderLeft": "1.5px solid rgba(255,255,255,0.45)",
                    "borderRight": "1.5px solid rgba(255,255,255,0.45)",
                }) if std > 0 else html.Div(),
            ]),
            html.Span("★" if is_load_bearing else "",
                      style={"fontSize": "12px", "color": AMBER, "width": "14px", "flexShrink": "0"}),
        ],
    )


def _shap_html(shap_data: list[dict], load_bearing: list[str]) -> list:
    if not shap_data:
        return [html.Div("No SHAP data yet.",
                         style={"color": MUTED, "fontSize": "11px", "padding": "12px"})]
    top = shap_data[:10]
    lb_set = set(load_bearing)
    mx = max(d["mean_shap"] for d in top) if top else 1
    return [_shap_bar_row(d["feature"], d["mean_shap"], d.get("shap_std", 0),
                          mx, d["feature"] in lb_set) for d in top]


# ---------------------------------------------------------------------------
# Experiment log rows
# ---------------------------------------------------------------------------
def _exp_row(e: dict, baseline_cv: float | None = None) -> html.Div:
    kept   = e.get("kept", False)
    eid    = e.get("exp_id", "?")
    cv     = e.get("cv_score", e.get("composite_score", 0))
    nf     = e.get("n_features", 0)
    desc   = _parse_desc(e.get("description", "—"))

    # Compute delta as cv_score vs baseline (first experiment)
    delta: float = 0.0
    if baseline_cv is not None:
        delta = float(cv) - float(baseline_cv)

    status_c = GREEN if kept else RED
    border_c = f"2px solid {EMERALD}" if kept else "1px solid transparent"
    bg_c     = "rgba(52,211,153,0.04)" if kept else "transparent"
    text_c   = TEXT if kept else MUTED

    delta_el = html.Span()
    if baseline_cv is not None and abs(delta) > 1e-8:
        is_improvement = (not IS_HB and delta < 0) or (IS_HB and delta > 0)
        sign = "+" if delta > 0 else "−"
        d_color = GREEN if is_improvement else RED
        delta_el = html.Span(
            f" {sign}{abs(delta):.4f}",
            style={"color": d_color, "fontSize": "10px", "fontFamily": MONO, "marginLeft": "6px"},
        )

    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "24px 36px 120px 32px 1fr",
               "columnGap": "12px", "alignItems": "center",
               "padding": "7px 14px", "borderLeft": border_c,
               "background": bg_c, "marginBottom": "1px"},
        children=[
            html.Span("✓" if kept else "✗",
                      style={"color": status_c, "fontSize": "12px", "fontWeight": "700", "textAlign": "center"}),
            html.Span(f"#{eid}", style={"color": MUTED, "fontSize": "11px", "fontFamily": MONO}),
            html.Div([
                html.Span(f"{cv:.4f}", style={"color": text_c, "fontWeight": "600" if kept else "400",
                                               "fontSize": "12px", "fontFamily": MONO}),
                delta_el,
            ], style={"display": "flex", "alignItems": "baseline"}),
            html.Span(f"{nf}f", style={"color": DIM, "fontSize": "11px", "fontFamily": MONO}),
            html.Span(desc, style={"color": text_c, "fontSize": "11px", "fontFamily": MONO,
                                   "overflow": "hidden", "textOverflow": "ellipsis",
                                   "whiteSpace": "nowrap", "display": "block"}),
        ],
    )


# ---------------------------------------------------------------------------
# Stat card
# ---------------------------------------------------------------------------
def _card(label: str, value: str, sub: str = "", value_color: str = TEXT,
          sub_color: str = MUTED) -> html.Div:
    return html.Div(
        className="card",
        children=[
            html.Div(label, className="card-label"),
            html.Div(value, className="card-value", style={"color": value_color}),
            html.Div(sub, className="card-sub", style={"color": sub_color}) if sub else html.Div(),
        ],
    )


# ---------------------------------------------------------------------------
# Graph view — build cytoscape elements from memory graph
# ---------------------------------------------------------------------------
def _build_cyto_elements(mg) -> list[dict]:
    """Convert NetworkX graph to Cytoscape elements list."""
    elements: list[dict] = []
    G = mg.graph

    # Skip transformation nodes that aren't connected to anything
    connected_trans = set()
    for u, v, d in G.edges(data=True):
        if G.nodes[u].get("node_type") == "Transformation":
            connected_trans.add(u)
        if G.nodes[v].get("node_type") == "Transformation":
            connected_trans.add(v)

    for node_id, data in G.nodes(data=True):
        ntype = data.get("node_type", "unknown")

        # Skip unconnected transformation nodes (they clutter the view)
        if ntype == "Transformation" and node_id not in connected_trans:
            continue

        label = data.get("name", node_id)

        # Build a shorter display label
        if ntype == "Column":
            label = data.get("name", node_id.replace("col_", ""))
        elif ntype == "Feature":
            label = _short_name(data.get("name", node_id.replace("feat_", "")))
        elif ntype == "Transformation":
            label = data.get("name", node_id.replace("trans_", ""))
        elif ntype == "FeatureSet":
            fset_id = data.get("fset_id", "")
            label = f"fset_{fset_id}" if fset_id else node_id
        elif ntype == "Experiment":
            eid = data.get("exp_id", "")
            label = f"exp_{eid}" if eid else node_id
        elif ntype == "Hypothesis":
            hid = data.get("hyp_id", "")
            label = f"hyp_{hid}" if hid else node_id

        # Status for styling
        status = "active"
        if ntype == "Feature":
            status = data.get("status", "active")
        elif ntype == "Experiment":
            status = "kept" if data.get("kept") else "dropped"
        elif ntype == "Hypothesis":
            if data.get("validated"):
                status = "validated"
            else:
                status = "pending"

        el = {
            "data": {
                "id": node_id,
                "label": label,
                "node_type": ntype,
                "status": status,
            },
        }

        # Attach extra info for the detail panel
        if ntype == "Column":
            el["data"]["dtype"]       = str(data.get("dtype", ""))
            el["data"]["cardinality"] = str(data.get("cardinality", ""))
        elif ntype == "Experiment":
            el["data"]["cv_score"]  = str(data.get("cv_score", ""))
            el["data"]["kept"]      = str(data.get("kept", ""))
            el["data"]["n_features"] = str(data.get("n_features", ""))
        elif ntype == "Hypothesis":
            el["data"]["text"] = str(data.get("text", ""))[:120]
            el["data"]["validated"] = str(data.get("validated", ""))

        elements.append(el)

    # Edges
    for u, v, data in G.edges(data=True):
        # Skip edges to unconnected transformation nodes
        u_type = G.nodes.get(u, {}).get("node_type", "")
        v_type = G.nodes.get(v, {}).get("node_type", "")
        if u_type == "Transformation" and u not in connected_trans:
            continue
        if v_type == "Transformation" and v not in connected_trans:
            continue

        rel = data.get("rel", "RELATED")
        edge_class = "edge-default"
        if rel in ("SUPPORTS",):
            edge_class = "edge-supports"
        elif rel in ("CONTRADICTS",):
            edge_class = "edge-contradicts"
        elif rel == "CORRELATED_WITH":
            edge_class = "edge-correlated"
        elif rel == "DERIVED_FROM":
            edge_class = "edge-lineage"
        elif rel == "MEMBER_OF":
            edge_class = "edge-membership"
        elif rel == "TESTED_IN":
            edge_class = "edge-tested"
        elif rel == "IMPROVED_OVER":
            edge_class = "edge-improved"

        elements.append({
            "data": {
                "source": u,
                "target": v,
                "rel": rel,
                "edge_class": edge_class,
            },
            "classes": edge_class,
        })

    return elements


# Cytoscape stylesheet
CYTO_STYLESHEET = [
    # ── Default node ──
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "font-size": "10px",
            "font-family": MONO,
            "color": TEXT,
            "text-valign": "bottom",
            "text-halign": "center",
            "text-margin-y": "6px",
            "background-color": MUTED,
            "border-width": 2,
            "border-color": MUTED,
            "width": 32,
            "height": 32,
            "text-wrap": "ellipsis",
            "text-max-width": "80px",
        },
    },
    # ── Node types ──
    {"selector": 'node[node_type="Column"]', "style": {
        "shape": "ellipse", "background-color": "#1a1a1a",
        "border-color": MUTED, "border-width": 2,
        "width": 36, "height": 36,
    }},
    {"selector": 'node[node_type="Transformation"]', "style": {
        "shape": "triangle", "background-color": "#1a1a1a",
        "border-color": TEAL, "border-width": 2,
        "width": 30, "height": 30,
    }},
    {"selector": 'node[node_type="Feature"]', "style": {
        "shape": "diamond", "background-color": "#1a1a1a",
        "border-color": GREEN, "border-width": 2,
        "width": 34, "height": 34,
    }},
    {"selector": 'node[node_type="Feature"][status="inactive"]', "style": {
        "border-color": RED, "opacity": 0.6,
    }},
    {"selector": 'node[node_type="FeatureSet"]', "style": {
        "shape": "hexagon", "background-color": "#1a1a1a",
        "border-color": INDIGO, "border-width": 2,
        "width": 36, "height": 36,
    }},
    {"selector": 'node[node_type="Experiment"]', "style": {
        "shape": "rectangle", "background-color": "#1a1a1a",
        "border-color": TEAL, "border-width": 2,
        "width": 36, "height": 36,
    }},
    {"selector": 'node[node_type="Experiment"][status="kept"]', "style": {
        "border-color": GREEN, "background-color": "rgba(52,211,153,0.1)",
    }},
    {"selector": 'node[node_type="Experiment"][status="dropped"]', "style": {
        "border-color": RED, "background-color": "rgba(248,113,113,0.1)",
    }},
    {"selector": 'node[node_type="Hypothesis"]', "style": {
        "shape": "diamond", "background-color": AMBER,
        "border-color": AMBER, "border-width": 2,
        "width": 28, "height": 28, "opacity": 0.85,
    }},

    # ── Selected node ──
    {"selector": "node:selected", "style": {
        "border-width": 3,
        "border-color": "#ffffff",
        "overlay-opacity": 0,
    }},

    # ── Default edge ──
    {"selector": "edge", "style": {
        "width": 1.5,
        "line-color": "rgba(100,116,139,0.3)",
        "target-arrow-color": "rgba(100,116,139,0.3)",
        "target-arrow-shape": "triangle",
        "arrow-scale": 0.7,
        "curve-style": "bezier",
    }},
    # Edge types
    {"selector": ".edge-lineage", "style": {
        "line-color": TEAL, "target-arrow-color": TEAL, "opacity": 0.5,
    }},
    {"selector": ".edge-membership", "style": {
        "line-color": INDIGO, "target-arrow-color": INDIGO, "opacity": 0.4,
    }},
    {"selector": ".edge-tested", "style": {
        "line-color": TEAL, "target-arrow-color": TEAL, "opacity": 0.5,
    }},
    {"selector": ".edge-improved", "style": {
        "line-color": GREEN, "target-arrow-color": GREEN,
        "line-style": "dashed", "opacity": 0.5,
    }},
    {"selector": ".edge-supports", "style": {
        "line-color": GREEN, "target-arrow-color": GREEN,
        "line-style": "dashed", "opacity": 0.6,
    }},
    {"selector": ".edge-contradicts", "style": {
        "line-color": RED, "target-arrow-color": RED,
        "line-style": "dashed", "opacity": 0.6,
    }},
    {"selector": ".edge-correlated", "style": {
        "line-color": AMBER, "target-arrow-color": AMBER,
        "line-style": "dotted", "opacity": 0.4,
        "target-arrow-shape": "none",
    }},
]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, title="Experiment Tracker",
                suppress_callback_exceptions=True)

app.index_string = f"""<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
        html, body {{ height: 100%; background: {BG}; color: {TEXT}; font-family: {FONT}; overflow: hidden; }}
        ::-webkit-scrollbar {{ width: 3px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: #333; border-radius: 2px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #555; }}

        .card {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 4px;
            padding: 14px 16px;
            margin-bottom: 8px;
        }}
        .card-label {{
            font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em;
            color: {MUTED}; font-weight: 500; margin-bottom: 6px;
            font-family: {MONO};
        }}
        .card-value {{
            font-size: 28px; font-weight: 600; color: {TEXT};
            line-height: 1.1; letter-spacing: -0.02em;
            font-family: {MONO};
        }}
        .card-sub {{
            font-size: 10px; color: {MUTED}; margin-top: 4px;
            font-family: {MONO};
        }}

        .sec-label {{
            font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em;
            color: {MUTED}; font-weight: 500; padding-bottom: 8px;
            border-bottom: 1px solid {BORDER}; margin-bottom: 10px;
            font-family: {MONO};
            display: flex; align-items: center; gap: 8px;
        }}

        .mom-row {{ display: flex; align-items: center; gap: 8px; margin-top: 4px; }}
        .mom-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
        .mom-label {{ font-size: 18px; font-weight: 600; }}

        .header {{
            height: 48px; background: {BG};
            border-bottom: 1px solid {BORDER};
            display: flex; align-items: center;
            padding: 0 20px; gap: 12px; flex-shrink: 0;
        }}
        .header-title {{
            font-size: 14px; font-weight: 600; color: {TEXT};
            letter-spacing: -0.01em; font-family: {MONO};
        }}
        .header-sub {{
            font-size: 11px; color: {MUTED}; font-family: {MONO};
        }}
        .header-spacer {{ flex: 1; }}
        .live-pill {{
            display: flex; align-items: center; gap: 6px;
            background: rgba(52,211,153,0.06);
            border: 1px solid rgba(52,211,153,0.2);
            border-radius: 3px; padding: 3px 10px;
        }}
        .live-dot {{
            width: 6px; height: 6px; border-radius: 50%;
            background: {GREEN};
            animation: pulse 2.2s ease-in-out infinite;
        }}
        .live-text {{
            font-size: 10px; letter-spacing: 0.14em; color: {GREEN}; font-weight: 600;
            font-family: {MONO};
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.3; }}
        }}

        .log-header {{
            display: grid;
            grid-template-columns: 24px 36px 120px 32px 1fr;
            column-gap: 12px;
            padding: 4px 14px 8px;
            font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
            color: {DIM}; border-bottom: 1px solid {BORDER}; margin-bottom: 4px;
            font-family: {MONO};
        }}

        .shap-legend {{
            display: flex; align-items: center; gap: 6px;
            font-size: 10px; color: {MUTED}; margin-bottom: 12px;
            font-family: {MONO};
        }}
        .shap-legend-dot {{ width: 8px; height: 8px; border-radius: 2px; }}

        .js-plotly-plot .plotly .modebar {{ display: none !important; }}

        /* ── Tabs ── */
        .tab-bar {{
            display: flex; align-items: center; gap: 0;
            border-bottom: 1px solid {BORDER};
            padding: 0 20px; flex-shrink: 0;
            background: {BG};
        }}
        .tab-btn {{
            padding: 10px 18px;
            font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
            text-transform: uppercase; font-family: {MONO};
            color: {MUTED}; cursor: pointer; border: none; background: none;
            border-bottom: 2px solid transparent;
            transition: color 0.15s, border-color 0.15s;
            display: flex; align-items: center; gap: 6px;
        }}
        .tab-btn:hover {{ color: {TEXT}; }}
        .tab-btn.active {{
            color: {GREEN}; border-bottom-color: {GREEN};
        }}

        /* ── Graph toolbar ── */
        .graph-toolbar {{
            display: flex; align-items: center; gap: 10px;
            padding: 10px 16px; flex-shrink: 0;
            border-bottom: 1px solid {BORDER};
        }}
        .graph-search {{
            background: {CARD}; border: 1px solid {BORDER};
            border-radius: 3px; padding: 5px 10px 5px 30px;
            font-size: 11px; color: {TEXT}; font-family: {MONO};
            outline: none; width: 180px;
        }}
        .graph-search::placeholder {{ color: {DIM}; }}
        .graph-search:focus {{ border-color: rgba(255,255,255,0.15); }}
        .search-wrap {{
            position: relative; display: flex; align-items: center;
        }}
        .search-icon {{
            position: absolute; left: 9px;
            font-size: 12px; color: {DIM}; pointer-events: none;
        }}

        .filter-pill {{
            padding: 3px 10px; font-size: 10px; font-weight: 500;
            font-family: {MONO}; letter-spacing: 0.06em;
            border: 1px solid {BORDER}; border-radius: 3px;
            background: transparent; color: {MUTED};
            cursor: pointer; transition: all 0.15s;
        }}
        .filter-pill:hover {{ border-color: rgba(255,255,255,0.2); color: {TEXT}; }}
        .filter-pill.active {{
            background: rgba(52,211,153,0.08); border-color: rgba(52,211,153,0.3);
            color: {GREEN};
        }}
        .toolbar-spacer {{ flex: 1; }}

        .layout-toggle {{
            display: flex; border: 1px solid {BORDER}; border-radius: 3px;
            overflow: hidden;
        }}
        .layout-btn {{
            padding: 4px 12px; font-size: 10px; font-weight: 500;
            font-family: {MONO}; color: {MUTED};
            background: transparent; border: none; cursor: pointer;
            transition: all 0.15s;
        }}
        .layout-btn:first-child {{ border-right: 1px solid {BORDER}; }}
        .layout-btn.active {{
            background: rgba(255,255,255,0.06); color: {TEXT};
        }}

        /* ── Node detail panel ── */
        .detail-panel {{
            padding: 16px 14px;
        }}
        .detail-title {{
            font-size: 16px; font-weight: 600; font-family: {MONO};
            color: {TEXT}; margin-bottom: 16px;
        }}
        .detail-section {{
            font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
            color: {DIM}; font-family: {MONO}; margin-bottom: 6px;
        }}
        .detail-value {{
            font-size: 12px; font-family: {MONO}; color: {TEXT};
            margin-bottom: 14px; display: flex; align-items: center; gap: 6px;
        }}
        .detail-badge {{
            display: inline-flex; align-items: center; gap: 4px;
            padding: 2px 8px; border-radius: 3px;
            font-size: 10px; font-family: {MONO}; font-weight: 500;
        }}
    </style>
</head>
<body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _progress_tab_content():
    """The progress view (chart + log + SHAP)."""
    return html.Div(
        style={"display": "flex", "flex": "1", "overflow": "hidden"},
        children=[
            # Center — chart + log
            html.Div(
                style={"flex": "1", "display": "flex", "flexDirection": "column",
                       "overflow": "hidden", "padding": "16px 20px 0 20px"},
                children=[
                    html.Div(className="sec-label", children=[
                        html.Span("✦", style={"fontSize": "11px"}),
                        html.Span("EXPERIMENT PROGRESS"),
                    ]),
                    dcc.Graph(id="progress-chart", config={"displayModeBar": False},
                              style={"flexShrink": "0", "margin": "0 -8px 12px"}),
                    html.Div(className="log-header", children=[
                        html.Span("STS"), html.Span("ID"),
                        html.Span(f"{METRIC} (Δ)"), html.Span("FEATS"),
                        html.Span("WHAT CHANGED"),
                    ]),
                    html.Div(id="exp-log", style={"overflowY": "auto", "flex": "1", "paddingBottom": "16px"}),
                ],
            ),
            # Right sidebar — SHAP + Hypotheses
            html.Div(
                style={"width": "320px", "flexShrink": "0", "background": BG,
                       "borderLeft": f"1px solid {BORDER}", "overflowY": "auto", "padding": "16px 14px"},
                children=[
                    html.Div(className="sec-label", children=[
                        html.Span("◎", style={"fontSize": "11px"}),
                        html.Span("FEATURE IMPORTANCE"),
                    ]),
                    html.Div(className="shap-legend", children=[
                        html.Div(className="shap-legend-dot", style={"background": AMBER}),
                        html.Span("Always Kept", style={"marginRight": "10px"}),
                        html.Div(className="shap-legend-dot", style={"background": EMERALD}),
                        html.Span("Active"),
                    ]),
                    html.Div(id="shap-panel"),
                    html.Div(style={"height": "20px"}),
                    html.Div(id="hypotheses-panel"),
                ],
            ),
        ],
    )


def _graph_tab_content():
    """The graph view (cytoscape + toolbar + node details)."""
    return html.Div(
        style={"display": "flex", "flex": "1", "overflow": "hidden"},
        children=[
            # Center — toolbar + graph
            html.Div(
                style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                children=[
                    # Toolbar
                    html.Div(className="graph-toolbar", children=[
                        # Search
                        html.Div(className="search-wrap", children=[
                            html.Span("⌕", className="search-icon"),
                            dcc.Input(id="graph-search", type="text",
                                      placeholder="Search nodes...",
                                      className="graph-search", debounce=True),
                        ]),
                        # Filter pills
                        *[html.Button(
                            short, id={"type": "filter-pill", "index": ntype},
                            className="filter-pill active",
                            n_clicks=0,
                        ) for ntype, short in NODE_TYPE_SHORT.items()],
                        html.Div(className="toolbar-spacer"),
                        # Layout toggle
                        html.Div(className="layout-toggle", children=[
                            html.Button("Hierarchical", id="layout-hierarchical",
                                        className="layout-btn active", n_clicks=0),
                            html.Button("Force", id="layout-force",
                                        className="layout-btn", n_clicks=0),
                        ]),
                    ]),
                    # Cytoscape graph
                    cyto.Cytoscape(
                        id="pipeline-graph",
                        layout={
                            "name": "breadthfirst",
                            "directed": True,
                            "spacingFactor": 1.4,
                            "animate": False,
                            "padding": 40,
                        },
                        stylesheet=CYTO_STYLESHEET,
                        elements=[],
                        style={"flex": "1", "background": BG},
                        responsive=True,
                        userZoomingEnabled=True,
                        userPanningEnabled=True,
                        minZoom=0.3,
                        maxZoom=3.0,
                    ),
                ],
            ),
            # Right panel — node details
            html.Div(
                style={"width": "320px", "flexShrink": "0", "background": BG,
                       "borderLeft": f"1px solid {BORDER}", "overflowY": "auto"},
                children=[html.Div(id="node-detail-panel")],
            ),
        ],
    )


app.layout = html.Div(
    style={"display": "flex", "flexDirection": "column", "height": "100vh",
           "background": BG, "overflow": "hidden"},
    children=[
        dcc.Interval(id="refresh", interval=3000, n_intervals=0),
        dcc.Store(id="active-tab", data="progress"),
        dcc.Store(id="active-filters", data=list(NODE_TYPE_SHORT.keys())),
        dcc.Store(id="graph-layout-name", data="breadthfirst"),

        # Header
        html.Div(className="header", children=[
            html.Span("⊙", style={"color": MUTED, "fontSize": "14px"}),
            html.Span("autoresearch-tabular", className="header-title"),
            html.Span("/", style={"color": DIM, "fontSize": "12px"}),
            html.Span(f"{METRIC} {DIRECTION}", className="header-sub"),
            html.Div(className="header-spacer"),
            html.Div(className="live-pill", children=[
                html.Div(className="live-dot"),
                html.Span("LIVE", className="live-text"),
            ]),
        ]),

        # Tabs
        html.Div(className="tab-bar", children=[
            html.Button(["↗ ", "PROGRESS"], id="tab-progress",
                        className="tab-btn active", n_clicks=0),
            html.Button(["⎔ ", "GRAPH"], id="tab-graph",
                        className="tab-btn", n_clicks=0),
        ]),

        # Body — both tabs always in DOM, toggled via display
        html.Div(
            style={"display": "flex", "flex": "1", "overflow": "hidden"},
            children=[
                # Left sidebar
                html.Div(
                    style={"width": "240px", "flexShrink": "0", "background": BG,
                           "borderRight": f"1px solid {BORDER}",
                           "overflowY": "auto", "padding": "16px 14px"},
                    children=[html.Div(id="stats-panel")],
                ),
                # Progress tab content (visible by default)
                html.Div(id="progress-tab-wrap",
                         style={"flex": "1", "display": "flex", "flexDirection": "column",
                                "overflow": "hidden"},
                         children=[_progress_tab_content()]),
                # Graph tab content (hidden by default)
                html.Div(id="graph-tab-wrap",
                         style={"flex": "1", "display": "none", "flexDirection": "column",
                                "overflow": "hidden"},
                         children=[_graph_tab_content()]),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Tab switching
# ---------------------------------------------------------------------------
@app.callback(
    Output("active-tab",         "data"),
    Output("tab-progress",       "className"),
    Output("tab-graph",          "className"),
    Output("progress-tab-wrap",  "style"),
    Output("graph-tab-wrap",     "style"),
    Input("tab-progress",        "n_clicks"),
    Input("tab-graph",           "n_clicks"),
    prevent_initial_call=True,
)
def switch_tab(n_prog, n_graph):
    ctx = callback_context
    if not ctx.triggered:
        return ("progress", "tab-btn active", "tab-btn",
                {"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                {"flex": "1", "display": "none", "flexDirection": "column", "overflow": "hidden"})
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "tab-graph":
        return ("graph", "tab-btn", "tab-btn active",
                {"flex": "1", "display": "none", "flexDirection": "column", "overflow": "hidden"},
                {"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"})
    return ("progress", "tab-btn active", "tab-btn",
            {"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
            {"flex": "1", "display": "none", "flexDirection": "column", "overflow": "hidden"})


# ---------------------------------------------------------------------------
# Graph layout toggle
# ---------------------------------------------------------------------------
@app.callback(
    Output("graph-layout-name",   "data"),
    Output("layout-hierarchical", "className"),
    Output("layout-force",        "className"),
    Input("layout-hierarchical",  "n_clicks"),
    Input("layout-force",         "n_clicks"),
    prevent_initial_call=True,
)
def toggle_layout(n_hier, n_force):
    ctx = callback_context
    if not ctx.triggered:
        return "breadthfirst", "layout-btn active", "layout-btn"
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "layout-force":
        return "cose", "layout-btn", "layout-btn active"
    return "breadthfirst", "layout-btn active", "layout-btn"


@app.callback(
    Output("pipeline-graph", "layout"),
    Input("graph-layout-name", "data"),
)
def update_graph_layout(layout_name):
    if layout_name == "cose":
        return {
            "name": "cose",
            "animate": False,
            "padding": 40,
            "nodeRepulsion": 8000,
            "idealEdgeLength": 80,
            "edgeElasticity": 40,
            "gravity": 0.3,
        }
    return {
        "name": "breadthfirst",
        "directed": True,
        "spacingFactor": 1.4,
        "animate": False,
        "padding": 40,
    }


# ---------------------------------------------------------------------------
# Graph filter pills (toggle individual type filters)
# ---------------------------------------------------------------------------
@app.callback(
    Output("active-filters", "data"),
    Input({"type": "filter-pill", "index": ALL}, "n_clicks"),
    State("active-filters", "data"),
    prevent_initial_call=True,
)
def toggle_filter(n_clicks_list, active_filters):
    ctx = callback_context
    if not ctx.triggered:
        return active_filters

    triggered_id = ctx.triggered[0]["prop_id"]
    # Extract the node type from the pattern-matching ID
    try:
        idx_str = triggered_id.split(".")[0]
        idx_dict = json.loads(idx_str)
        ntype = idx_dict["index"]
    except Exception:
        return active_filters

    if ntype in active_filters:
        active_filters.remove(ntype)
    else:
        active_filters.append(ntype)
    return active_filters


# ---------------------------------------------------------------------------
# Graph elements update
# ---------------------------------------------------------------------------
_last_elements_hash: str | None = None
_last_elements: list[dict] = []


@app.callback(
    Output("pipeline-graph", "elements"),
    Input("refresh",         "n_intervals"),
    Input("active-filters",  "data"),
    Input("graph-search",    "value"),
)
def update_graph_elements(_n, active_filters, search_query):
    global _last_elements_hash, _last_elements

    if not DEFAULT_GRAPH_PATH.exists():
        return []

    mg = load_graph(DEFAULT_GRAPH_PATH)
    all_elements = _build_cyto_elements(mg)

    # Filter by active node types
    filtered = []
    visible_nodes = set()
    for el in all_elements:
        if "source" in el.get("data", {}):
            continue  # process edges after
        ntype = el["data"].get("node_type", "")
        if ntype in active_filters:
            # Search filter
            if search_query:
                label = el["data"].get("label", "").lower()
                nid = el["data"].get("id", "").lower()
                if search_query.lower() not in label and search_query.lower() not in nid:
                    continue
            visible_nodes.add(el["data"]["id"])
            filtered.append(el)

    # Add edges where both endpoints are visible
    for el in all_elements:
        if "source" in el.get("data", {}):
            if el["data"]["source"] in visible_nodes and el["data"]["target"] in visible_nodes:
                filtered.append(el)

    # Only push new elements if something actually changed,
    # otherwise dash.no_update prevents cose from re-simulating.
    h = hashlib.md5(json.dumps(filtered, sort_keys=True).encode()).hexdigest()
    if h == _last_elements_hash:
        return dash.no_update
    _last_elements_hash = h
    _last_elements = filtered
    return filtered


# ---------------------------------------------------------------------------
# Update filter pill styles
# ---------------------------------------------------------------------------
@app.callback(
    Output({"type": "filter-pill", "index": ALL}, "className"),
    Input("active-filters", "data"),
)
def update_pill_styles(active_filters):
    return [
        "filter-pill active" if ntype in active_filters else "filter-pill"
        for ntype in NODE_TYPE_SHORT.keys()
    ]


# ---------------------------------------------------------------------------
# Node detail panel
# ---------------------------------------------------------------------------
@app.callback(
    Output("node-detail-panel", "children"),
    Input("pipeline-graph",     "tapNodeData"),
)
def show_node_detail(node_data):
    if not node_data:
        return html.Div(
            className="detail-panel",
            children=[
                html.Div("NODE DETAILS", className="sec-label",
                         style={"marginTop": "0"}),
                html.Div("Click a node to view details.",
                         style={"color": DIM, "fontSize": "11px", "fontFamily": MONO,
                                "marginTop": "12px"}),
            ],
        )

    ntype  = node_data.get("node_type", "unknown")
    label  = node_data.get("label", node_data.get("id", ""))
    status = node_data.get("status", "")
    nid    = node_data.get("id", "")

    style_info = NODE_STYLES.get(ntype, {"color": MUTED})
    type_color = style_info["color"]

    # Status color
    status_color = GREEN
    if status in ("dropped", "inactive"):
        status_color = RED
    elif status == "pending":
        status_color = AMBER

    detail_rows: list = [
        html.Div("NODE DETAILS", className="sec-label", style={"marginTop": "0"}),
        html.Div(label, className="detail-title"),

        html.Div("TYPE", className="detail-section"),
        html.Div(className="detail-value", children=[
            html.Span(className="detail-badge",
                      style={"background": f"rgba({_hex_to_rgb(type_color)},0.1)",
                             "color": type_color,
                             "border": f"1px solid {type_color}"},
                      children=[
                          html.Span("⊙", style={"fontSize": "10px"}),
                          html.Span(ntype.lower()),
                      ]),
        ]),

        html.Div("STATUS", className="detail-section"),
        html.Div(className="detail-value", children=[
            html.Span("●", style={"color": status_color, "fontSize": "8px"}),
            html.Span(status, style={"color": status_color}),
        ]),

        html.Div("ID", className="detail-section"),
        html.Div(nid, className="detail-value", style={"color": MUTED}),
    ]

    # Type-specific details
    if ntype == "Column":
        dtype = node_data.get("dtype", "")
        card = node_data.get("cardinality", "")
        if dtype:
            detail_rows.append(html.Div("DTYPE", className="detail-section"))
            detail_rows.append(html.Div(dtype, className="detail-value"))
        if card:
            detail_rows.append(html.Div("CARDINALITY", className="detail-section"))
            detail_rows.append(html.Div(card, className="detail-value"))

    elif ntype == "Experiment":
        cv = node_data.get("cv_score", "")
        kept = node_data.get("kept", "")
        nf = node_data.get("n_features", "")
        if cv:
            detail_rows.append(html.Div(f"{METRIC}", className="detail-section"))
            detail_rows.append(html.Div(cv, className="detail-value"))
        if nf:
            detail_rows.append(html.Div("FEATURES", className="detail-section"))
            detail_rows.append(html.Div(nf, className="detail-value"))

    elif ntype == "Hypothesis":
        text = node_data.get("text", "")
        if text:
            detail_rows.append(html.Div("TEXT", className="detail-section"))
            detail_rows.append(html.Div(text, style={
                "fontSize": "11px", "fontFamily": MONO, "color": MUTED,
                "lineHeight": "1.5", "wordBreak": "break-word", "marginBottom": "14px",
            }))

    return html.Div(className="detail-panel", children=detail_rows)


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R,G,B' for rgba()."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "255,255,255"
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


# ---------------------------------------------------------------------------
# Progress tab callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("stats-panel",      "children"),
    Output("progress-chart",   "figure"),
    Output("exp-log",          "children"),
    Output("shap-panel",       "children"),
    Output("hypotheses-panel", "children"),
    Input("refresh",           "n_intervals"),
)
def refresh_progress(_n: int) -> tuple:
    empty_fig = go.Figure()
    empty_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", height=100)
    empty = html.Div()

    if not DEFAULT_GRAPH_PATH.exists():
        msg = html.Div("No data yet. Run: uv run autoresearch prepare",
                       style={"color": MUTED, "fontSize": "12px"})
        return msg, empty_fig, msg, [], empty

    mg   = load_graph(DEFAULT_GRAPH_PATH)
    exps = sorted(mg.get_experiment_history(n=1000), key=lambda e: e.get("exp_id", 0))
    best = mg.get_best_experiment(is_higher_better=IS_HB)
    dr   = get_diminishing_returns_signal(mg.graph, min_delta=MIN_DELTA, is_higher_better=IS_HB)

    try:
        shap_data = get_shap_ranking(mg.graph, is_higher_better=IS_HB)
        load_bear = get_load_bearing_features(mg.graph)
    except Exception:
        shap_data, load_bear = [], []

    try:
        hypotheses = mg.get_active_hypotheses()
    except Exception:
        hypotheses = []

    n_total = len(exps)
    n_kept  = sum(1 for e in exps if e.get("kept"))
    keep_pct = f"{n_kept/n_total:.0%}" if n_total else "—"

    best_cv = f"{best.get('cv_score', 0):.4f}" if best else "—"
    best_nf = best.get("n_features", 0) if best else 0

    # Compute delta as best cv_score vs baseline (first experiment's cv_score)
    baseline_cv_val = exps[0].get("cv_score", 0) if exps else 0
    if best and len(exps) >= 2:
        d = best.get("cv_score", 0) - baseline_cv_val
        if abs(d) > 1e-8:
            improvement = (d < 0) if not IS_HB else (d > 0)
            sign = "+" if d > 0 else "−"
            delta_str = f"{sign}{abs(d):.4f} vs baseline"
            delta_col = GREEN if improvement else RED
        else:
            delta_str, delta_col = "", MUTED
    else:
        delta_str, delta_col = "", MUTED

    mom_label, mom_desc, mom_col = _momentum(dr, n_kept, n_total)

    stats = html.Div([
        _card("⊞  EXPERIMENTS", str(n_total),
              f"{n_kept} kept · {keep_pct} success rate"),
        _card(f"↗  BEST {METRIC}", best_cv, delta_str,
              value_color=INDIGO, sub_color=delta_col if delta_str else MUTED),
        _card("#  FEATURES IN BEST", str(best_nf),
              f"+0.001 {METRIC} penalty each", value_color=GREEN),
        html.Div(className="card", children=[
            html.Div("✦  MOMENTUM", className="card-label"),
            html.Div(className="mom-row", children=[
                html.Div(className="mom-dot", style={"background": mom_col, "boxShadow": f"0 0 6px {mom_col}"}),
                html.Span(mom_label, className="mom-label", style={"color": mom_col}),
            ]),
            html.Div(mom_desc, className="card-sub"),
        ]),
    ])

    # Pass baseline cv_score (first experiment) to each row for delta calculation
    baseline_cv_for_rows = exps[0].get("cv_score", 0) if exps else None
    log_rows = [_exp_row(e, baseline_cv=baseline_cv_for_rows) for e in reversed(exps)]
    if not log_rows:
        log_rows = [html.Div("No experiments yet.",
                             style={"color": MUTED, "fontSize": "12px", "padding": "12px"})]

    shap_rows = _shap_html(shap_data, load_bear)

    hyp_els: list = [html.Div(className="sec-label", children=[
        html.Span("✦", style={"fontSize": "11px"}), html.Span("HYPOTHESES"),
    ], style={"marginTop": "0"})]
    if hypotheses:
        for h in hypotheses[:8]:
            et = h.get("edge_type")
            icon = "✓" if et == "SUPPORTS" else ("✗" if et == "CONTRADICTS" else "·")
            color = GREEN if et == "SUPPORTS" else (RED if et == "CONTRADICTS" else MUTED)
            raw_text = h.get("text", "")
            display_text = _parse_desc(raw_text) if raw_text.startswith("col=") else raw_text
            hyp_els.append(html.Div(
                style={"display": "flex", "gap": "8px", "alignItems": "flex-start", "marginBottom": "10px"},
                children=[
                    html.Span(icon, style={"color": color, "fontSize": "12px", "flexShrink": "0",
                                           "marginTop": "1px", "fontWeight": "700"}),
                    html.Span(display_text[:120], style={
                        "color": MUTED, "fontSize": "11px", "lineHeight": "1.5", "wordBreak": "break-word",
                    }),
                ],
            ))
    else:
        hyp_els.append(html.Div("No hypotheses recorded yet.", style={"color": DIM, "fontSize": "11px"}))

    return stats, _progress_chart(exps), log_rows, shap_rows, html.Div(hyp_els)


if __name__ == "__main__":
    print(f"  Dashboard  →  http://localhost:8050")
    print(f"  Metric: {METRIC}  ({DIRECTION})")
    print(f"  Live auto-refresh every 3s.")
    app.run(debug=False, port=8050)