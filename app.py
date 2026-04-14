

import streamlit as st
import random
import numpy as np
import plotly.graph_objects as go
import time

from routing_engine import (
    build_network, Graph, EdgeAttr, GAConfig, GAResult, PathMetrics,
    GenSnapshot, dijkstra_cost, dijkstra_latency, dijkstra_reliability,
    run_ga, get_all_edges, get_top_routes,
    NODE_POSITIONS, NODE_LABELS, LINK_COLORS, CHALLENGES,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="GA vs Dijkstra — Network Routing",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
    /* Dark premium theme */
    .stApp { background-color: #0a0e17; }
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    .metric-card {
        padding: 18px 22px;
        border-radius: 12px;
        border: 1.5px solid #1e293b;
        background: linear-gradient(135deg, #0f172a 0%, #1a1f2e 100%);
        margin-bottom: 12px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    }
    .metric-card h4 { margin: 0 0 8px 0; color: #94a3b8; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 1.8em; font-weight: 700; margin: 4px 0; }
    .metric-card .detail { color: #64748b; font-size: 0.85em; }

    /* Status badges */
    .badge-win { background: linear-gradient(135deg, #059669, #10b981); color: white; padding: 6px 16px; border-radius: 20px; font-weight: 700; display: inline-block; }
    .badge-fail { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; padding: 6px 16px; border-radius: 20px; font-weight: 700; display: inline-block; }
    .badge-tie { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; padding: 6px 16px; border-radius: 20px; font-weight: 700; display: inline-block; }

    /* Winner banner */
    .winner-banner {
        padding: 20px 28px;
        border-radius: 14px;
        text-align: center;
        margin: 16px 0;
        font-size: 1.1em;
    }
    .winner-ga {
        background: linear-gradient(135deg, rgba(0,198,255,0.12), rgba(124,58,237,0.12));
        border: 2px solid #00C6FF;
    }
    .winner-tie {
        background: linear-gradient(135deg, rgba(249,115,22,0.12), rgba(217,119,6,0.12));
        border: 2px solid #F97316;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4em;
        font-weight: 700;
        color: #e2e8f0;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #1e293b;
    }

    /* Route path display */
    .route-path {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        background: #1e293b;
        padding: 8px 14px;
        border-radius: 8px;
        color: #e2e8f0;
        font-size: 0.9em;
        display: inline-block;
        margin: 4px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
    }

    /* Link type dots */
    .link-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VISUALIZATION HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_network_figure(graph, paths=None, path_labels=None, path_colors=None,
                          highlight_nodes=None, title="", height=520):
    """Create an interactive Plotly network visualization."""
    fig = go.Figure()

    # Draw all edges (thin, transparent)
    seen_edges = set()
    for u in sorted(graph.nodes):
        for v, attr in graph.neighbors(u):
            key = tuple(sorted([u, v]))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            x0, y0 = NODE_POSITIONS[u]
            x1, y1 = NODE_POSITIONS[v]
            color = LINK_COLORS.get(attr.link_type, "#444")
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=1.5, color=color),
                opacity=0.25,
                hoverinfo='text',
                text=[f"{u} ↔ {v}<br>Type: {attr.link_type}<br>"
                      f"Cost: {attr.cost} | Lat: {attr.latency}ms | Rel: {attr.reliability:.3f}"] * 3,
                showlegend=False,
            ))

    # Draw highlighted paths
    if paths:
        for i, path in enumerate(paths):
            if not path:
                continue
            label = path_labels[i] if path_labels else f"Path {i + 1}"
            color = path_colors[i] if path_colors else "#00C6FF"
            for j in range(len(path) - 1):
                x0, y0 = NODE_POSITIONS[path[j]]
                x1, y1 = NODE_POSITIONS[path[j + 1]]
                attr = graph.edge_attr(path[j], path[j + 1])
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(width=5 - i * 1.5, color=color),
                    name=label if j == 0 else None,
                    showlegend=(j == 0),
                    legendgroup=label,
                    hoverinfo='text',
                    text=[f"{path[j]} → {path[j+1]}<br>"
                          f"Cost: {attr.cost} | Lat: {attr.latency}ms | Rel: {attr.reliability:.3f}"
                          if attr else ""] * 2,
                ))

    # Draw nodes
    node_list = sorted(graph.nodes)
    nx_vals = [NODE_POSITIONS[n][0] for n in node_list]
    ny_vals = [NODE_POSITIONS[n][1] for n in node_list]

    # Determine node colors
    node_colors = []
    node_sizes = []
    for n in node_list:
        if highlight_nodes and n in highlight_nodes:
            node_colors.append("#00C6FF")
            node_sizes.append(28)
        else:
            node_colors.append("#1E3A5F")
            node_sizes.append(22)

    fig.add_trace(go.Scatter(
        x=nx_vals, y=ny_vals,
        mode='markers+text',
        marker=dict(size=node_sizes, color=node_colors,
                    line=dict(width=2, color="#00C6FF")),
        text=[f"<b>{n}</b>" for n in node_list],
        textposition='top center',
        textfont=dict(color='#e2e8f0', size=11),
        hovertext=[f"<b>{n}</b> — {NODE_LABELS[n]}" for n in node_list],
        hoverinfo='text',
        showlegend=False,
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.08, 1.02]),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        title=dict(text=title, font=dict(color='#e2e8f0', size=14)),
        legend=dict(bgcolor='rgba(15,23,42,0.9)', bordercolor='#1e293b',
                    font=dict(color='#e2e8f0', size=11),
                    x=0.01, y=0.99),
        dragmode='pan',
    )
    return fig


def make_convergence_chart(snapshots, dijkstra_cost_val, title="GA Convergence"):
    """Plotly convergence chart showing GA approaching/beating Dijkstra."""
    gens = [s.gen for s in snapshots]
    bests = [s.best_cost if s.best_cost < 1e6 else None for s in snapshots]
    avgs = [s.avg_cost if s.avg_cost < 1e6 else None for s in snapshots]
    divs = [s.diversity for s in snapshots]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=gens, y=bests, mode='lines', name='Best Cost',
        line=dict(color='#00C6FF', width=3),
        fill='tozeroy', fillcolor='rgba(0,198,255,0.08)',
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=avgs, mode='lines', name='Population Avg',
        line=dict(color='#7C3AED', width=2, dash='dash'), opacity=0.7,
    ))
    if dijkstra_cost_val < 1e6:
        fig.add_hline(y=dijkstra_cost_val, line_dash="dot",
                      line_color="#F97316", line_width=2,
                      annotation_text=f"Dijkstra Optimal ({dijkstra_cost_val:.1f})",
                      annotation_font_color="#F97316")

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Generation", color='#94a3b8', gridcolor='#1e293b'),
        yaxis=dict(title="Path Cost", color='#94a3b8', gridcolor='#1e293b'),
        title=dict(text=title, font=dict(color='#e2e8f0', size=14)),
        legend=dict(bgcolor='rgba(15,23,42,0.9)', bordercolor='#1e293b',
                    font=dict(color='#e2e8f0')),
        margin=dict(l=50, r=20, t=50, b=40),
        height=400,
    )
    return fig


def make_diversity_chart(snapshots):
    """Population diversity over generations."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[s.gen for s in snapshots],
        y=[s.diversity for s in snapshots],
        mode='lines+markers', name='Diversity',
        line=dict(color='#A78BFA', width=2),
        marker=dict(size=4, color='#A78BFA'),
        fill='tozeroy', fillcolor='rgba(167,139,250,0.08)',
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Generation", color='#94a3b8', gridcolor='#1e293b'),
        yaxis=dict(title="Diversity (unique/total)", color='#94a3b8',
                   gridcolor='#1e293b', range=[0, 1.05]),
        title=dict(text="Population Diversity", font=dict(color='#e2e8f0', size=14)),
        margin=dict(l=50, r=20, t=50, b=40), height=300,
    )
    return fig


def render_result_card(label, metrics, is_winner=False, is_failed=False):
    """Render a styled result card for an algorithm."""
    if is_failed:
        border_color = "#ef4444"
        badge = '<span class="badge-fail">❌ CONSTRAINT VIOLATED</span>'
    elif is_winner:
        border_color = "#00C6FF"
        badge = '<span class="badge-win">🏆 WINNER</span>'
    else:
        border_color = "#059669"
        badge = '<span class="badge-tie">✅ VALID</span>'

    path_str = " → ".join(metrics.path) if metrics.path else "—"
    violation_html = ""
    if metrics.violation:
        violation_html = f'<div style="color:#ef4444;font-size:0.85em;margin-top:6px;">⚠️ {metrics.violation}</div>'

    st.markdown(f"""
    <div class="metric-card" style="border-color:{border_color};">
        <h4>{label}</h4>
        {badge}
        <div style="margin-top:12px;">
            <div><b>Cost:</b> <span class="value" style="color:#00C6FF;">{metrics.cost:.1f}</span></div>
            <div class="detail">Latency: {metrics.latency:.1f} ms</div>
            <div class="detail">Reliability: {metrics.reliability:.4f}</div>
        </div>
        <div class="route-path" style="margin-top:10px;">{path_str}</div>
        {violation_html}
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD NETWORK (cached)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource
def get_network():
    return build_network()


graph = get_network()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<div style="text-align:center; padding: 10px 0 20px 0;">
    <h1 style="font-size:2.2em; background: linear-gradient(90deg, #00C6FF, #7C3AED);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               margin-bottom: 4px;">
        🧬 Optimal Network Routing
    </h1>
    <p style="color:#94a3b8; font-size:1.1em; max-width:700px; margin:0 auto;">
        Genetic Algorithm vs Dijkstra — an interactive demonstration of
        why heuristic search beats classical algorithms for constrained routing
    </p>
</div>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tab_overview, tab_network, tab_lab, tab_evolution, tab_challenges = st.tabs([
    "📋 Overview", "🗺️ Network", "🔬 Route Lab", "🧬 GA Evolution", "🏆 Challenges"
])


# ━━━━━━━ TAB 1: OVERVIEW ━━━━━━━━━━━━━━━━━
with tab_overview:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### The Problem
        Real-world network routing isn't just about finding the cheapest path.
        Networks must satisfy **Quality of Service (QoS) constraints**:

        - ⏱️ **Latency limits** — packets must arrive within a time bound
        - 🛡️ **Reliability thresholds** — the route must maintain uptime guarantees
        - 💰 **Cost optimization** — minimize cost *within* these constraints

        **Dijkstra's algorithm** finds the mathematically optimal path for a *single* metric
        (e.g., cheapest). But it **cannot** simultaneously optimize cost while respecting
        latency and reliability constraints. This is the **Constrained Shortest Path Problem**,
        which is **NP-Hard**.

        ### The Solution: Genetic Algorithms
        A Genetic Algorithm (GA) evolves a *population* of candidate routes over many
        *generations*, using:
        - **Selection** — fittest routes survive
        - **Crossover** — combine good routes to create better ones
        - **Mutation** — random changes to explore new paths
        - **Constraint enforcement** — routes violating QoS get eliminated

        The GA can search the vast space of constrained-valid routes and find solutions
        that Dijkstra simply cannot.
        """)

    with col2:
        st.markdown("""
        ### Quick Stats
        """)
        st.metric("Network Nodes", "30 Indian Cities")
        st.metric("Network Links", f"{len(get_all_edges(graph))} bidirectional")
        st.metric("Link Types", "Fiber · Cable · Satellite")

        st.markdown("---")
        st.markdown("### How This Demo Works")
        st.markdown("""
        1. **🗺️ Network** — Explore the ISP backbone
        2. **🔬 Route Lab** — Set constraints & compare algorithms
        3. **🧬 Evolution** — Watch the GA evolve in real-time
        4. **🏆 Challenges** — Pre-built scenarios proving GA superiority
        """)

    st.markdown("---")
    st.markdown("### The Key Insight")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <h4>🛰️ Satellite Link</h4>
            <div class="value" style="color:#EF4444;">Cost: 2</div>
            <div class="detail">Latency: 120ms · Reliability: 85%</div>
            <div class="detail" style="margin-top:6px;">Dirt cheap — but terrible QoS.<br>Dijkstra loves this path!</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <h4>🔌 Cable Link</h4>
            <div class="value" style="color:#F97316;">Cost: 3–7</div>
            <div class="detail">Latency: 4-20ms · Reliability: 92-98%</div>
            <div class="detail" style="margin-top:6px;">The backbone of the internet.<br>Good balance of all metrics.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <h4>💎 Fiber Link</h4>
            <div class="value" style="color:#00C6FF;">Cost: 3–10</div>
            <div class="detail">Latency: 3-8ms · Reliability: 99.5-99.9%</div>
            <div class="detail" style="margin-top:6px;">Premium performance.<br>The GA finds these when needed!</div>
        </div>
        """, unsafe_allow_html=True)


# ━━━━━━━ TAB 2: NETWORK ━━━━━━━━━━━━━━━━━
with tab_network:
    st.markdown('<div class="section-header">🗺️ ISP Backbone Network — 30 Indian Cities</div>',
                unsafe_allow_html=True)
    st.markdown("Hover over edges to see cost, latency, and reliability. "
                "Colors indicate link type.")

    fig_net = create_network_figure(graph, title="Indian ISP Backbone Network", height=550)
    st.plotly_chart(fig_net, use_container_width=True, config={"displayModeBar": False})

    # Link type legend
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<span class="link-dot" style="background:#00C6FF;"></span> '
                    '**Fiber** — Low latency, high reliability, moderate cost',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<span class="link-dot" style="background:#F97316;"></span> '
                    '**Cable** — Cheap, moderate latency & reliability',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<span class="link-dot" style="background:#EF4444;"></span> '
                    '**Satellite** — Dirt cheap, terrible latency & reliability',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 All Network Links</div>',
                unsafe_allow_html=True)

    edges = get_all_edges(graph)
    edge_data = []
    for u, v, attr in edges:
        edge_data.append({
            "From": f"{u} ({NODE_LABELS[u]})",
            "To": f"{v} ({NODE_LABELS[v]})",
            "Type": attr.link_type,
            "Cost": attr.cost,
            "Latency (ms)": attr.latency,
            "Reliability": f"{attr.reliability:.3f}",
        })
    st.dataframe(edge_data, use_container_width=True, hide_index=True)


# ━━━━━━━ TAB 3: ROUTE LAB ━━━━━━━━━━━━━━━
with tab_lab:
    st.markdown('<div class="section-header">🔬 Interactive Route Comparison</div>',
                unsafe_allow_html=True)
    st.markdown("Pick source & destination, set QoS constraints, and watch the GA "
                "outperform Dijkstra in real-time.")

    # Controls
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.5, 1.5, 2, 2])
    cities = sorted(graph.nodes)
    with ctrl1:
        src = st.selectbox("🟢 Source", cities, index=cities.index("DEL"), key="lab_src")
    with ctrl2:
        dst = st.selectbox("🔴 Destination", cities, index=cities.index("CHN"), key="lab_dst")
    with ctrl3:
        max_lat = st.slider("⏱️ Max Latency (ms)", 10.0, 200.0, 50.0, 5.0, key="lab_lat",
                            help="Reject routes exceeding this latency")
    with ctrl4:
        min_rel = st.slider("🛡️ Min Reliability", 0.0, 0.999, 0.0, 0.01, key="lab_rel",
                            format="%.3f",
                            help="Reject routes below this reliability")

    with st.expander("🧬 GA Parameters (Advanced)", expanded=False):
        ga_c1, ga_c2, ga_c3 = st.columns(3)
        with ga_c1:
            pop_size = st.slider("Population", 20, 120, 60, key="lab_pop")
        with ga_c2:
            max_gen = st.slider("Generations", 20, 200, 100, key="lab_gen")
        with ga_c3:
            st.markdown("**Multi-Objective Weights**")
            alpha = st.number_input("Cost (α)", 0.0, 5.0, 1.0, 0.1, key="lab_a")
            beta = st.number_input("Latency (β)", 0.0, 5.0, 0.0, 0.1, key="lab_b")
            gamma = st.number_input("Reliability (γ)", 0.0, 5.0, 0.0, 0.1, key="lab_g")

    if src == dst:
        st.warning("Source and destination must be different.")
    else:
        run_btn = st.button("🚀 Run Comparison", use_container_width=True,
                            type="primary", key="lab_run")

        if run_btn:
            with st.spinner("Running algorithms..."):
                random.seed(42)
                np.random.seed(42)

                # Dijkstra variants
                dc_path, _ = dijkstra_cost(graph, src, dst)
                dl_path, _ = dijkstra_latency(graph, src, dst)
                dr_path, _ = dijkstra_reliability(graph, src, dst)

                dc_m = graph.path_metrics(dc_path, src, dst, max_lat, min_rel)
                dl_m = graph.path_metrics(dl_path, src, dst, max_lat, min_rel)
                dr_m = graph.path_metrics(dr_path, src, dst, max_lat, min_rel)

                # Best valid Dijkstra
                valid_dijk = [m for m in [dc_m, dl_m, dr_m] if m.valid]
                best_dijk = min(valid_dijk, key=lambda m: m.cost) if valid_dijk else None

                # GA
                use_cost_only = (beta == 0.0 and gamma == 0.0)
                cfg = GAConfig(
                    pop_size=pop_size, max_gen=max_gen,
                    max_latency=max_lat, min_reliability=min_rel,
                    alpha=alpha, beta=beta, gamma=gamma,
                    cost_only=use_cost_only,
                )
                ga_result = run_ga(graph, src, dst, cfg)
                ga_m = graph.path_metrics(ga_result.best_path, src, dst, max_lat, min_rel)

            # ── WINNER DETERMINATION ──
            if not ga_m.valid and not best_dijk:
                winner = "none"
            elif not ga_m.valid:
                winner = "dijkstra"
            elif not best_dijk:
                winner = "ga"
            elif ga_m.cost < best_dijk.cost - 0.01:
                winner = "ga"
            elif best_dijk.cost < ga_m.cost - 0.01:
                winner = "dijkstra"
            else:
                winner = "tie"

            # ── WINNER BANNER ──
            if winner == "ga":
                savings = best_dijk.cost - ga_m.cost if best_dijk else ga_m.cost
                pct = (savings / best_dijk.cost * 100) if best_dijk and best_dijk.cost > 0 else 0
                if not best_dijk:
                    st.markdown(f"""
                    <div class="winner-banner winner-ga">
                        <span style="font-size:1.8em;">🏆</span><br>
                        <b style="font-size:1.3em; color:#00C6FF;">GA WINS — Only Valid Solver!</b><br>
                        <span style="color:#94a3b8;">All Dijkstra variants violated constraints.
                        GA found a valid route at cost {ga_m.cost:.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="winner-banner winner-ga">
                        <span style="font-size:1.8em;">🏆</span><br>
                        <b style="font-size:1.3em; color:#00C6FF;">GA WINS — {pct:.1f}% Cheaper!</b><br>
                        <span style="color:#94a3b8;">GA route costs {ga_m.cost:.1f} vs Dijkstra's best valid {best_dijk.cost:.1f}
                        — saving {savings:.1f} per packet</span>
                    </div>
                    """, unsafe_allow_html=True)
            elif winner == "tie":
                st.markdown(f"""
                <div class="winner-banner winner-tie">
                    <span style="font-size:1.8em;">🤝</span><br>
                    <b style="font-size:1.3em; color:#F97316;">TIE — Both Found Optimal</b><br>
                    <span style="color:#94a3b8;">Both algorithms found cost {ga_m.cost:.1f}.
                    Try adding constraints to see the GA advantage!</span>
                </div>
                """, unsafe_allow_html=True)

            # ── RESULT CARDS ──
            st.markdown("---")
            c1, c2, c3 = st.columns(3)

            with c1:
                render_result_card(
                    "Dijkstra (Min Cost)",
                    dc_m,
                    is_winner=(winner == "dijkstra" and best_dijk and dc_m.cost == best_dijk.cost and dc_m.valid),
                    is_failed=not dc_m.valid,
                )
            with c2:
                dijk_label = "Dijkstra (Min Latency)"
                render_result_card(
                    dijk_label,
                    dl_m,
                    is_winner=(winner == "dijkstra" and best_dijk and dl_m.cost == best_dijk.cost and dl_m.valid),
                    is_failed=not dl_m.valid,
                )
            with c3:
                render_result_card(
                    "Genetic Algorithm",
                    ga_m,
                    is_winner=(winner == "ga"),
                    is_failed=not ga_m.valid,
                )

            # ── NETWORK MAP ──
            st.markdown("---")
            st.markdown('<div class="section-header">🗺️ Route Visualization</div>',
                        unsafe_allow_html=True)

            paths_to_show = []
            labels = []
            colors = []

            if dc_m.path:
                paths_to_show.append(dc_m.path)
                labels.append(f"Dijkstra (cost={dc_m.cost:.0f})" +
                              (" ❌" if not dc_m.valid else ""))
                colors.append("#F97316")
            if ga_m.path:
                paths_to_show.append(ga_m.path)
                labels.append(f"GA (cost={ga_m.cost:.0f})" +
                              (" 🏆" if winner == "ga" else ""))
                colors.append("#00C6FF")

            fig = create_network_figure(
                graph, paths=paths_to_show, path_labels=labels,
                path_colors=colors,
                highlight_nodes={src, dst},
                title=f"Routing: {src} → {dst}"
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

            # ── STORE GA RESULT FOR EVOLUTION TAB ──
            st.session_state['last_ga_result'] = ga_result
            st.session_state['last_dc_cost'] = dc_m.cost if dc_m.valid else float('inf')
            st.session_state['last_src'] = src
            st.session_state['last_dst'] = dst


# ━━━━━━━ TAB 4: GA EVOLUTION ━━━━━━━━━━━━━
with tab_evolution:
    st.markdown('<div class="section-header">🧬 GA Evolution Deep Dive</div>',
                unsafe_allow_html=True)

    if 'last_ga_result' not in st.session_state:
        st.info("👆 Run a comparison in the **Route Lab** tab first, then come here "
                "to see the GA's generation-by-generation evolution.")
    else:
        ga_res = st.session_state['last_ga_result']
        dc_cost = st.session_state['last_dc_cost']
        src_e = st.session_state['last_src']
        dst_e = st.session_state['last_dst']

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Final Cost", f"{ga_res.best_cost:.1f}")
        with c2:
            st.metric("Generations", len(ga_res.generations))
        with c3:
            st.metric("Execution Time", f"{ga_res.time_taken * 1000:.0f} ms")
        with c4:
            if ga_res.generations:
                first_valid_gen = next(
                    (s.gen for s in ga_res.generations if s.best_cost < 1e6),
                    None
                )
                st.metric("First Valid Solution", f"Gen {first_valid_gen}"
                          if first_valid_gen is not None else "—")

        # Convergence chart
        st.markdown("---")
        conv_fig = make_convergence_chart(ga_res.generations, dc_cost,
                                          f"Convergence: {src_e} → {dst_e}")
        st.plotly_chart(conv_fig, use_container_width=True,
                        config={"displayModeBar": False})

        # Diversity chart
        div_fig = make_diversity_chart(ga_res.generations)
        st.plotly_chart(div_fig, use_container_width=True,
                        config={"displayModeBar": False})

        # Path evolution table
        st.markdown("---")
        st.markdown('<div class="section-header">📜 Best Path at Key Generations</div>',
                    unsafe_allow_html=True)

        key_gens = [0, len(ga_res.generations) // 4,
                    len(ga_res.generations) // 2,
                    3 * len(ga_res.generations) // 4,
                    len(ga_res.generations) - 1]
        key_gens = sorted(set(g for g in key_gens if 0 <= g < len(ga_res.generations)))

        path_evo_data = []
        for gi in key_gens:
            s = ga_res.generations[gi]
            path_evo_data.append({
                "Generation": s.gen,
                "Best Cost": f"{s.best_cost:.1f}" if s.best_cost < 1e6 else "∞",
                "Avg Cost": f"{s.avg_cost:.1f}" if s.avg_cost < 1e6 else "∞",
                "Diversity": f"{s.diversity:.2f}",
                "Best Path": " → ".join(s.best_path) if s.best_path else "—",
            })
        st.dataframe(path_evo_data, use_container_width=True, hide_index=True)


# ━━━━━━━ TAB 5: CHALLENGE SCENARIOS ━━━━━━
with tab_challenges:
    st.markdown('<div class="section-header">🏆 Challenge Scenarios — Proving GA Superiority</div>',
                unsafe_allow_html=True)
    st.markdown("Select a pre-configured real-world scenario to see how each "
                "algorithm handles it. The GA's constraint-handling ability is the key differentiator.")

    scenario_name = st.selectbox(
        "Choose a scenario:",
        list(CHALLENGES.keys()),
        key="challenge_select",
    )

    scenario = CHALLENGES[scenario_name]
    st.markdown(scenario["description"])

    if st.button("⚡ Run Challenge", type="primary", key="challenge_run",
                 use_container_width=True):
        with st.spinner("Executing challenge..."):
            random.seed(42)
            np.random.seed(42)

            ch_src = scenario["src"]
            ch_dst = scenario["dst"]
            ch_max_lat = scenario["max_latency"]
            ch_min_rel = scenario["min_reliability"]

            # Dijkstra
            dc_path, _ = dijkstra_cost(graph, ch_src, ch_dst)
            dl_path, _ = dijkstra_latency(graph, ch_src, ch_dst)
            dr_path, _ = dijkstra_reliability(graph, ch_src, ch_dst)

            dc_m = graph.path_metrics(dc_path, ch_src, ch_dst, ch_max_lat, ch_min_rel)
            dl_m = graph.path_metrics(dl_path, ch_src, ch_dst, ch_max_lat, ch_min_rel)
            dr_m = graph.path_metrics(dr_path, ch_src, ch_dst, ch_max_lat, ch_min_rel)

            valid_dijk = [m for m in [dc_m, dl_m, dr_m] if m.valid]
            best_dijk = min(valid_dijk, key=lambda m: m.cost) if valid_dijk else None

            # GA
            cfg = GAConfig(pop_size=60, max_gen=100, cost_only=True,
                           max_latency=ch_max_lat, min_reliability=ch_min_rel)
            ga_res = run_ga(graph, ch_src, ch_dst, cfg)
            ga_m = graph.path_metrics(ga_res.best_path, ch_src, ch_dst, ch_max_lat, ch_min_rel)

        # Results
        st.markdown("---")
        st.markdown(f"**Constraints:** Max Latency = `{ch_max_lat}ms` · "
                    f"Min Reliability = `{ch_min_rel:.3f}`")

        # Dijkstra results summary
        st.markdown("#### Dijkstra Results")
        dijk_cols = st.columns(3)

        dijk_variants = [
            ("Min Cost", dc_m),
            ("Min Latency", dl_m),
            ("Max Reliability", dr_m),
        ]
        for col, (label, m) in zip(dijk_cols, dijk_variants):
            with col:
                status = "✅" if m.valid else "❌"
                st.markdown(f"""
                <div class="metric-card" style="border-color:{'#059669' if m.valid else '#ef4444'};">
                    <h4>{label} {status}</h4>
                    <div>Cost: <b>{m.cost:.1f}</b> · Lat: {m.latency:.1f}ms · Rel: {m.reliability:.4f}</div>
                    <div class="route-path" style="font-size:0.8em;">{' → '.join(m.path)}</div>
                    {'<div style="color:#ef4444;font-size:0.8em;">⚠️ ' + m.violation + '</div>' if m.violation else ''}
                </div>
                """, unsafe_allow_html=True)

        # GA result
        st.markdown("#### Genetic Algorithm Result")
        if ga_m.valid:
            if best_dijk:
                savings = best_dijk.cost - ga_m.cost
                pct = savings / best_dijk.cost * 100 if best_dijk.cost > 0 else 0
                if savings > 0.01:
                    st.markdown(f"""
                    <div class="winner-banner winner-ga">
                        <span style="font-size:2em;">🏆</span><br>
                        <b style="font-size:1.5em; color:#00C6FF;">GA WINS!</b><br>
                        <span style="color:#e2e8f0; font-size:1.1em;">
                            Cost: <b>{ga_m.cost:.1f}</b> vs Dijkstra's best valid: <b>{best_dijk.cost:.1f}</b><br>
                            Savings: <b>{savings:.1f}</b> ({pct:.1f}% cheaper)<br>
                            Route: <span class="route-path">{' → '.join(ga_m.path)}</span>
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="winner-banner winner-tie">
                        <span style="font-size:2em;">🤝</span><br>
                        <b style="font-size:1.5em; color:#F97316;">TIE</b><br>
                        <span style="color:#e2e8f0;">Both found cost {ga_m.cost:.1f}.
                        Route: <span class="route-path">{' → '.join(ga_m.path)}</span></span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="winner-banner winner-ga">
                    <span style="font-size:2em;">🏆</span><br>
                    <b style="font-size:1.5em; color:#00C6FF;">GA WINS — Only Valid Solver!</b><br>
                    <span style="color:#e2e8f0;">All 3 Dijkstra variants violated constraints!<br>
                    GA found: cost={ga_m.cost:.1f}, lat={ga_m.latency:.1f}ms, rel={ga_m.reliability:.4f}<br>
                    Route: <span class="route-path">{' → '.join(ga_m.path)}</span></span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("No algorithm found a valid route under these constraints.")

        # Network map for challenge
        paths_to_show = []
        labels = []
        colors = []
        if dc_m.path:
            paths_to_show.append(dc_m.path)
            labels.append(f"Dijkstra Min-Cost {'❌' if not dc_m.valid else '✓'}")
            colors.append("#F97316")
        if ga_m.path and ga_m.valid:
            paths_to_show.append(ga_m.path)
            labels.append(f"GA Solution 🏆")
            colors.append("#00C6FF")

        fig = create_network_figure(
            graph, paths=paths_to_show, path_labels=labels,
            path_colors=colors, highlight_nodes={ch_src, ch_dst},
            title=f"Challenge: {ch_src} → {ch_dst}",
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
