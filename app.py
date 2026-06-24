"""
Tabnine AI Model Comparison Dashboard
======================================
Streamlit app providing a live, interactive comparison of every AI model
available in Tabnine — covering capability scores, official benchmarks,
hardware requirements, and task-specific recommendations.

Data sources
------------
- Static: Tabnine official docs (ai-models, models-settings, system-requirements)
- Live:   EvalPlus HumanEval+, PapersWithCode SWE-bench, LiveCodeBench APIs
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data.models import MODELS, TASKS
from data.live_benchmarks import fetch_all_live_scores
from utils.dataframe import (
    build_master_df,
    capability_long_df,
    benchmark_long_df,
    best_model_for_task,
    compute_overall_score,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Tabnine Model Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Brand palette (Tabnine blue)
# ---------------------------------------------------------------------------

TABNINE_BLUE  = "#1F46C1"
TABNINE_DARK  = "#131A3A"
ACCENT_ORANGE = "#FE9A00"
ACCENT_GREEN  = "#00C950"

PROVIDER_COLORS = {
    "OpenAI":    "#10A37F",
    "Anthropic": "#D4A96A",
    "Google":    "#4285F4",
    "xAI":       "#000000",
    "Mistral":   "#FF7000",
    "MiniMax":   "#7C3AED",
    "Poolside":  "#0EA5E9",
    "Kodu AI":   "#E11D48",
    "Tabnine":   TABNINE_BLUE,
}

# ---------------------------------------------------------------------------
# Data-source citations — shown in chart hover tooltips and table header tips
# ---------------------------------------------------------------------------

SOURCES: dict[str, str] = {
    # Capability scores
    "Code Completion":  "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Code Generation":  "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Reasoning":        "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Chat":             "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Debugging":        "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Refactoring":      "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Documentation":    "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Multi-file":       "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Overall Score":    "Weighted average of 8 capability scores · Tabnine AI Models docs",
    # Benchmarks
    "HumanEval (%)":       "EvalPlus Leaderboard · evalplus.github.io/leaderboard.html · pass@1 on HumanEval+",
    "MBPP (%)":            "EvalPlus Leaderboard · evalplus.github.io/leaderboard.html · pass@1 on MBPP+",
    "SWE-bench (%)":       "SWE-bench Verified · swebench.com · % of GitHub issues resolved end-to-end",
    "GPQA (%)":            "Published model cards & papers · Graduate-level STEM Q&A (Diamond set)",
    "MMLU (%)":            "Published model cards & papers · Massive Multitask Language Understanding",
    "LiveCodeBench (%)":   "LiveCodeBench · livecodebench.github.io · contamination-free coding benchmark",
    # Technical / hardware specs
    "Context (K)":  "Tabnine System Requirements · docs.tabnine.com/main/welcome/readme/system-requirements",
    "GPU (min)":    "Tabnine System Requirements · docs.tabnine.com/main/welcome/readme/system-requirements",
    "VRAM (GB)":    "Tabnine System Requirements · docs.tabnine.com/main/welcome/readme/system-requirements",
    # Model metadata
    "Provider":    "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Family":      "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Category":    "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Deployment":  "Tabnine Model Settings · docs.tabnine.com/main/administering-tabnine/managing-your-team/settings/models-settings",
    "Plan":        "Tabnine Model Settings · docs.tabnine.com/main/administering-tabnine/managing-your-team/settings/models-settings",
    "Thinking":    "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Tool Calling":"Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "License":            "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Status":             "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
    "Tabnine Available":  "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models",
}

# Generic fallback shown when a specific key is not in SOURCES
_FALLBACK_SOURCE = "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models"


def with_source(df_slice: pd.DataFrame, key: str) -> pd.DataFrame:
    """Return a copy of *df_slice* with a '📄 Source' column appended.

    Pass the result straight to a Plotly Express call and include
    ``hover_data={"📄 Source": True}`` — the source citation then appears
    at the bottom of every hover tooltip.
    """
    out = df_slice.copy()
    out["📄 Source"] = SOURCES.get(key, _FALLBACK_SOURCE)
    return out


def col_cfg(*field_names: str) -> dict:
    """Build a ``column_config`` dict for ``st.dataframe``.

    Each named field gets a ``help=`` tooltip (the ℹ icon on the column
    header) showing its authoritative source when the user hovers over it.

    Usage::

        st.dataframe(df, column_config=col_cfg("HumanEval (%)", "MBPP (%)"))
    """
    config = {}
    for name in field_names:
        source = SOURCES.get(name, _FALLBACK_SOURCE)
        config[name] = st.column_config.Column(help=f"Source: {source}")
    return config

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <style>
      .hero {{
          background: linear-gradient(135deg, {TABNINE_DARK} 0%, {TABNINE_BLUE} 100%);
          color: white;
          padding: 2rem 2.5rem 1.5rem;
          border-radius: 12px;
          margin-bottom: 1.5rem;
      }}
      .hero h1 {{ margin: 0; font-size: 2.2rem; font-weight: 800; }}
      .hero p  {{ margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }}

      .metric-card {{
          background: #f8faff;
          border: 1px solid #dce8ff;
          border-radius: 10px;
          padding: 1rem 1.2rem;
          text-align: center;
      }}
      .metric-card h2 {{ color: {TABNINE_BLUE}; font-size: 2rem; margin: 0; }}
      .metric-card p  {{ color: #555; font-size: 0.82rem; margin: 0.2rem 0 0; }}

      .winner-badge {{
          background: {TABNINE_BLUE};
          color: white;
          padding: 0.25rem 0.8rem;
          border-radius: 999px;
          font-size: 0.78rem;
          font-weight: 600;
          display: inline-block;
      }}

      .section-header {{
          border-left: 4px solid {TABNINE_BLUE};
          padding-left: 0.75rem;
          margin: 1.5rem 0 0.75rem;
          font-weight: 700;
          font-size: 1.1rem;
          color: {TABNINE_DARK};
      }}

      .live-badge {{
          background: {ACCENT_GREEN};
          color: white;
          padding: 0.2rem 0.6rem;
          border-radius: 999px;
          font-size: 0.72rem;
          font-weight: 700;
          vertical-align: middle;
      }}
      .stale-badge {{
          background: {ACCENT_ORANGE};
          color: white;
          padding: 0.2rem 0.6rem;
          border-radius: 999px;
          font-size: 0.72rem;
          font-weight: 700;
          vertical-align: middle;
      }}

      /* Availability badges */
      .avail-yes {{
          background: {ACCENT_GREEN};
          color: white;
          padding: 0.18rem 0.6rem;
          border-radius: 999px;
          font-size: 0.75rem;
          font-weight: 600;
          white-space: nowrap;
      }}
      .avail-soon {{
          background: {ACCENT_ORANGE};
          color: white;
          padding: 0.18rem 0.6rem;
          border-radius: 999px;
          font-size: 0.75rem;
          font-weight: 600;
          white-space: nowrap;
      }}

      /* Source-aware HTML table */
      .src-table {{
          width: 100%;
          border-collapse: collapse;
          font-size: 0.82rem;
          margin-top: 0.5rem;
      }}
      .src-table th {{
          background: {TABNINE_DARK};
          color: white;
          padding: 0.45rem 0.7rem;
          text-align: left;
          font-weight: 600;
          white-space: nowrap;
      }}
      .src-table tr:nth-child(even) {{ background: #f4f7ff; }}
      .src-table tr:hover td {{ background: #dce8ff !important; }}
      .src-table td {{
          padding: 0.38rem 0.7rem;
          border-bottom: 1px solid #e5eaf5;
          cursor: default;
      }}
      /* Dim rows for coming-soon models */
      .src-table tr.coming-soon td {{
          opacity: 0.55;
          font-style: italic;
      }}
      .src-table tr.coming-soon:hover td {{
          opacity: 0.8 !important;
          background: #fff8e8 !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pchart(fig, height=None):
    """Render a Plotly figure full-width, compatible with Streamlit 1.58+."""
    st.plotly_chart(fig, use_container_width=True)


def dtable(df_arg, height=None, column_config=None):
    """Render a DataFrame full-width, compatible with Streamlit 1.58+.

    Pass ``column_config=col_cfg(...)`` to attach source-citation tooltips
    to individual column headers.
    """
    kwargs = {"use_container_width": True}
    if height:
        kwargs["height"] = height
    if column_config:
        kwargs["column_config"] = column_config
    st.dataframe(df_arg, **kwargs)


def html_table(df_arg: pd.DataFrame, max_rows: int = 200) -> None:
    """Render *df_arg* as a styled HTML table with per-cell hover source tooltips.

    Each cell carries a ``title`` attribute — the browser shows it as a
    native tooltip on hover (no JS required). Coming-soon models are
    rendered in a dimmed italic style.

    Args:
        df_arg:   DataFrame to render. If it has a 'Tabnine Available' column,
                  that is used to apply the coming-soon row class and is then
                  dropped from display.
        max_rows: Safety cap to avoid giant DOM payloads.
    """
    import html as _html

    display_df = df_arg.head(max_rows).copy()

    # Extract availability info before dropping the sentinel column
    if "Tabnine Available" in display_df.columns:
        avail_map = display_df["Tabnine Available"].to_dict()
        display_df = display_df.drop(columns=["Tabnine Available"])
    else:
        avail_map = {}

    def _cell_source(col: str) -> str:
        return SOURCES.get(col, _FALLBACK_SOURCE)

    def _fmt(val) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "—"
        return _html.escape(str(val))

    # Build header
    headers = "".join(
        f'<th title="Source: {_cell_source(col)}">{_html.escape(str(col))}</th>'
        for col in display_df.columns
    )

    # Build rows
    rows_html = []
    for idx, row in display_df.iterrows():
        is_avail = avail_map.get(idx, True)
        row_class = "" if is_avail else " class=\"coming-soon\""
        cells = "".join(
            f'<td title="📄 Source: {_cell_source(col)}">{_fmt(row[col])}</td>'
            for col in display_df.columns
        )
        rows_html.append(f"<tr{row_class}>{cells}</tr>")

    table_html = (
        f'<div style="overflow-x:auto;max-height:520px;overflow-y:auto">'
        f'<table class="src-table">'
        f'<thead><tr>{headers}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Live data loading (cached 1 h)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_live_scores() -> tuple[dict, bool]:
    try:
        scores = fetch_all_live_scores()
        is_live = bool(scores)
    except Exception:
        scores = {}
        is_live = False
    return scores, is_live


# ---------------------------------------------------------------------------
# Build DataFrames
# ---------------------------------------------------------------------------

with st.spinner("Fetching live benchmark data…"):
    live_scores, is_live = load_live_scores()

df_raw    = build_master_df(live_scores=live_scores)
df_scored = compute_overall_score(df_raw)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="hero">
      <h1>⚡ Tabnine Model Dashboard</h1>
      <p>
        Live comparison of every AI model available in Tabnine — capability scores,
        official benchmarks, hardware requirements, and task recommendations.
        &nbsp;&nbsp;
        <span class="{'live-badge' if is_live else 'stale-badge'}">
          {'● LIVE DATA' if is_live else '● STATIC FALLBACK'}
        </span>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://www.tabnine.com/wp-content/uploads/2023/05/tabnine-logo.svg",
        width=160,
    )
    st.markdown("---")
    st.markdown("### Filters")

    sel_providers = st.multiselect(
        "Provider",
        options=sorted(df_scored["Provider"].unique()),
        default=sorted(df_scored["Provider"].unique()),
    )
    sel_categories = st.multiselect(
        "Category",
        options=sorted(df_scored["Category"].unique()),
        default=sorted(df_scored["Category"].unique()),
    )
    sel_deployments = st.multiselect(
        "Deployment",
        options=sorted(df_scored["Deployment"].unique()),
        default=sorted(df_scored["Deployment"].unique()),
    )
    thinking_only    = st.checkbox("Thinking models only", value=False)
    show_upcoming    = st.checkbox("Include upcoming models (not yet in Tabnine)", value=True)

    st.markdown("---")
    st.markdown("### Compare Models")
    all_model_names = sorted(df_scored["Model"].tolist())
    compare_models = st.multiselect(
        "Select models to compare (max 8)",
        options=all_model_names,
        default=all_model_names[:4],
    )
    # Enforce max 8 client-side without relying on max_selections param
    if len(compare_models) > 8:
        st.warning("Please select at most 8 models.")
        compare_models = compare_models[:8]

    st.markdown("---")
    st.caption(
        "Data sources: [Tabnine Docs](https://docs.tabnine.com/main/welcome/readme/ai-models) · "
        "[EvalPlus](https://evalplus.github.io/leaderboard.html) · "
        "[SWE-bench](https://www.swebench.com/) · "
        "[LiveCodeBench](https://livecodebench.github.io/)"
    )

    st.markdown("### Data Sources")
    st.caption("Toggle which sources are visible across all tabs.")

    src_tabnine  = st.checkbox("Tabnine Docs (scores & metadata)", value=True, key="src_tabnine")
    src_evalplus = st.checkbox("EvalPlus (HumanEval / MBPP)",      value=True, key="src_evalplus")
    src_swebench = st.checkbox("SWE-bench Verified",               value=True, key="src_swebench")
    src_livecodebench = st.checkbox("LiveCodeBench",               value=True, key="src_livecodebench")
    src_gpqa_mmlu     = st.checkbox("Model Cards (GPQA / MMLU)",   value=True, key="src_gpqa_mmlu")

# ---------------------------------------------------------------------------
# Source-visibility gates — built from sidebar toggles
# ---------------------------------------------------------------------------

ACTIVE_BENCH_COLS: list[str] = []
if src_evalplus:
    ACTIVE_BENCH_COLS += ["HumanEval (%)", "MBPP (%)"]
if src_swebench:
    ACTIVE_BENCH_COLS += ["SWE-bench (%)"]
if src_gpqa_mmlu:
    ACTIVE_BENCH_COLS += ["GPQA (%)", "MMLU (%)"]
if src_livecodebench:
    ACTIVE_BENCH_COLS += ["LiveCodeBench (%)"]

CAP_COLS_ALL = [
    "Code Completion", "Code Generation", "Reasoning", "Chat",
    "Debugging", "Refactoring", "Documentation", "Multi-file",
]
ACTIVE_CAP_COLS = CAP_COLS_ALL if src_tabnine else []

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

mask = (
    df_scored["Provider"].isin(sel_providers) &
    df_scored["Category"].isin(sel_categories) &
    df_scored["Deployment"].isin(sel_deployments)
)
if thinking_only:
    mask &= df_scored["Thinking"] == "✓"
if not show_upcoming:
    mask &= df_scored["Tabnine Available"] == True

df = df_scored[mask].copy()

if df.empty:
    st.warning("No models match the current filters. Adjust the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f'<div class="metric-card"><h2>{len(df)}</h2><p>Models shown</p></div>',
        unsafe_allow_html=True,
    )
with col2:
    best_overall = df.loc[df["Overall Score"].idxmax(), "Model"]
    st.markdown(
        f'<div class="metric-card"><h2 style="font-size:1.1rem">{best_overall}</h2>'
        f'<p>Highest overall score</p></div>',
        unsafe_allow_html=True,
    )
with col3:
    n_thinking = (df["Thinking"] == "✓").sum()
    st.markdown(
        f'<div class="metric-card"><h2>{n_thinking}</h2><p>Thinking models</p></div>',
        unsafe_allow_html=True,
    )
with col4:
    n_self = (df["Deployment"] == "Self Hosted").sum()
    st.markdown(
        f'<div class="metric-card"><h2>{n_self}</h2><p>Self-hosted</p></div>',
        unsafe_allow_html=True,
    )
with col5:
    max_ctx = int(df["Context (K)"].max())
    st.markdown(
        f'<div class="metric-card"><h2>{max_ctx:,}K</h2><p>Max context window</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_compare, tab_benchmarks, tab_task, tab_hardware, tab_table = st.tabs([
    "Overview",
    "Side-by-Side Compare",
    "Benchmarks",
    "Task Advisor",
    "Hardware",
    "Full Table",
])

# ============================================================
# TAB 1 — Overview
# ============================================================
with tab_overview:

    st.markdown('<div class="section-header">Overall Model Scores</div>', unsafe_allow_html=True)

    _df_bar = with_source(df.sort_values("Overall Score", ascending=True), "Overall Score")
    fig_bar = px.bar(
        _df_bar,
        x="Overall Score",
        y="Model",
        color="Provider",
        color_discrete_map=PROVIDER_COLORS,
        orientation="h",
        text="Overall Score",
        range_x=[0, 10],
        height=max(400, len(df) * 38),
        labels={"Overall Score": "Weighted Average Score (0–10)"},
        hover_data={"📄 Source": True, "Status": True},
        pattern_shape="Status",
        pattern_shape_map={"✅ Available": "", "🔜 Coming Soon": "/"},
    )
    fig_bar.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        marker_opacity=_df_bar["Tabnine Available"].map({True: 1.0, False: 0.45}).tolist(),
    )
    fig_bar.update_layout(
        plot_bgcolor="white",
        legend_title_text="Provider",
        xaxis_showgrid=True,
        yaxis_title=None,
    )
    pchart(fig_bar)

    st.markdown('<div class="section-header">Code Generation vs Reasoning</div>', unsafe_allow_html=True)

    _df_scatter = with_source(df, "Code Generation")
    fig_scatter = px.scatter(
        _df_scatter,
        x="Reasoning",
        y="Code Generation",
        color="Provider",
        color_discrete_map=PROVIDER_COLORS,
        symbol="Status",
        symbol_map={"✅ Available": "circle", "🔜 Coming Soon": "diamond-open"},
        size="Context (K)",
        size_max=28,
        hover_name="Model",
        hover_data={"Deployment": True, "Thinking": True, "Context (K)": True,
                    "Status": True, "📄 Source": True},
        text="Model",
        labels={"Code Generation": "Code Generation Score", "Reasoning": "Reasoning Score"},
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=9)
    fig_scatter.update_layout(
        xaxis_range=[6.5, 10],
        yaxis_range=[6.5, 10],
        plot_bgcolor="white",
        height=500,
    )
    pchart(fig_scatter)

    st.markdown('<div class="section-header">Capability Heatmap (All Tasks)</div>', unsafe_allow_html=True)

    if not ACTIVE_CAP_COLS:
        st.info("Enable **Tabnine Docs** in the Data Sources sidebar to show capability scores.")
    else:
        heatmap_data = df.set_index("Model")[ACTIVE_CAP_COLS]
        fig_heat = px.imshow(
            heatmap_data,
            text_auto=".1f",
            color_continuous_scale="Blues",
            zmin=5,
            zmax=10,
            aspect="auto",
            height=max(400, len(df) * 36),
        )
        fig_heat.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            coloraxis_colorbar_title="Score",
        )
        fig_heat.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>%{x}: %{z:.1f}<br>"
                "<span style='color:#888;font-size:0.85em'>📄 Source: "
                "Tabnine AI Models docs · docs.tabnine.com/main/welcome/readme/ai-models"
                "</span><extra></extra>"
            )
        )
        pchart(fig_heat)


# ============================================================
# TAB 2 — Side-by-Side Compare
# ============================================================
with tab_compare:

    st.markdown('<div class="section-header">Spider / Radar Comparison</div>', unsafe_allow_html=True)

    df_cmp = df[df["Model"].isin(compare_models)].copy()

    if df_cmp.empty:
        st.info("Select at least one model in the sidebar to compare.")
    elif not ACTIVE_CAP_COLS:
        st.info("Enable **Tabnine Docs** in the Data Sources sidebar to show capability scores.")
    else:
        categories = ACTIVE_CAP_COLS + [ACTIVE_CAP_COLS[0]]

        fig_radar = go.Figure()
        _radar_source = SOURCES.get("Code Completion", _FALLBACK_SOURCE)
        for _, row in df_cmp.iterrows():
            values = [row[c] for c in ACTIVE_CAP_COLS] + [row[ACTIVE_CAP_COLS[0]]]
            provider = row["Provider"]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=row["Model"],
                line_color=PROVIDER_COLORS.get(provider, "#888"),
                opacity=0.65,
                hovertemplate=(
                    "<b>" + row["Model"] + "</b><br>"
                    "%{theta}: %{r:.1f}<br>"
                    "<span style='color:#888;font-size:0.85em'>"
                    f"📄 Source: {_radar_source}"
                    "</span><extra></extra>"
                ),
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            legend_title_text="Model",
            height=550,
        )
        pchart(fig_radar)

        st.markdown('<div class="section-header">Capability Scores — Grouped Bar</div>', unsafe_allow_html=True)

        long = with_source(capability_long_df(df_cmp), "Code Completion")
        fig_grp = px.bar(
            long,
            x="Task",
            y="Score",
            color="Model",
            barmode="group",
            range_y=[0, 10],
            height=420,
            hover_data={"📄 Source": True},
        )
        fig_grp.update_layout(plot_bgcolor="white", xaxis_title=None)
        pchart(fig_grp)

        st.markdown('<div class="section-header">Summary Table</div>', unsafe_allow_html=True)

        display_cols = ["Model", "Status", "Provider", "Category", "Deployment", "Context (K)",
                        "Thinking", "Tool Calling"] + ACTIVE_CAP_COLS + ["Overall Score"]
        st.caption("Hover any cell to see its data source. Dimmed rows = coming-soon models.")
        html_table(df_cmp[display_cols + ["Tabnine Available"]].set_index("Model"))


# ============================================================
# TAB 3 — Benchmarks
# ============================================================
with tab_benchmarks:

    st.markdown('<div class="section-header">Official Public Benchmarks</div>', unsafe_allow_html=True)

    if is_live:
        st.success("Benchmark data refreshed from live APIs (EvalPlus · SWE-bench · LiveCodeBench).")
    else:
        st.info(
            "Live APIs unavailable — showing static baseline scores from Tabnine docs "
            "and published model cards."
        )

    bench_cols = ["HumanEval (%)", "MBPP (%)", "SWE-bench (%)", "GPQA (%)", "MMLU (%)", "LiveCodeBench (%)"]
    # Filter to only columns whose source is enabled in the sidebar
    visible_bench_cols = [c for c in bench_cols if c in ACTIVE_BENCH_COLS]
    df_bench = df[["Model", "Provider"] + bench_cols].copy()

    if not visible_bench_cols:
        st.info("Enable at least one benchmark source in the **Data Sources** sidebar.")
        st.stop()

    bench_sel = st.selectbox("Select benchmark", options=visible_bench_cols, index=0)
    df_b_filtered = df_bench.dropna(subset=[bench_sel]).sort_values(bench_sel, ascending=True)

    if df_b_filtered.empty:
        st.warning(f"No models have reported {bench_sel} scores yet.")
    else:
        _df_bench_bar = with_source(df_b_filtered, bench_sel)
        fig_bench = px.bar(
            _df_bench_bar,
            x=bench_sel,
            y="Model",
            color="Provider",
            color_discrete_map=PROVIDER_COLORS,
            orientation="h",
            text=bench_sel,
            height=max(350, len(df_b_filtered) * 42),
            labels={bench_sel: f"{bench_sel} Score"},
            hover_data={"📄 Source": True},
        )
        fig_bench.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bench.update_layout(
            plot_bgcolor="white",
            yaxis_title=None,
            xaxis_range=[0, 100],
        )
        pchart(fig_bench)

    # Pure Plotly heatmap — no matplotlib dependency
    st.markdown('<div class="section-header">All Benchmarks — Heatmap</div>', unsafe_allow_html=True)

    bench_heat_df = df_bench.set_index("Model")[visible_bench_cols].astype(float)
    fig_bench_heat = px.imshow(
        bench_heat_df,
        text_auto=".1f",
        color_continuous_scale="Blues",
        zmin=0,
        zmax=100,
        aspect="auto",
        height=max(350, len(bench_heat_df) * 38),
        labels={"color": "Score (%)"},
    )
    fig_bench_heat.update_layout(xaxis_title=None, yaxis_title=None)
    # imshow doesn't support hover_data — embed source per benchmark column in hovertemplate
    fig_bench_heat.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>%{x}: %{z:.1f}%<br>"
            "<span style='color:#888;font-size:0.85em'>"
            "📄 Source: see Benchmark definitions below"
            "</span><extra></extra>"
        )
    )
    pchart(fig_bench_heat)

    with st.expander("Benchmark definitions"):
        st.markdown(
            """
| Benchmark | What it measures | Source |
|---|---|---|
| **HumanEval (%)** | Python function synthesis, pass@1 | [EvalPlus](https://evalplus.github.io/leaderboard.html) |
| **MBPP (%)** | Mostly Basic Python Problems, pass@1 | [EvalPlus](https://evalplus.github.io/leaderboard.html) |
| **SWE-bench (%)** | % of real GitHub issues resolved end-to-end | [SWE-bench](https://www.swebench.com/) |
| **GPQA (%)** | Graduate-level science reasoning | Model cards / papers |
| **MMLU (%)** | Massive Multitask Language Understanding | Model cards / papers |
| **LiveCodeBench (%)** | Contamination-free coding, continuously updated | [LiveCodeBench](https://livecodebench.github.io/) |
            """
        )


# ============================================================
# TAB 4 — Task Advisor
# ============================================================
with tab_task:

    st.markdown('<div class="section-header">Which model is best for my task?</div>', unsafe_allow_html=True)

    task_label = st.selectbox("Choose a task", options=list(TASKS.keys()))

    col_map = {
        "Inline Code Completion":       "Code Completion",
        "Code Generation":              "Code Generation",
        "Complex Reasoning / Planning": "Reasoning",
        "Chat / Q&A":                   "Chat",
        "Debugging":                    "Debugging",
        "Refactoring":                  "Refactoring",
        "Documentation":                "Documentation",
        "Multi-file / Agentic":         "Multi-file",
    }
    score_col = col_map[task_label]

    df_task = df[["Model", "Status", "Tabnine Available", "Provider", "Category",
                  "Deployment", "Thinking", "Context (K)", score_col, "Overall Score"]].copy()
    df_task = df_task.sort_values(score_col, ascending=False).reset_index(drop=True)
    df_task.index += 1

    winner = df_task.iloc[0]
    st.markdown(
        f"""
        <div style="background:#EFF6FF;border:1px solid {TABNINE_BLUE};border-radius:10px;padding:1rem 1.4rem;margin-bottom:1rem;">
          <span class="winner-badge">🏆 Top pick</span>&nbsp;&nbsp;
          <strong style="font-size:1.1rem">{winner['Model']}</strong>
          &nbsp;·&nbsp; Provider: {winner['Provider']}
          &nbsp;·&nbsp; Score: <strong>{winner[score_col]:.1f}/10</strong>
          &nbsp;·&nbsp; Deployment: {winner['Deployment']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    _df_task_bar = with_source(df_task.head(10), score_col)
    fig_task = px.bar(
        _df_task_bar,
        x=score_col,
        y="Model",
        color="Provider",
        color_discrete_map=PROVIDER_COLORS,
        orientation="h",
        text=score_col,
        range_x=[0, 10],
        height=420,
        labels={score_col: f"{task_label} Score"},
        title=f"Top 10 models for: {task_label}",
        hover_data={"📄 Source": True},
    )
    fig_task.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_task.update_layout(plot_bgcolor="white", yaxis_title=None)
    pchart(fig_task)

    st.caption("Hover any cell to see its data source. Dimmed rows = coming-soon models.")
    html_table(
        df_task[["Model", "Status", "Provider", "Deployment", "Thinking", "Context (K)",
                 score_col, "Tabnine Available"]].set_index("Model")
    )

    st.markdown('<div class="section-header">Cloud vs Self-Hosted for this task</div>', unsafe_allow_html=True)

    _df_dep = with_source(df[[score_col, "Deployment"]], score_col)
    fig_dep = px.box(
        _df_dep,
        x="Deployment",
        y=score_col,
        color="Deployment",
        points="all",
        labels={score_col: f"{task_label} Score"},
        hover_data={"📄 Source": True},
    )
    fig_dep.update_layout(showlegend=False, plot_bgcolor="white", height=350)
    pchart(fig_dep)


# ============================================================
# TAB 5 — Hardware
# ============================================================
with tab_hardware:

    st.markdown('<div class="section-header">Self-Hosted Hardware Requirements</div>', unsafe_allow_html=True)

    df_hw = df[df["Deployment"] == "Self Hosted"].copy()

    if df_hw.empty:
        st.info("No self-hosted models match current filters.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            _df_vram = with_source(
                df_hw.dropna(subset=["VRAM (GB)"]).sort_values("VRAM (GB)", ascending=True),
                "VRAM (GB)",
            )
            fig_vram = px.bar(
                _df_vram,
                x="VRAM (GB)",
                y="Model",
                color="Provider",
                color_discrete_map=PROVIDER_COLORS,
                orientation="h",
                text="VRAM (GB)",
                height=350,
                title="Minimum GPU VRAM (GB)",
                hover_data={"📄 Source": True},
            )
            fig_vram.update_traces(texttemplate="%{text} GB", textposition="outside")
            fig_vram.update_layout(plot_bgcolor="white", yaxis_title=None)
            pchart(fig_vram)

        with col_b:
            _df_ctx = with_source(df_hw.sort_values("Context (K)", ascending=True), "Context (K)")
            fig_ctx = px.bar(
                _df_ctx,
                x="Context (K)",
                y="Model",
                color="Provider",
                color_discrete_map=PROVIDER_COLORS,
                orientation="h",
                text="Context (K)",
                height=350,
                title="Context Window (K tokens)",
                hover_data={"📄 Source": True},
            )
            fig_ctx.update_traces(texttemplate="%{text:.0f}K", textposition="outside")
            fig_ctx.update_layout(plot_bgcolor="white", yaxis_title=None)
            pchart(fig_ctx)

        st.markdown('<div class="section-header">Deployment Specs</div>', unsafe_allow_html=True)

        hw_cols = ["Model", "Status", "Provider", "GPU (min)", "VRAM (GB)",
                   "Context (K)", "Thinking", "Tool Calling", "License"]
        st.caption("Hover any cell to see its data source. Dimmed rows = coming-soon models.")
        html_table(df_hw[hw_cols + ["Tabnine Available"]].set_index("Model"))

        st.markdown('<div class="section-header">VRAM vs Context Window (self-hosted)</div>', unsafe_allow_html=True)

        df_hw_clean = df_hw.dropna(subset=["VRAM (GB)"])
        _df_bubble = with_source(df_hw_clean, "VRAM (GB)")
        fig_bubble = px.scatter(
            _df_bubble,
            x="Context (K)",
            y="VRAM (GB)",
            color="Provider",
            color_discrete_map=PROVIDER_COLORS,
            size="Overall Score",
            size_max=40,
            text="Model",
            hover_data={"GPU (min)": True, "📄 Source": True},
            labels={"VRAM (GB)": "Min VRAM (GB)", "Context (K)": "Context Window (K tokens)"},
        )
        fig_bubble.update_traces(textposition="top center", textfont_size=9)
        fig_bubble.update_layout(plot_bgcolor="white", height=420)
        pchart(fig_bubble)

    st.markdown("---")
    st.markdown('<div class="section-header">Cloud Models — Context Window</div>', unsafe_allow_html=True)

    df_cloud = df[df["Deployment"] != "Self Hosted"].copy()
    _df_ctx_cloud = with_source(df_cloud.sort_values("Context (K)", ascending=True), "Context (K)")
    fig_ctx_cloud = px.bar(
        _df_ctx_cloud,
        x="Context (K)",
        y="Model",
        color="Provider",
        color_discrete_map=PROVIDER_COLORS,
        orientation="h",
        text="Context (K)",
        height=max(300, len(df_cloud) * 38),
        title="Cloud Model Context Windows",
        hover_data={"📄 Source": True},
    )
    fig_ctx_cloud.update_traces(texttemplate="%{text:.0f}K", textposition="outside")
    fig_ctx_cloud.update_layout(plot_bgcolor="white", yaxis_title=None)
    pchart(fig_ctx_cloud)


# ============================================================
# TAB 6 — Full Table
# ============================================================
with tab_table:

    st.markdown('<div class="section-header">Complete Model Registry</div>', unsafe_allow_html=True)

    _meta_cols  = ["Model", "Status", "Provider", "Family", "Category", "Deployment", "Plan",
                   "Context (K)", "Thinking", "Tool Calling"]
    _cap_cols   = (ACTIVE_CAP_COLS + ["Overall Score"]) if ACTIVE_CAP_COLS else []
    _bench_cols = [c for c in ["HumanEval (%)", "MBPP (%)", "SWE-bench (%)",
                               "GPQA (%)", "MMLU (%)", "LiveCodeBench (%)"]
                   if c in ACTIVE_BENCH_COLS]
    _hw_cols    = ["GPU (min)", "VRAM (GB)", "License"]
    table_cols  = _meta_cols + _cap_cols + _bench_cols + _hw_cols

    # Legend
    st.markdown(
        '<span class="avail-yes">✅ Available</span> &nbsp; model is live in Tabnine today &nbsp;&nbsp; '
        '<span class="avail-soon">🔜 Coming Soon</span> &nbsp; tracked frontier model, not yet in Tabnine',
        unsafe_allow_html=True,
    )
    st.caption("Hover any cell to see its data source. Dimmed rows = coming-soon models.")

    search = st.text_input("Search model name or provider", "")
    # Include Tabnine Available as a sentinel column for row styling (hidden from display)
    df_tbl = df[table_cols + ["Tabnine Available"]].copy()
    if search:
        mask_s = (
            df_tbl["Model"].str.contains(search, case=False, na=False) |
            df_tbl["Provider"].str.contains(search, case=False, na=False)
        )
        df_tbl = df_tbl[mask_s]

    html_table(df_tbl.set_index("Model"))

    csv = df_tbl.to_csv(index=False).encode()
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="tabnine_models.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center;color:#888;font-size:0.78rem;padding:0.5rem 0 1rem;">
      Built with Streamlit · Data from
      <a href="https://docs.tabnine.com/main/welcome/readme/ai-models" target="_blank">Tabnine Docs</a> ·
      <a href="https://evalplus.github.io/leaderboard.html" target="_blank">EvalPlus</a> ·
      <a href="https://www.swebench.com/" target="_blank">SWE-bench</a> ·
      <a href="https://livecodebench.github.io/" target="_blank">LiveCodeBench</a>
      &nbsp;|&nbsp; Benchmark scores are refreshed every hour.
    </div>
    """,
    unsafe_allow_html=True,
)
