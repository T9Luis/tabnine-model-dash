# v3 — benchmark-only, real data
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from data.models import MODELS, TASKS
from utils.dataframe import build_master_df, benchmark_long_df, compute_overall_score, BENCH_DISPLAY

st.set_page_config(page_title="Tabnine Model Dashboard", page_icon="⚡", layout="wide")

TABNINE_BLUE  = "#1F46C1"
TABNINE_DARK  = "#131A3A"
ACCENT_GREEN  = "#00C950"
ACCENT_ORANGE = "#FE9A00"

PROVIDER_COLORS = {
    "OpenAI": "#10A37F", "Anthropic": "#D4A96A", "Google": "#4285F4",
    "Mistral": "#FF7000", "MiniMax": "#7C3AED", "Poolside": "#0EA5E9",
    "Zhipu AI": "#E11D48", "Alibaba": "#FF6A00", "Meta": "#0866FF", "Tabnine": TABNINE_BLUE,
}

_CW = {1: 1.0, 2: 1.6, 3: 2.5}
_BENCH_RAW_TO_DISPLAY = {v: k for k, v in BENCH_DISPLAY.items()}
BENCH_COLS = list(BENCH_DISPLAY.keys())

df_raw    = build_master_df()
df_scored = compute_overall_score(df_raw)

st.markdown(
    f"""
    <style>
      .hero {{ background: linear-gradient(135deg, {TABNINE_DARK} 0%, {TABNINE_BLUE} 100%); color:white; padding:1.8rem 2.2rem 1.4rem; border-radius:12px; margin-bottom:1.4rem; }}
      .hero h1 {{ margin:0; font-size:2rem; font-weight:800; }}
      .hero p  {{ margin:0.3rem 0 0; opacity:0.85; font-size:0.95rem; }}
      .kpi {{ background:#f8faff; border:1px solid #dce8ff; border-radius:10px; padding:0.9rem 1rem; text-align:center; }}
      .kpi h2 {{ color:{TABNINE_BLUE}; font-size:1.8rem; margin:0; }}
      .kpi p  {{ color:#555; font-size:0.8rem; margin:0.15rem 0 0; }}
      .winner-card {{ background:#EFF6FF; border:1.5px solid {TABNINE_BLUE}; border-radius:10px; padding:1rem 1.4rem; margin-bottom:0.6rem; }}
      .winner-card h3 {{ margin:0 0 0.4rem; font-size:1.15rem; color:{TABNINE_DARK}; }}
      .winner-card p  {{ margin:0.2rem 0; font-size:0.88rem; color:#333; }}
      .runner-card {{ background:#F0FDF4; border:1px solid #86EFAC; border-radius:10px; padding:1rem 1.4rem; margin-bottom:0.6rem; }}
      .runner-card h3 {{ margin:0 0 0.4rem; font-size:1rem; color:#166534; }}
      .runner-card p  {{ margin:0.2rem 0; font-size:0.85rem; color:#333; }}
      .section-hdr {{ border-left:4px solid {TABNINE_BLUE}; padding-left:0.65rem; margin:1.4rem 0 0.65rem; font-weight:700; font-size:1.05rem; color:{TABNINE_DARK}; }}
      .badge-avail {{ background:{ACCENT_GREEN}; color:white; padding:0.15rem 0.55rem; border-radius:999px; font-size:0.73rem; font-weight:600; }}
      .badge-soon  {{ background:{ACCENT_ORANGE}; color:white; padding:0.15rem 0.55rem; border-radius:999px; font-size:0.73rem; font-weight:600; }}
      .src-table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
      .src-table th {{ background:{TABNINE_DARK}; color:white; padding:0.42rem 0.65rem; text-align:left; font-weight:600; white-space:nowrap; }}
      .src-table tr:nth-child(even) {{ background:#f4f7ff; }}
      .src-table tr:hover td {{ background:#dce8ff !important; }}
      .src-table td {{ padding:0.35rem 0.65rem; border-bottom:1px solid #e5eaf5; cursor:default; }}
      .src-table tr.cs td {{ opacity:0.52; font-style:italic; }}
      .src-table tr.cs:hover td {{ opacity:0.8 !important; background:#fff8e8 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def pchart(fig):
    st.plotly_chart(fig, use_container_width=True)


def html_table(df_arg: pd.DataFrame, max_rows: int = 300) -> None:
    import html as _html
    display_df = df_arg.head(max_rows).copy()
    if "Tabnine Available" in display_df.columns:
        avail_map = display_df["Tabnine Available"].to_dict()
        display_df = display_df.drop(columns=["Tabnine Available"])
    else:
        avail_map = {}
    def _fmt(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "—"
        return _html.escape(str(val))
    headers = "".join(f"<th>{_html.escape(str(c))}</th>" for c in display_df.columns)
    rows_html = []
    for idx, row in display_df.iterrows():
        row_class = "" if avail_map.get(idx, True) else ' class="cs"'
        cells = "".join(f"<td>{_fmt(row[c])}</td>" for c in display_df.columns)
        rows_html.append(f"<tr{row_class}>{cells}</tr>")
    st.markdown(
        f'<div style="overflow-x:auto;max-height:500px;overflow-y:auto">'
        f'<table class="src-table"><thead><tr>{headers}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def _safe_idxmax(series: pd.Series):
    valid = series.dropna()
    return valid.idxmax() if not valid.empty else None


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>⚡ Tabnine Model Dashboard</h1>
      <p>Find the right AI model for every task — ranked by effectiveness, performance, and cost.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://www.tabnine.com/wp-content/uploads/2023/05/tabnine-logo.svg",
        width=150,
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
    thinking_only = st.checkbox("Thinking models only", value=False)
    show_upcoming = st.checkbox("Include upcoming models", value=True)

    st.markdown("---")
    st.markdown("### Compare Models")
    _cmp_pool = df_scored if show_upcoming else df_scored[df_scored["Tabnine Available"] == True]
    all_names = sorted(_cmp_pool["Model"].tolist())
    compare_models = st.multiselect(
        "Select models",
        options=all_names,
        default=all_names[:4],
    )

    st.markdown("---")
    st.caption(
        "Sources: [Tabnine Docs](https://docs.tabnine.com/main/welcome/readme/ai-models) · "
        "[EvalPlus](https://evalplus.github.io/leaderboard.html) · "
        "[SWE-bench](https://www.swebench.com/) · "
        "[LiveCodeBench](https://livecodebench.github.io/)"
    )

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
# KPI strip
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    n_avail = int((df["Tabnine Available"] == True).sum())
    st.markdown(
        f'<div class="kpi"><h2>{n_avail}</h2><p>Available in Tabnine</p></div>',
        unsafe_allow_html=True,
    )
with k2:
    avail_df = df[df["Tabnine Available"] == True]
    _idx = _safe_idxmax(avail_df["Overall Score"])
    best_name = avail_df.loc[_idx, "Model"] if _idx is not None else "—"
    st.markdown(
        f'<div class="kpi"><h2 style="font-size:1rem;padding-top:0.3rem">{best_name}</h2>'
        f'<p>Top overall score</p></div>',
        unsafe_allow_html=True,
    )
with k3:
    _idx2 = _safe_idxmax(avail_df["Efficiency Score"])
    best_val_name = avail_df.loc[_idx2, "Model"] if _idx2 is not None else "—"
    st.markdown(
        f'<div class="kpi"><h2 style="font-size:1rem;padding-top:0.3rem">{best_val_name}</h2>'
        f'<p>Best effectiveness</p></div>',
        unsafe_allow_html=True,
    )
with k4:
    n_self = int((df["Deployment"] == "Self Hosted").sum())
    st.markdown(
        f'<div class="kpi"><h2>{n_self}</h2><p>Self-hosted models</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_task, tab_overview, tab_compare, tab_benchmarks, tab_hardware, tab_table = st.tabs([
    "🎯 Task Advisor",
    "📊 Overview",
    "⚖️ Compare",
    "📈 Benchmarks",
    "🖥️ Hardware",
    "📋 Full Table",
])

# ============================================================
# TAB 1 — Task Advisor
# ============================================================
with tab_task:

    st.markdown(
        f"""
        <div style="margin-bottom:0.3rem">
          <span style="font-size:0.78rem;font-weight:600;text-transform:uppercase;
                       letter-spacing:0.07em;color:{TABNINE_BLUE}">
            Step 1 — Choose your task
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    task_label = st.selectbox(
        "What do you need the model to do?",
        options=list(TASKS.keys()),
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="margin-bottom:0.3rem">
          <span style="font-size:0.78rem;font-weight:600;text-transform:uppercase;
                       letter-spacing:0.07em;color:{TABNINE_BLUE}">
            Step 2 — Optimise for
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "Optimise for",
        options=["💚 Effectiveness (score / cost)", "⚡ Raw benchmark score"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    perf_mode = mode.startswith("⚡")
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # Map task → benchmark display-column
    bench_raw = TASKS[task_label]                           # e.g. "bench_swebench"
    bench_col = _BENCH_RAW_TO_DISPLAY.get(bench_raw, "")   # e.g. "SWE-bench (%)"

    # Build per-task working frame — only models that have a score for this benchmark
    _base_cols = ["Model", "Status", "Tabnine Available", "Provider", "Category",
                  "Deployment", "Thinking", "Context (K)", "Cost Tier", "Cost Label"]

    if bench_col and bench_col in df.columns:
        df_task = df[_base_cols + [bench_col, "Overall Score"]].copy()
        df_task[bench_col] = pd.to_numeric(df_task[bench_col], errors="coerce")
        df_task["Task Efficiency"] = (
            df_task[bench_col] / df_task["Cost Tier"].map(_CW)
        ).round(2)
        sort_col   = bench_col if perf_mode else "Task Efficiency"
        bar_label  = (f"{bench_col}" if perf_mode
                      else f"Effectiveness ({bench_col} ÷ cost weight)")
        bar_title  = (f"All models — {task_label} ({bench_col})"
                      if perf_mode else f"All models — {task_label} effectiveness")
    else:
        df_task    = df[_base_cols + ["Overall Score"]].copy()
        df_task["Task Efficiency"] = (
            df_task["Overall Score"] / df_task["Cost Tier"].map(_CW)
        ).round(2)
        bench_col  = "Overall Score"
        sort_col   = bench_col if perf_mode else "Task Efficiency"
        bar_label  = "Overall Score (%)" if perf_mode else "Effectiveness (overall ÷ cost weight)"
        bar_title  = f"All models — {task_label}"
        st.info(
            f"No dedicated benchmark is mapped for **{task_label}** yet. "
            f"Showing Overall Score (mean of available benchmarks) as a proxy."
        )

    df_task = df_task.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    df_task.index += 1

    df_avail    = df_task[df_task["Tabnine Available"] == True]
    df_upcoming = df_task[df_task["Tabnine Available"] == False]

    if df_avail.empty:
        st.warning("No available Tabnine models match the current sidebar filters.")
    else:
        # Winner / runner-up — only from models that actually have a score
        df_avail_scored = df_avail.dropna(subset=[sort_col])
        winner    = df_avail_scored.iloc[0] if not df_avail_scored.empty else None
        runner_up = df_avail_scored.iloc[1] if len(df_avail_scored) > 1 else None

        card_l, card_r = st.columns([1, 1])

        with card_l:
            if winner is not None:
                score_val = winner[sort_col]
                if perf_mode:
                    score_line = f"Score: <strong>{score_val:.1f}%</strong>"
                    why_line   = (f"Highest {bench_col} among {len(df_avail_scored)} "
                                  f"models with reported data.")
                else:
                    score_line = (
                        f"Benchmark: <strong>{winner[bench_col]:.1f}%</strong> · "
                        f"Cost: <strong>{winner['Cost Label']}</strong> · "
                        f"Effectiveness: <strong>{winner['Task Efficiency']:.2f}</strong>"
                    )
                    why_line = (
                        f"Best score-to-cost ratio for <em>{task_label}</em> "
                        f"across {len(df_avail_scored)} models with reported data."
                    )

                extras = []
                if winner["Thinking"] == "✓":
                    extras.append("Extended thinking mode — handles multi-step reasoning.")
                ctx_k = winner["Context (K)"]
                if pd.notna(ctx_k) and float(ctx_k) >= 100:
                    extras.append(f"{int(ctx_k)}K token context window.")
                if winner["Deployment"] == "Cloud":
                    extras.append("Cloud-hosted — no GPU required.")
                else:
                    extras.append("Self-hosted — on-premises / air-gapped friendly.")
                extra_html = "".join(f"<p>{e}</p>" for e in extras)

                st.markdown(
                    f"""
                    <div class="winner-card">
                      <h3>🥇 {winner['Model']}</h3>
                      <p><strong>{winner['Provider']}</strong> · {winner['Deployment']}</p>
                      <p>{score_line}</p>
                      <p style="color:#555;font-size:0.83rem">{why_line}</p>
                      {extra_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No benchmark data available for the currently filtered models.")

        with card_r:
            if runner_up is not None:
                if perf_mode:
                    ru_score_line = f"Score: <strong>{runner_up[bench_col]:.1f}%</strong>"
                    gap_line = f"Gap to winner: {winner[bench_col] - runner_up[bench_col]:.1f} pp"
                else:
                    ru_score_line = (
                        f"Benchmark: <strong>{runner_up[bench_col]:.1f}%</strong> · "
                        f"Cost: <strong>{runner_up['Cost Label']}</strong> · "
                        f"Effectiveness: <strong>{runner_up['Task Efficiency']:.2f}</strong>"
                    )
                    gap_line = (
                        f"Effectiveness gap to winner: "
                        f"{winner['Task Efficiency'] - runner_up['Task Efficiency']:.2f}"
                    )
                st.markdown(
                    f"""
                    <div class="runner-card">
                      <h3>🥈 Runner-up: {runner_up['Model']}</h3>
                      <p><strong>{runner_up['Provider']}</strong> · {runner_up['Deployment']}</p>
                      <p>{ru_score_line}</p>
                      <p style="color:#555;font-size:0.83rem">{gap_line}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Coming-soon alert
        if not df_upcoming.empty and winner is not None:
            ahead = df_upcoming.dropna(subset=[sort_col])
            ahead = ahead[ahead[sort_col] > winner[sort_col]]
            if not ahead.empty:
                metric_lbl = "raw score" if perf_mode else "effectiveness"
                names = ", ".join(
                    f"{r['Model']} ({r[sort_col]:.2f})" for _, r in ahead.iterrows()
                )
                st.info(
                    f"**Coming soon:** {names} rank higher by {metric_lbl} "
                    f"but are not yet available in Tabnine."
                )

    # Bar chart
    df_task_chart = df_task.dropna(subset=[sort_col]).sort_values(sort_col, ascending=True)
    _task_order   = df_task_chart["Model"].tolist()

    if not df_task_chart.empty:
        fig_task = px.bar(
            df_task_chart,
            x=sort_col,
            y="Model",
            color="Provider",
            color_discrete_map=PROVIDER_COLORS,
            orientation="h",
            text=sort_col,
            height=max(420, len(df_task_chart) * 36),
            labels={sort_col: bar_label},
            title=bar_title,
            hover_data={"Status": True, "Cost Label": True},
            pattern_shape="Status",
            pattern_shape_map={"✅ Available": "", "🔜 Coming Soon": "/"},
            category_orders={"Model": _task_order},
        )
        fig_task.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_task.update_layout(plot_bgcolor="white", yaxis_title=None)
        pchart(fig_task)

    # Lean table
    display_cols = ["Model", "Status", "Provider", "Deployment",
                    "Thinking", "Context (K)", "Cost Label", bench_col, "Task Efficiency"]
    available_cols = [c for c in display_cols if c in df_task.columns]
    html_table(
        df_task[available_cols + ["Tabnine Available"]].set_index("Model")
    )
    st.caption(
        f"Sorted by **{sort_col}**. — = score not publicly reported for this model. "
        f"Benchmark: [{bench_col}]"
    )


# ============================================================
# TAB 2 — Overview
# ============================================================
with tab_overview:

    st.markdown('<div class="section-hdr">Overall Model Scores</div>', unsafe_allow_html=True)
    st.caption(
        "Overall Score = mean of all publicly reported benchmark scores (%). "
        "Models with no benchmark data show — in the table."
    )

    df_bar = df.dropna(subset=["Overall Score"]).sort_values("Overall Score", ascending=True)
    _bar_order = df_bar["Model"].tolist()

    fig_bar = px.bar(
        df_bar,
        x="Overall Score",
        y="Model",
        color="Provider",
        color_discrete_map=PROVIDER_COLORS,
        orientation="h",
        text="Overall Score",
        range_x=[0, 100],
        height=max(400, len(df_bar) * 36),
        labels={"Overall Score": "Avg Benchmark Score (%)"},
        hover_data={"Status": True, "Cost Label": True},
        pattern_shape="Status",
        pattern_shape_map={"✅ Available": "", "🔜 Coming Soon": "/"},
        category_orders={"Model": _bar_order},
    )
    fig_bar.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        marker_opacity=df_bar["Tabnine Available"].map({True: 1.0, False: 0.4}).tolist(),
    )
    fig_bar.update_layout(plot_bgcolor="white", yaxis_title=None)
    pchart(fig_bar)

    st.markdown('<div class="section-hdr">Effectiveness vs Performance</div>', unsafe_allow_html=True)
    st.caption(
        "X-axis = Overall Score (mean benchmark %). Y-axis = Effectiveness Score (score ÷ cost weight). "
        "Models in the top-right are both high-performing and cost-effective. "
        "Hatched markers = coming soon."
    )

    df_eff = df.dropna(subset=["Overall Score", "Efficiency Score"])
    fig_eff = px.scatter(
        df_eff,
        x="Overall Score",
        y="Efficiency Score",
        color="Provider",
        color_discrete_map=PROVIDER_COLORS,
        symbol="Status",
        symbol_map={"✅ Available": "circle", "🔜 Coming Soon": "diamond-open"},
        size="Context (K)",
        size_max=28,
        hover_name="Model",
        text="Model",
        hover_data={"Deployment": True, "Cost Label": True, "Thinking": True, "Status": True},
        labels={"Overall Score": "Performance (% avg benchmark)", "Efficiency Score": "Effectiveness"},
    )
    fig_eff.update_traces(textposition="top center", textfont_size=9)
    fig_eff.update_layout(plot_bgcolor="white", height=480)
    pchart(fig_eff)


# ============================================================
# TAB 3 — Compare
# ============================================================
with tab_compare:

    df_cmp = df[df["Model"].isin(compare_models)].copy()

    if df_cmp.empty:
        st.info("Select models in the sidebar to compare.")
    else:
        st.markdown('<div class="section-hdr">Benchmark Comparison</div>', unsafe_allow_html=True)
        st.caption(
            "Only benchmarks where at least one selected model has a reported score are shown. "
            "— = not publicly reported."
        )

        long = benchmark_long_df(df_cmp)
        if long.empty:
            st.info("None of the selected models have publicly reported benchmark scores.")
        else:
            _bench_order = list(BENCH_DISPLAY.keys())
            fig_grp = px.bar(
                long,
                x="Benchmark",
                y="Score",
                color="Model",
                barmode="group",
                range_y=[0, 100],
                height=440,
                labels={"Score": "Score (%)"},
                title="Benchmark scores — selected models",
                category_orders={"Benchmark": _bench_order},
            )
            fig_grp.update_layout(plot_bgcolor="white", xaxis_title=None)
            pchart(fig_grp)

        st.markdown('<div class="section-hdr">Summary</div>', unsafe_allow_html=True)
        show_cols = (
            ["Model", "Status", "Provider", "Deployment", "Cost Label",
             "Context (K)", "Thinking"]
            + BENCH_COLS
            + ["Overall Score", "Efficiency Score"]
        )
        html_table(df_cmp[[c for c in show_cols if c in df_cmp.columns] + ["Tabnine Available"]].set_index("Model"))


# ============================================================
# TAB 4 — Benchmarks
# ============================================================
with tab_benchmarks:

    st.markdown('<div class="section-hdr">Official Public Benchmarks</div>', unsafe_allow_html=True)
    st.caption(
        "Sources: [EvalPlus](https://evalplus.github.io/leaderboard.html) · "
        "[SWE-bench](https://www.swebench.com/) · "
        "[LiveCodeBench](https://livecodebench.github.io/) · "
        "Published model cards (GPQA / MMLU). Scores are manually curated."
    )

    bench_sel = st.selectbox("Select benchmark", options=BENCH_COLS, index=0)

    df_bench = df[["Model", "Provider", "Status", "Tabnine Available"] + BENCH_COLS].copy()
    df_b     = df_bench.dropna(subset=[bench_sel]).sort_values(bench_sel, ascending=True)

    if df_b.empty:
        st.warning(f"No models have a reported {bench_sel} score.")
    else:
        _bench_order = df_b["Model"].tolist()
        fig_bench = px.bar(
            df_b,
            x=bench_sel,
            y="Model",
            color="Provider",
            color_discrete_map=PROVIDER_COLORS,
            orientation="h",
            text=bench_sel,
            height=max(320, len(df_b) * 40),
            range_x=[0, 100],
            labels={bench_sel: f"{bench_sel}"},
            hover_data={"Status": True},
            pattern_shape="Status",
            pattern_shape_map={"✅ Available": "", "🔜 Coming Soon": "/"},
            category_orders={"Model": _bench_order},
        )
        fig_bench.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bench.update_layout(plot_bgcolor="white", yaxis_title=None)
        pchart(fig_bench)

    with st.expander("Benchmark definitions"):
        st.markdown(
            "| Benchmark | What it measures | Source |\n"
            "|---|---|---|\n"
            "| **HumanEval (%)** | Python function synthesis, pass@1 | [EvalPlus](https://evalplus.github.io/leaderboard.html) |\n"
            "| **MBPP (%)** | Mostly Basic Python Problems, pass@1 | [EvalPlus](https://evalplus.github.io/leaderboard.html) |\n"
            "| **SWE-bench (%)** | Real GitHub issues resolved end-to-end | [SWE-bench](https://www.swebench.com/) |\n"
            "| **GPQA (%)** | Graduate-level science reasoning | Model cards / papers |\n"
            "| **MMLU (%)** | Massive Multitask Language Understanding | Model cards / papers |\n"
            "| **LiveCodeBench (%)** | Contamination-free coding benchmark | [LiveCodeBench](https://livecodebench.github.io/) |\n"
        )


# ============================================================
# TAB 5 — Hardware
# ============================================================
with tab_hardware:

    df_hw = df[df["Deployment"] == "Self Hosted"].copy()

    if df_hw.empty:
        st.info("No self-hosted models match the current filters.")
    else:
        st.markdown('<div class="section-hdr">Self-Hosted Hardware Requirements</div>', unsafe_allow_html=True)

        hw_left, hw_right = st.columns(2)

        with hw_left:
            df_vram = df_hw.dropna(subset=["VRAM (GB)"]).sort_values("VRAM (GB)", ascending=True)
            fig_vram = px.bar(
                df_vram,
                x="VRAM (GB)",
                y="Model",
                color="Provider",
                color_discrete_map=PROVIDER_COLORS,
                orientation="h",
                text="VRAM (GB)",
                height=max(300, len(df_vram) * 40),
                title="Min GPU VRAM (GB)",
                category_orders={"Model": df_vram["Model"].tolist()},
            )
            fig_vram.update_traces(texttemplate="%{text} GB", textposition="outside")
            fig_vram.update_layout(plot_bgcolor="white", yaxis_title=None)
            pchart(fig_vram)

        with hw_right:
            df_ctx = df_hw.sort_values("Context (K)", ascending=True)
            fig_ctx = px.bar(
                df_ctx,
                x="Context (K)",
                y="Model",
                color="Provider",
                color_discrete_map=PROVIDER_COLORS,
                orientation="h",
                text="Context (K)",
                height=max(300, len(df_ctx) * 40),
                title="Context Window (K tokens)",
                category_orders={"Model": df_ctx["Model"].tolist()},
            )
            fig_ctx.update_traces(texttemplate="%{text:.0f}K", textposition="outside")
            fig_ctx.update_layout(plot_bgcolor="white", yaxis_title=None)
            pchart(fig_ctx)

        st.markdown('<div class="section-hdr">Specs Table</div>', unsafe_allow_html=True)
        hw_cols = ["Model", "Status", "Provider", "GPU (min)", "VRAM (GB)",
                   "Context (K)", "Thinking", "License"]
        html_table(df_hw[[c for c in hw_cols if c in df_hw.columns] + ["Tabnine Available"]].set_index("Model"))


# ============================================================
# TAB 6 — Full Table
# ============================================================
with tab_table:

    st.markdown('<div class="section-hdr">Complete Model Registry</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="badge-avail">✅ Available</span>&nbsp; live in Tabnine today &nbsp;&nbsp;'
        '<span class="badge-soon">🔜 Coming Soon</span>&nbsp; tracked but not yet in Tabnine',
        unsafe_allow_html=True,
    )

    search = st.text_input("Search by model name or provider", "")

    tbl_cols = [
        "Model", "Status", "Provider", "Family", "Category", "Deployment", "Plan",
        "Context (K)", "Thinking", "Tool Calling", "Cost Label",
        "Overall Score", "Efficiency Score",
    ] + BENCH_COLS + [
        "GPU (min)", "VRAM (GB)", "License",
    ]

    df_tbl = df[[c for c in tbl_cols if c in df.columns] + ["Tabnine Available"]].copy()
    if search:
        mask_s = (
            df_tbl["Model"].str.contains(search, case=False, na=False) |
            df_tbl["Provider"].str.contains(search, case=False, na=False)
        )
        df_tbl = df_tbl[mask_s]

    html_table(df_tbl.set_index("Model"))

    st.download_button(
        label="Download as CSV",
        data=df_tbl.to_csv(index=False).encode(),
        file_name="tabnine_models.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#888;font-size:0.78rem;padding:0.4rem 0 1rem;">'
    'Built with Streamlit · '
    '<a href="https://docs.tabnine.com/main/welcome/readme/ai-models" target="_blank">Tabnine Docs</a> · '
    '<a href="https://evalplus.github.io/leaderboard.html" target="_blank">EvalPlus</a> · '
    '<a href="https://www.swebench.com/" target="_blank">SWE-bench</a> · '
    '<a href="https://livecodebench.github.io/" target="_blank">LiveCodeBench</a>'
    '</div>',
    unsafe_allow_html=True,
)
