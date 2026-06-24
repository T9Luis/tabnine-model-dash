"""
Utility functions that turn the model registry into pandas DataFrames
ready for Streamlit display and Plotly charts.

All scoring is derived from real public benchmark data only.
The score_* / capability scoring system has been removed.
"""

from __future__ import annotations

import pandas as pd

from data.models import MODELS, TASKS, BENCHMARK_COLUMNS

# ---------------------------------------------------------------------------
# Benchmark columns present in the DataFrame
# ---------------------------------------------------------------------------
BENCH_FIELDS = [
    "bench_humaneval",
    "bench_mbpp",
    "bench_swebench",
    "bench_gpqa",
    "bench_mmlu",
    "bench_livecodebench",
]

# Display-name → raw field mapping (mirrors BENCHMARK_COLUMNS in models.py)
BENCH_DISPLAY = {
    "HumanEval (%)":      "bench_humaneval",
    "MBPP (%)":           "bench_mbpp",
    "SWE-bench (%)":      "bench_swebench",
    "GPQA (%)":           "bench_gpqa",
    "MMLU (%)":           "bench_mmlu",
    "LiveCodeBench (%)":  "bench_livecodebench",
}


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_master_df() -> pd.DataFrame:
    """
    Build the master DataFrame from the static model registry.

    Each row represents one model. Score columns are real benchmark
    percentages; None is preserved as NaN for honest missing-data handling.

    Returns
    -------
    pd.DataFrame
        One row per model; dtypes: str / float / bool / int as appropriate.
    """
    rows = []
    for m in MODELS:
        row = {
            # Identity
            "id":           m.id,
            "Model":        m.display_name,
            "Provider":     m.provider,
            "Family":       m.family,
            "Category":     m.category.title(),
            "Deployment":   m.deployment,
            "Plan":         m.plan,
            "Tags":         m.tags,
            # Technical specs
            "Context (K)":  round(m.context_window / 1000, 0),
            "Thinking":     "✓" if m.thinking_mode else "–",
            "Tool Calling": "✓" if m.tool_calling else "–",
            "GPU (min)":    m.gpu_min or "Cloud",
            "VRAM (GB)":    m.vram_gb,
            "License":      m.license_note,
            # Availability & cost
            "Tabnine Available": m.tabnine_available,
            "Status":       "✅ Available" if m.tabnine_available else "🔜 Coming Soon",
            "Cost Tier":    m.cost_tier,
            "Cost Label":   {1: "💚 Low", 2: "🟡 Medium", 3: "🔴 High"}[m.cost_tier],
            # Real benchmark scores (None → NaN automatically)
            "HumanEval (%)":     m.bench_humaneval,
            "MBPP (%)":          m.bench_mbpp,
            "SWE-bench (%)":     m.bench_swebench,
            "GPQA (%)":          m.bench_gpqa,
            "MMLU (%)":          m.bench_mmlu,
            "LiveCodeBench (%)": m.bench_livecodebench,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

def benchmark_long_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt benchmark score columns into long format for multi-model bar/dot
    comparison charts.

    Rows where Score is NaN are dropped so charts only show real data.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_master_df() (optionally filtered).

    Returns
    -------
    pd.DataFrame
        Columns: Model, Provider, Benchmark, Score.
    """
    bench_cols = list(BENCH_DISPLAY.keys())
    return (
        df[["Model", "Provider"] + bench_cols]
        .melt(id_vars=["Model", "Provider"], var_name="Benchmark", value_name="Score")
        .dropna(subset=["Score"])
    )


def compute_overall_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Overall Score and Efficiency Score columns derived from real
    benchmark data.

    Overall Score
        Mean of all non-NaN benchmark columns for each model, normalised
        to a 0–100 scale (benchmarks are already in %).  Models with no
        benchmark data receive NaN — never a fabricated number.

    Efficiency Score
        Overall Score / cost_weight, where cost_weight reflects the
        relative pricing tier:
            Tier 1 (Low)    → weight 1.0
            Tier 2 (Medium) → weight 1.6
            Tier 3 (High)   → weight 2.5

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_master_df().

    Returns
    -------
    pd.DataFrame
        Copy of df with Overall Score and Efficiency Score appended.
    """
    bench_cols = list(BENCH_DISPLAY.keys())
    cost_weight = {1: 1.0, 2: 1.6, 3: 2.5}

    df = df.copy()

    # Mean across whichever benchmarks are reported; NaN if none reported
    df["Overall Score"] = df[bench_cols].mean(axis=1, skipna=True).round(2)
    # Where every benchmark is NaN, mean returns NaN — keep it honest
    all_missing = df[bench_cols].isna().all(axis=1)
    df.loc[all_missing, "Overall Score"] = pd.NA

    df["Efficiency Score"] = (
        df["Overall Score"] / df["Cost Tier"].map(cost_weight)
    ).round(2)

    return df


def best_model_for_task(df: pd.DataFrame, task: str) -> pd.Series | None:
    """
    Return the row with the highest benchmark score for a given task.

    Parameters
    ----------
    df : pd.DataFrame
        Output of compute_overall_score() or build_master_df().
    task : str
        A key from TASKS (e.g. "Debugging", "Code Generation").

    Returns
    -------
    pd.Series or None
        The row for the top model, or None if no model has data for
        the relevant benchmark.
    """
    from data.models import TASKS  # local import avoids circular deps
    bench_col_raw = TASKS.get(task)
    if bench_col_raw is None:
        return None

    # Map raw field name to display column name
    reverse = {v: k for k, v in BENCH_DISPLAY.items()}
    col = reverse.get(bench_col_raw)
    if col is None or col not in df.columns:
        return None

    valid = df.dropna(subset=[col])
    if valid.empty:
        return None
    return valid.loc[valid[col].idxmax()]
