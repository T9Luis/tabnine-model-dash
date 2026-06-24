"""
Utility functions that turn the model registry + live scores into
pandas DataFrames ready for Streamlit display and Plotly charts.
"""

from __future__ import annotations

import pandas as pd

from data.models import MODELS, TabnineModel, TASKS, BENCHMARK_COLUMNS

# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

SCORE_FIELDS = [
    "score_code_completion",
    "score_code_generation",
    "score_reasoning",
    "score_chat",
    "score_debugging",
    "score_refactoring",
    "score_documentation",
    "score_multifile",
]

BENCH_FIELDS = [
    "bench_humaneval",
    "bench_mbpp",
    "bench_swebench",
    "bench_gpqa",
    "bench_mmlu",
]


def build_master_df() -> pd.DataFrame:
    """
    Build the master DataFrame from the static model registry.

    Returns:
        DataFrame with one row per model.
    """
    rows = []
    for m in MODELS:
        row = {
            "id":           m.id,
            "Model":        m.display_name,
            "Provider":     m.provider,
            "Family":       m.family,
            "Category":     m.category.title(),
            "Deployment":   m.deployment.replace("-", " ").title(),
            "Plan":         m.plan,
            "Context (K)":  round(m.context_window / 1000, 0),
            "Thinking":     "✓" if m.thinking_mode else "–",
            "Tool Calling": "✓" if m.tool_calling else "–",
            "GPU (min)":         m.gpu_min or "Cloud",
            "VRAM (GB)":         m.vram_gb,
            "License":           m.license_note,
            "Tabnine Available": m.tabnine_available,
            "Status":            "✅ Available" if m.tabnine_available else "🔜 Coming Soon",
            # capability scores
            "Code Completion":  m.score_code_completion,
            "Code Generation":  m.score_code_generation,
            "Reasoning":        m.score_reasoning,
            "Chat":             m.score_chat,
            "Debugging":        m.score_debugging,
            "Refactoring":      m.score_refactoring,
            "Documentation":    m.score_documentation,
            "Multi-file":       m.score_multifile,
            # benchmarks (static seed)
            "HumanEval (%)":     m.bench_humaneval,
            "MBPP (%)":          m.bench_mbpp,
            "SWE-bench (%)":     m.bench_swebench,
            "GPQA (%)":          m.bench_gpqa,
            "MMLU (%)":          m.bench_mmlu,
            "LiveCodeBench (%)": m.bench_livecodebench,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

def capability_long_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt capability score columns into long format for multi-model radar/bar
    comparison charts.
    """
    cap_cols = [
        "Code Completion", "Code Generation", "Reasoning", "Chat",
        "Debugging", "Refactoring", "Documentation", "Multi-file",
    ]
    return df[["Model", "Provider"] + cap_cols].melt(
        id_vars=["Model", "Provider"],
        var_name="Task",
        value_name="Score",
    )


def benchmark_long_df(df: pd.DataFrame) -> pd.DataFrame:
    """Melt benchmark columns into long format."""
    bench_cols = ["HumanEval (%)", "MBPP (%)", "SWE-bench (%)", "GPQA (%)", "MMLU (%)", "LiveCodeBench (%)"]
    return df[["Model", "Provider"] + bench_cols].melt(
        id_vars=["Model", "Provider"],
        var_name="Benchmark",
        value_name="Score",
    ).dropna(subset=["Score"])


def best_model_for_task(df: pd.DataFrame, task_col: str) -> pd.Series:
    """Return the row with the highest score for a given task column."""
    return df.loc[df[task_col].idxmax()]


def compute_overall_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add a weighted overall score column (equal weights across all tasks)."""
    cap_cols = [
        "Code Completion", "Code Generation", "Reasoning", "Chat",
        "Debugging", "Refactoring", "Documentation", "Multi-file",
    ]
    df = df.copy()
    df["Overall Score"] = df[cap_cols].mean(axis=1).round(2)
    return df
