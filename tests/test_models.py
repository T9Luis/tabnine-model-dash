"""
Unit tests for the model registry and DataFrame utilities.

All tests use only real benchmark data. The score_* capability system
has been removed; tests reflect the benchmark-only data model.
"""

import pytest
import pandas as pd

from data.models import MODELS, TASKS, TabnineModel, BENCHMARK_COLUMNS
from utils.dataframe import (
    build_master_df,
    compute_overall_score,
    benchmark_long_df,
    best_model_for_task,
    BENCH_DISPLAY,
)


# ---------------------------------------------------------------------------
# Model registry tests
# ---------------------------------------------------------------------------

class TestModelRegistry:

    def test_models_not_empty(self):
        assert len(MODELS) > 0

    def test_all_models_are_dataclass(self):
        for m in MODELS:
            assert isinstance(m, TabnineModel)

    def test_unique_ids(self):
        ids = [m.id for m in MODELS]
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"

    def test_benchmark_values_in_range_or_none(self):
        """Benchmark percentages must be in [0, 100] when reported."""
        bench_attrs = [
            "bench_humaneval", "bench_mbpp", "bench_swebench",
            "bench_gpqa", "bench_mmlu", "bench_livecodebench",
        ]
        for m in MODELS:
            for attr in bench_attrs:
                val = getattr(m, attr)
                if val is not None:
                    assert 0 <= val <= 100, (
                        f"{m.id}.{attr} = {val} is outside [0, 100]"
                    )

    def test_context_window_positive(self):
        for m in MODELS:
            assert m.context_window > 0, f"{m.id} has non-positive context_window"

    def test_deployment_values(self):
        valid = {"Cloud", "Self Hosted"}
        for m in MODELS:
            assert m.deployment in valid, (
                f"{m.id} deployment='{m.deployment}' not in {valid}"
            )

    def test_category_values(self):
        valid = {"thinking", "coding", "general"}
        for m in MODELS:
            assert m.category in valid, (
                f"{m.id} category='{m.category}' not in {valid}"
            )

    def test_self_hosted_have_gpu_info(self):
        for m in MODELS:
            if m.deployment == "Self Hosted":
                assert m.gpu_min is not None, (
                    f"{m.id} self-hosted but gpu_min is None"
                )
                assert m.vram_gb is not None, (
                    f"{m.id} self-hosted but vram_gb is None"
                )
                assert m.tensor_parallel is not None, (
                    f"{m.id} self-hosted but tensor_parallel is None"
                )

    def test_cloud_models_no_gpu_required(self):
        for m in MODELS:
            if m.deployment == "Cloud":
                assert m.gpu_min is None, (
                    f"{m.id} cloud model should not have gpu_min"
                )

    def test_cost_tier_values(self):
        for m in MODELS:
            assert m.cost_tier in (1, 2, 3), (
                f"{m.id} cost_tier={m.cost_tier} must be 1, 2, or 3"
            )

    def test_tabnine_available_is_bool(self):
        for m in MODELS:
            assert isinstance(m.tabnine_available, bool), (
                f"{m.id} tabnine_available must be bool"
            )


# ---------------------------------------------------------------------------
# TASKS dict tests
# ---------------------------------------------------------------------------

class TestTasks:

    def test_tasks_not_empty(self):
        assert len(TASKS) > 0

    def test_task_values_reference_valid_bench_fields(self):
        """Each TASKS value must map to a known benchmark field name."""
        valid_bench = {v for v in BENCH_DISPLAY.values()}
        for task, bench_raw in TASKS.items():
            assert bench_raw in valid_bench, (
                f"TASKS['{task}'] = '{bench_raw}' is not a known benchmark field"
            )


# ---------------------------------------------------------------------------
# DataFrame utility tests
# ---------------------------------------------------------------------------

class TestDataFrame:

    @pytest.fixture(scope="class")
    def df(self):
        return compute_overall_score(build_master_df())

    def test_df_row_count(self, df):
        assert len(df) == len(MODELS)

    def test_required_columns_present(self, df):
        required = [
            "Model", "Provider", "Overall Score", "Efficiency Score",
            "HumanEval (%)", "SWE-bench (%)", "GPQA (%)", "MMLU (%)",
            "Cost Tier", "Status", "Deployment",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_score_star_columns(self, df):
        """score_* columns must not appear in the DataFrame."""
        stale = [c for c in df.columns if c.startswith("score_")]
        assert stale == [], f"Stale score_* columns found: {stale}"

    def test_overall_score_range_or_nan(self, df):
        """Overall Score must be in [0, 100] where reported; NaN otherwise."""
        valid = df["Overall Score"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), (
            "Overall Score values outside [0, 100] found"
        )

    def test_overall_score_nan_where_no_benchmarks(self, df):
        """Models with no benchmark data must have NaN Overall Score."""
        bench_cols = list(BENCH_DISPLAY.keys())
        no_data = df[df[bench_cols].isna().all(axis=1)]
        assert no_data["Overall Score"].isna().all(), (
            "Expected NaN Overall Score for models with no benchmark data"
        )

    def test_context_k_positive(self, df):
        assert (df["Context (K)"] > 0).all()

    def test_benchmark_long_df_no_nan(self, df):
        long = benchmark_long_df(df)
        assert "Benchmark" in long.columns
        assert "Score" in long.columns
        # benchmark_long_df drops NaN rows
        assert long["Score"].notna().all(), (
            "benchmark_long_df should have dropped all NaN score rows"
        )

    def test_best_model_for_task_returns_series_or_none(self, df):
        result = best_model_for_task(df, "Debugging")
        # Should return a Series for Debugging (maps to SWE-bench, which has data)
        assert result is None or isinstance(result, pd.Series)

    def test_best_model_for_unknown_task_returns_none(self, df):
        result = best_model_for_task(df, "nonexistent_task_xyz")
        assert result is None
