"""
Unit tests for the model registry and DataFrame utilities.
"""

import pytest
import pandas as pd

from data.models import MODELS, TASKS, TabnineModel
from utils.dataframe import (
    build_master_df,
    compute_overall_score,
    capability_long_df,
    benchmark_long_df,
    best_model_for_task,
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

    def test_score_range(self):
        score_attrs = [
            "score_code_completion", "score_code_generation",
            "score_reasoning", "score_chat", "score_debugging",
            "score_refactoring", "score_documentation", "score_multifile",
        ]
        for m in MODELS:
            for attr in score_attrs:
                val = getattr(m, attr)
                assert 0 <= val <= 10, f"{m.id}.{attr} = {val} out of [0, 10]"

    def test_context_window_positive(self):
        for m in MODELS:
            assert m.context_window > 0, f"{m.id} has non-positive context_window"

    def test_deployment_values(self):
        valid = {"cloud", "self-hosted", "both"}
        for m in MODELS:
            assert m.deployment in valid, f"{m.id} deployment='{m.deployment}' invalid"

    def test_category_values(self):
        valid = {"thinking", "coding", "general"}
        for m in MODELS:
            assert m.category in valid, f"{m.id} category='{m.category}' invalid"

    def test_self_hosted_have_gpu_info(self):
        for m in MODELS:
            if m.deployment == "self-hosted":
                assert m.gpu_min is not None, f"{m.id} self-hosted but gpu_min is None"
                assert m.vram_gb is not None, f"{m.id} self-hosted but vram_gb is None"
                assert m.tensor_parallel is not None

    def test_cloud_models_no_gpu_required(self):
        for m in MODELS:
            if m.deployment == "cloud":
                assert m.gpu_min is None, f"{m.id} cloud model should not have gpu_min"


# ---------------------------------------------------------------------------
# TASKS dict tests
# ---------------------------------------------------------------------------

class TestTasks:

    def test_tasks_not_empty(self):
        assert len(TASKS) > 0

    def test_task_values_are_strings(self):
        for k, v in TASKS.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# DataFrame utility tests
# ---------------------------------------------------------------------------

class TestDataFrame:

    @pytest.fixture(scope="class")
    def df(self):
        return compute_overall_score(build_master_df())

    def test_df_not_empty(self, df):
        assert len(df) == len(MODELS)

    def test_required_columns(self, df):
        required = [
            "Model", "Provider", "Overall Score",
            "Code Completion", "Code Generation", "Reasoning",
            "HumanEval (%)", "SWE-bench (%)",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_overall_score_range(self, df):
        assert df["Overall Score"].between(0, 10).all()

    def test_context_k_positive(self, df):
        assert (df["Context (K)"] > 0).all()

    def test_capability_long_df(self, df):
        long = capability_long_df(df)
        assert "Task" in long.columns
        assert "Score" in long.columns
        assert len(long) > 0

    def test_benchmark_long_df(self, df):
        long = benchmark_long_df(df)
        assert "Benchmark" in long.columns
        assert "Score" in long.columns

    def test_best_model_for_task(self, df):
        best = best_model_for_task(df, "Code Generation")
        assert isinstance(best, pd.Series)
        assert best["Code Generation"] == df["Code Generation"].max()

    def test_live_score_override(self):
        fake_live = {
            "gpt-4.1": {
                "humaneval": 95.5,
                "swebench": 60.0,
                "livecodebench": 70.0,
            }
        }
        df_live = build_master_df(live_scores=fake_live)
        row = df_live[df_live["id"] == "gpt-4.1"].iloc[0]
        assert row["HumanEval (%)"] == 95.5
        assert row["SWE-bench (%)"] == 60.0
        assert row["LiveCodeBench (%)"] == 70.0
