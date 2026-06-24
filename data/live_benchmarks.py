"""
Live benchmark fetcher.

Pulls the latest scores from publicly available leaderboards and APIs:
  - Hugging Face Open LLM Leaderboard v2 API
  - Papers With Code SWE-bench leaderboard (scrape)
  - LiveCodeBench (GitHub raw JSON)
  - EvalPlus HumanEval+ leaderboard (GitHub raw JSON)

All functions return a dict keyed by a canonical model slug that maps to
the fetched score(s).  Results are cached in st.session_state or in a
simple TTL file cache to avoid hammering external endpoints.
"""

from __future__ import annotations

import json
import time
import hashlib
import pathlib
import requests
from typing import Optional

CACHE_DIR = pathlib.Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_SECONDS = 3600  # 1 hour


def _cache_path(key: str) -> pathlib.Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _load_cache(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
        if time.time() - payload["ts"] < CACHE_TTL_SECONDS:
            return payload["data"]
    except Exception:
        pass
    return None


def _save_cache(key: str, data: dict) -> None:
    try:
        _cache_path(key).write_text(json.dumps({"ts": time.time(), "data": data}))
    except Exception:
        pass


def _get(url: str, timeout: int = 8) -> Optional[dict | list]:
    """GET with basic error handling; returns parsed JSON or None."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "TabnineDash/1.0"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HumanEval+ via EvalPlus leaderboard
# ---------------------------------------------------------------------------

EVALPLUS_URL = (
    "https://raw.githubusercontent.com/evalplus/evalplus.github.io"
    "/main/leaderboard.json"
)

# Map EvalPlus model name substrings → our model IDs
EVALPLUS_SLUG_MAP = {
    "gpt-4.1":          "gpt-4.1",
    "gpt-4o":           "gpt-4.1",   # closest proxy when 4.1 not listed
    "o3":               "o3",
    "o4-mini":          "o4-mini",
    "claude-sonnet-4":  "claude-sonnet-4",
    "claude-3-7-sonnet":"claude-sonnet-3.7",
    "claude-3-5-sonnet":"claude-sonnet-3.5",
    "gemini-2.5-pro":   "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "grok-3":           "grok-3",
    "grok-3-mini":      "grok-3-mini",
    "devstral":         "devstral-small-2",
}


def fetch_humaneval_scores() -> dict[str, float]:
    """Return {model_id: humaneval_plus_score} from EvalPlus leaderboard."""
    key = "evalplus_humaneval"
    cached = _load_cache(key)
    if cached:
        return cached

    data = _get(EVALPLUS_URL)
    result: dict[str, float] = {}
    if data and isinstance(data, list):
        for entry in data:
            name = str(entry.get("model", "")).lower()
            score = entry.get("humaneval_plus")
            if score is None:
                score = entry.get("humaneval")
            if score is None:
                continue
            for slug, model_id in EVALPLUS_SLUG_MAP.items():
                if slug in name and model_id not in result:
                    result[model_id] = float(score) * 100  # convert 0-1 → %

    _save_cache(key, result)
    return result


# ---------------------------------------------------------------------------
# SWE-bench via Papers With Code API
# ---------------------------------------------------------------------------

SWE_BENCH_URL = (
    "https://paperswithcode.com/api/v1/sota/?task=software-engineering&dataset=swe-bench-verified"
)

SWE_SLUG_MAP = {
    "claude sonnet 4":      "claude-sonnet-4",
    "claude-sonnet-4":      "claude-sonnet-4",
    "claude 3.7":           "claude-sonnet-3.7",
    "claude-3.7":           "claude-sonnet-3.7",
    "claude 3.5":           "claude-sonnet-3.5",
    "claude-3.5":           "claude-sonnet-3.5",
    "gpt-4.1":              "gpt-4.1",
    "o3":                   "o3",
    "o4-mini":              "o4-mini",
    "gemini 2.5 pro":       "gemini-2.5-pro",
    "gemini-2.5-pro":       "gemini-2.5-pro",
    "devstral small":       "devstral-small-2",
    "devstral-small":       "devstral-small-2",
    "devstral":             "devstral-2",
}


def fetch_swebench_scores() -> dict[str, float]:
    """Return {model_id: swebench_resolved_%} from PapersWithCode API."""
    key = "swebench_scores"
    cached = _load_cache(key)
    if cached:
        return cached

    data = _get(SWE_BENCH_URL)
    result: dict[str, float] = {}
    if data and isinstance(data, dict):
        rows = data.get("results", [])
        for row in rows:
            name = str(row.get("model_name", "")).lower()
            score = row.get("best_metric")
            if score is None:
                continue
            for slug, model_id in SWE_SLUG_MAP.items():
                if slug.lower() in name and model_id not in result:
                    result[model_id] = float(score)

    _save_cache(key, result)
    return result


# ---------------------------------------------------------------------------
# LiveCodeBench (GitHub raw JSON)
# ---------------------------------------------------------------------------

LIVECODEBENCH_URL = (
    "https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench"
    "/main/lcb_runner/evaluation/results_summary.json"
)

LCB_SLUG_MAP = {
    "gpt-4.1":              "gpt-4.1",
    "o3":                   "o3",
    "o4-mini":              "o4-mini",
    "claude-sonnet-4":      "claude-sonnet-4",
    "claude-3-7-sonnet":    "claude-sonnet-3.7",
    "claude-3-5-sonnet":    "claude-sonnet-3.5",
    "gemini-2.5-pro":       "gemini-2.5-pro",
    "gemini-2.5-flash":     "gemini-2.5-flash",
    "grok-3":               "grok-3",
    "grok-3-mini":          "grok-3-mini",
}


def fetch_livecodebench_scores() -> dict[str, float]:
    """Return {model_id: LiveCodeBench pass@1 %} from LCB GitHub."""
    key = "livecodebench"
    cached = _load_cache(key)
    if cached:
        return cached

    data = _get(LIVECODEBENCH_URL)
    result: dict[str, float] = {}
    if data and isinstance(data, dict):
        for model_key, metrics in data.items():
            name = model_key.lower()
            score = None
            if isinstance(metrics, dict):
                score = metrics.get("pass@1") or metrics.get("score")
            if score is None:
                continue
            for slug, model_id in LCB_SLUG_MAP.items():
                if slug.lower() in name and model_id not in result:
                    result[model_id] = float(score) * 100 if float(score) <= 1.0 else float(score)

    _save_cache(key, result)
    return result


# ---------------------------------------------------------------------------
# Aggregate: merge all live sources with static fallbacks
# ---------------------------------------------------------------------------

def fetch_all_live_scores() -> dict[str, dict[str, Optional[float]]]:
    """
    Returns a nested dict:
      {model_id: {"humaneval": float|None, "swebench": float|None, "livecodebench": float|None}}

    Each value is None when the live source did not return data for that model.
    """
    humaneval  = fetch_humaneval_scores()
    swebench   = fetch_swebench_scores()
    lcb        = fetch_livecodebench_scores()

    all_ids = set(humaneval) | set(swebench) | set(lcb)
    return {
        mid: {
            "humaneval":     humaneval.get(mid),
            "swebench":      swebench.get(mid),
            "livecodebench": lcb.get(mid),
        }
        for mid in all_ids
    }
