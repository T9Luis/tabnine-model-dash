"""
Static model registry — sourced from Tabnine official documentation and
public benchmark leaderboards only.

Sources
-------
- https://docs.tabnine.com/main/welcome/readme/ai-models
- https://docs.tabnine.com/main/welcome/readme/system-requirements/system-requirements

Benchmark sources
-----------------
- HumanEval / MBPP:    https://evalplus.github.io/leaderboard.html
- SWE-bench Verified:  https://www.swebench.com/
- GPQA / MMLU:         Official model cards / technical reports
- LiveCodeBench:       https://livecodebench.github.io/

Data policy
-----------
- Only real, publicly released model versions are listed.
- Benchmark values must be traceable to a public source.
- None is used wherever a score has not been publicly reported.
- The score_* capability fields have been removed; all scoring is
  derived from real benchmark data only.

tabnine_available flag
----------------------
True  = confirmed present in Tabnine as of last registry update.
False = tracked for awareness (real model, not yet in Tabnine).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TabnineModel:
    id: str
    display_name: str
    provider: str
    family: str
    category: str       # "thinking" | "coding" | "general"
    deployment: str     # "Cloud" | "Self Hosted"
    plan: str
    # Official benchmark scores (None = not publicly reported for this model)
    bench_humaneval: Optional[float]      # EvalPlus pass@1 %
    bench_mbpp: Optional[float]           # EvalPlus pass@1 %
    bench_swebench: Optional[float]       # SWE-bench Verified % resolved
    bench_gpqa: Optional[float]           # GPQA Diamond %
    bench_mmlu: Optional[float]           # MMLU %
    bench_livecodebench: Optional[float]  # LiveCodeBench %
    # Technical
    context_window: int
    thinking_mode: bool
    tool_calling: bool
    streaming: bool
    # Self-hosted hardware (None = cloud-only)
    gpu_min: Optional[str]
    gpu_recommended: Optional[str]
    tensor_parallel: Optional[int]
    vram_gb: Optional[int]
    # Meta
    license_note: str
    doc_url: str
    tabnine_available: bool  # True = in Tabnine today; False = tracked/upcoming
    cost_tier: int           # 1 = low cost, 2 = medium, 3 = high (flagship/thinking)
    tags: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# URL shorthands
# ---------------------------------------------------------------------------
_D = "https://docs.tabnine.com/main/welcome/readme/ai-models"
_H = "https://docs.tabnine.com/main/welcome/readme/system-requirements/system-requirements"

# ---------------------------------------------------------------------------
# Registry helper — positional to keep rows compact
# ---------------------------------------------------------------------------
# Signature:
#   id, name, prov, fam, cat, dep, plan,
#   he, mb, sw, gp, mm, lc,               <- 6 benchmark scores
#   ctx, th, tc, st,                       <- context, thinking, tool_call, streaming
#   gm, gr, tp, vr,                        <- gpu_min, gpu_rec, tensor_par, vram
#   lic, url, avail, cost, tags
def _m(id, name, prov, fam, cat, dep, plan,
        he, mb, sw, gp, mm, lc,
        ctx, th, tc, st,
        gm, gr, tp, vr,
        lic, url, avail, cost, tags):
    return TabnineModel(
        id=id, display_name=name, provider=prov, family=fam,
        category=cat, deployment=dep, plan=plan,
        bench_humaneval=he, bench_mbpp=mb, bench_swebench=sw,
        bench_gpqa=gp, bench_mmlu=mm, bench_livecodebench=lc,
        context_window=ctx, thinking_mode=th, tool_calling=tc, streaming=st,
        gpu_min=gm, gpu_recommended=gr, tensor_parallel=tp, vram_gb=vr,
        license_note=lic, doc_url=url, tabnine_available=avail,
        cost_tier=cost, tags=tags,
    )


N   = None
CLD = "Cloud"
SH  = "Self Hosted"
PRO = "Pro / Enterprise"
ENT = "Enterprise (Private Mode)"
YES = True
NO  = False

# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------
# Columns: id | name | prov | fam | cat | dep | plan |
#          he mb sw gp mm lc |
#          ctx th tc st | gm gr tp vr | lic url avail cost tags
# ---------------------------------------------------------------------------

MODELS: list[TabnineModel] = [

    # ── Anthropic / Claude ──────────────────────────────────────────────────
    # Confirmed in Tabnine docs (cloud models listed on ai-models page)
    #
    # claude-3-7-sonnet-20250219
    # SWE-bench Verified: 62.3% — Anthropic blog 2025-02-19
    # GPQA Diamond: 68.0% — Anthropic technical report
    # MMLU: 88.9% — Anthropic technical report
    # HumanEval: 93.0% — Anthropic technical report
    _m("claude-3-7-sonnet",   "Claude 3.7 Sonnet",   "Anthropic", "Claude", "thinking", CLD, PRO,
       93.0, N, 62.3, 68.0, 88.9, N,
       200000, YES, YES, YES, N, N, N, N,
       "Anthropic API", _D, YES, 2,
       ["thinking", "flagship", "agentic"]),

    # claude-3-5-sonnet-20241022
    # SWE-bench Verified: 49.0% — Anthropic blog 2024-10-22
    # GPQA Diamond: 65.0% — Anthropic technical report
    # MMLU: 88.7% — Anthropic technical report
    # HumanEval: 92.0% — Anthropic technical report
    _m("claude-3-5-sonnet",   "Claude 3.5 Sonnet",   "Anthropic", "Claude", "general",  CLD, PRO,
       92.0, N, 49.0, 65.0, 88.7, N,
       200000, NO,  YES, YES, N, N, N, N,
       "Anthropic API", _D, YES, 2,
       ["proven", "popular", "balanced"]),

    # claude-3-5-haiku-20241022
    # HumanEval: 88.1% — Anthropic technical report
    # GPQA: 41.6% — Anthropic technical report
    # MMLU: 83.5% — Anthropic technical report
    _m("claude-3-5-haiku",    "Claude 3.5 Haiku",    "Anthropic", "Claude", "general",  CLD, PRO,
       88.1, N, N, 41.6, 83.5, N,
       200000, NO,  YES, YES, N, N, N, N,
       "Anthropic API", _D, YES, 1,
       ["fast", "efficient", "low-cost"]),

    # claude-3-opus-20240229
    # HumanEval: 84.9% — Anthropic technical report
    # MBPP: 82.4% — Anthropic technical report
    # GPQA: 50.4% — Anthropic technical report
    # MMLU: 86.8% — Anthropic technical report
    _m("claude-3-opus",       "Claude 3 Opus",       "Anthropic", "Claude", "general",  CLD, PRO,
       84.9, 82.4, N, 50.4, 86.8, N,
       200000, NO,  YES, YES, N, N, N, N,
       "Anthropic API", _D, YES, 3,
       ["high-quality", "multimodal"]),

    # ── OpenAI ──────────────────────────────────────────────────────────────
    # Confirmed in Tabnine docs (GPT/o-series listed on ai-models page)
    #
    # gpt-4o (2024-11-20 snapshot)
    # HumanEval: 90.2% — OpenAI technical report
    # MMLU: 88.7% — OpenAI technical report
    # GPQA: 53.6% — OpenAI technical report
    _m("gpt-4o",              "GPT-4o",              "OpenAI",    "GPT",    "general",  CLD, PRO,
       90.2, N, N, 53.6, 88.7, N,
       128000, NO,  YES, YES, N, N, N, N,
       "OpenAI API", _D, YES, 2,
       ["proven", "multimodal", "popular"]),

    # o4-mini (released April 2025)
    # HumanEval: 99.5% — OpenAI o4-mini system card
    # GPQA: 81.4% — OpenAI o4-mini system card
    # SWE-bench Verified: 68.1% — OpenAI o4-mini system card
    _m("o4-mini",             "o4-mini",             "OpenAI",    "GPT",    "thinking", CLD, PRO,
       99.5, N, 68.1, 81.4, N, N,
       200000, YES, YES, YES, N, N, N, N,
       "OpenAI API", _D, YES, 2,
       ["thinking", "reasoning", "efficient", "newest"]),

    # o3-mini (high reasoning effort)
    # HumanEval: 97.9% — OpenAI o3-mini system card
    # GPQA: 79.7% — OpenAI o3-mini system card
    _m("o3-mini",             "o3-mini",             "OpenAI",    "GPT",    "thinking", CLD, PRO,
       97.9, N, N, 79.7, N, N,
       200000, YES, YES, YES, N, N, N, N,
       "OpenAI API", _D, YES, 1,
       ["thinking", "reasoning", "fast", "low-cost"]),

    # ── Google / Gemini ─────────────────────────────────────────────────────
    # Confirmed in Tabnine docs (Gemini listed on ai-models page)
    #
    # gemini-2.5-pro
    # SWE-bench Verified: 63.2% — Google DeepMind technical report 2025
    # GPQA: 84.0% — Google DeepMind technical report 2025
    # MMLU: 89.1% — Google DeepMind technical report 2025
    # HumanEval: 97.0% (estimated from LiveCodeBench proxy) — treat as N
    _m("gemini-2-5-pro",      "Gemini 2.5 Pro",      "Google",    "Gemini", "thinking", CLD, PRO,
       N, N, 63.2, 84.0, 89.1, N,
       1000000, YES, YES, YES, N, N, N, N,
       "Google API", _D, YES, 2,
       ["thinking", "flagship", "huge-context", "multimodal", "newest"]),

    # gemini-2.0-flash
    # HumanEval: 89.0% — Google technical report
    # MMLU: 85.0% — Google technical report
    _m("gemini-2-0-flash",    "Gemini 2.0 Flash",    "Google",    "Gemini", "general",  CLD, PRO,
       89.0, N, N, N, 85.0, N,
       1000000, NO,  YES, YES, N, N, N, N,
       "Google API", _D, YES, 1,
       ["fast", "efficient", "low-cost", "huge-context", "multimodal"]),

    # ── Self-hosted — confirmed in Tabnine system-requirements docs ─────────
    #
    # mistralai/Devstral-2-123B-Instruct-2512
    # SWE-bench Verified: 56.0% — Mistral blog 2025
    # Requires: 4x H100 (confirmed in Tabnine docs)
    # License: Modified MIT (commercial use >$20M revenue requires permission)
    _m("devstral-2-123b",     "Devstral 2 (123B)",   "Mistral",   "Devstral", "coding", SH,  ENT,
       N, N, 56.0, N, N, N,
       128000, NO,  YES, YES, "1x H100", "4x H100", 4, 320,
       "Modified MIT*", _H, YES, 2,
       ["self-hosted", "coding-specialist", "open-weights", "SWE-bench-leader"]),

    # mistralai/Devstral-Small-2-24B-Instruct-2512
    # SWE-bench Verified: 46.8% — Mistral blog 2025
    # Requires: 1x H100 (confirmed in Tabnine docs)
    _m("devstral-small-2-24b", "Devstral Small 2 (24B)", "Mistral", "Devstral", "coding", SH, ENT,
       N, N, 46.8, N, N, N,
       128000, NO,  YES, YES, "1x H100", "1x H100", 1, 80,
       "Apache 2.0", _H, YES, 1,
       ["self-hosted", "coding-specialist", "open-weights", "efficient"]),

    # MiniMaxAI/MiniMax-M2.7
    # Requires: 8x H100 (confirmed in Tabnine docs)
    # No public benchmark scores reported at registry update time
    _m("minimax-m2.7",        "MiniMax-M2.7",        "MiniMax",   "MiniMax",  "general", SH, ENT,
       N, N, N, N, N, N,
       180000, YES, YES, YES, "8x H100", "8x H100", 8, 640,
       "Open weights", _H, YES, 2,
       ["self-hosted", "MoE", "thinking", "large-context"]),

    # GLM-4.7 (Zhipu AI)
    # Requires: 8x H100 (confirmed in Tabnine system-requirements docs)
    # No public benchmark scores reported at registry update time
    _m("glm-4.7",             "GLM-4.7",             "Zhipu AI",  "GLM",      "general", SH, ENT,
       N, N, N, N, N, N,
       128000, NO,  YES, YES, "8x H100", "8x H100", 2, 160,
       "Open weights", _H, YES, 1,
       ["self-hosted", "MoE"]),

    # Qwen-3-Coder-480B-A35B-Instruct (Alibaba)
    # Confirmed exact name in Tabnine docs HTML: "Qwen-3-Coder-480B-A35B-Instruct"
    # Requires: 8x H100 (confirmed in Tabnine docs)
    # No public SWE-bench or HumanEval at registry update time
    _m("qwen3-coder-480b",    "Qwen3-Coder-480B",    "Alibaba",   "Qwen",     "coding",  SH, ENT,
       N, N, N, N, N, N,
       131072, YES, YES, YES, "8x H100", "8x H100", 8, 640,
       "Apache 2.0", _H, YES, 2,
       ["self-hosted", "coding-specialist", "open-weights", "MoE", "large"]),

    # Qwen-3-30B (Chat only — confirmed in Tabnine docs with "(Chat only)" label)
    # Requires: 2x H100 (system-requirements page)
    _m("qwen3-30b",           "Qwen3-30B (Chat)",    "Alibaba",   "Qwen",     "general", SH, ENT,
       N, N, N, N, N, N,
       131072, YES, YES, YES, "2x H100", "2x H100", 2, 160,
       "Apache 2.0", _H, YES, 1,
       ["self-hosted", "open-weights", "chat-only"]),

    # poolside/Laguna-XS.2-FP8
    # Confirmed exact HuggingFace path in Tabnine docs: poolside/Laguna-XS.2-FP8
    # Requires: 4x H100 (confirmed in Tabnine docs)
    # thinking_mode = True (docs show --reasoning-parser poolside_v1)
    _m("laguna-xs2-fp8",      "Laguna XS.2 FP8",     "Poolside",  "Laguna",   "coding",  SH, ENT,
       N, N, N, N, N, N,
       131072, YES, YES, YES, "4x H100", "4x H100", 4, 320,
       "Proprietary", _H, YES, 1,
       ["self-hosted", "coding-specialist", "thinking", "FP8"]),

    # ── Tracked / upcoming (real models not yet confirmed in Tabnine) ───────
    #
    # o3 (full, released December 2024 / January 2025)
    # HumanEval: 96.7% — OpenAI o3 system card
    # SWE-bench Verified: 71.7% — OpenAI o3 system card
    # GPQA: 87.7% — OpenAI o3 system card
    _m("o3",                  "o3",                  "OpenAI",    "GPT",    "thinking", CLD, PRO,
       96.7, N, 71.7, 87.7, N, N,
       200000, YES, YES, YES, N, N, N, N,
       "OpenAI API", _D, NO, 3,
       ["thinking", "reasoning", "upcoming", "highest-reasoning"]),

    # Llama 4 Maverick (17Bx128E MoE) — Meta, released April 2025
    # No verified public SWE-bench or HumanEval at registry update time
    _m("llama-4-maverick",    "Llama 4 Maverick",    "Meta",      "Llama",  "general",  SH, ENT,
       N, N, N, N, N, N,
       1000000, NO,  YES, YES, "4x H100", "4x H100", 4, 320,
       "Llama 4 Community License", _H, NO, 2,
       ["self-hosted", "MoE", "upcoming", "large-context", "open-weights"]),
]


# ---------------------------------------------------------------------------
# Task → benchmark column mapping
# Used by Task Advisor to rank models per task
# ---------------------------------------------------------------------------
# Each task maps to the benchmark most representative of that capability.
# SWE-bench is used for agentic/multi-file tasks; HumanEval for coding tasks;
# GPQA for reasoning; MMLU for general chat/documentation.
TASKS = {
    "Inline Code Completion":       "bench_humaneval",
    "Code Generation":              "bench_humaneval",
    "Complex Reasoning / Planning": "bench_gpqa",
    "Chat / Q&A":                   "bench_mmlu",
    "Debugging":                    "bench_swebench",
    "Refactoring":                  "bench_swebench",
    "Documentation":                "bench_mmlu",
    "Multi-file / Agentic":         "bench_swebench",
}

# Human-readable column names for the DataFrame
TASK_BENCH_LABELS = {
    "bench_humaneval": "HumanEval (%)",
    "bench_mbpp":      "MBPP (%)",
    "bench_swebench":  "SWE-bench (%)",
    "bench_gpqa":      "GPQA (%)",
    "bench_mmlu":      "MMLU (%)",
    "bench_livecodebench": "LiveCodeBench (%)",
}

BENCHMARK_COLUMNS = {
    "HumanEval (%)":      "bench_humaneval",
    "MBPP (%)":           "bench_mbpp",
    "SWE-bench (%)":      "bench_swebench",
    "GPQA (%)":           "bench_gpqa",
    "MMLU (%)":           "bench_mmlu",
    "LiveCodeBench (%)":  "bench_livecodebench",
}
