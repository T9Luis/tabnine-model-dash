"""
Static model registry — sourced from Tabnine official documentation.

Sources
-------
- https://docs.tabnine.com/main/welcome/readme/ai-models
- https://docs.tabnine.com/main/welcome/readme/system-requirements/system-requirements
- https://docs.tabnine.com/main/administering-tabnine/managing-your-team/settings/models-settings

tabnine_available flag
----------------------
True  = confirmed in the Tabnine docs as of the last registry update.
False = frontier model tracked for awareness; expected to be added soon.
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
    # Capability scores (0–10, curated from docs + benchmarks)
    score_code_completion: float
    score_code_generation: float
    score_reasoning: float
    score_chat: float
    score_debugging: float
    score_refactoring: float
    score_documentation: float
    score_multifile: float
    # Official benchmark scores (None = not publicly reported)
    bench_humaneval: Optional[float]
    bench_mbpp: Optional[float]
    bench_swebench: Optional[float]
    bench_gpqa: Optional[float]
    bench_mmlu: Optional[float]
    bench_livecodebench: Optional[float]
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
    tabnine_available: bool   # True = in Tabnine today; False = tracked/upcoming
    cost_tier: int            # 1 = low cost, 2 = medium, 3 = high (flagship/thinking)
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
#   cc, cg, rs, ch, db, rf, dc, mf,       <- 8 capability scores
#   he, mb, sw, gp, mm, lc,               <- 6 benchmark scores
#   ctx, th, tc, st,                       <- context, thinking, tool_call, streaming
#   gm, gr, tp, vr,                        <- gpu_min, gpu_rec, tensor_par, vram
#   lic, url, avail, tags
def _m(id, name, prov, fam, cat, dep, plan,
        cc, cg, rs, ch, db, rf, dc, mf,
        he, mb, sw, gp, mm, lc,
        ctx, th, tc, st,
        gm, gr, tp, vr,
        lic, url, avail, cost, tags):
    return TabnineModel(
        id=id, display_name=name, provider=prov, family=fam,
        category=cat, deployment=dep, plan=plan,
        score_code_completion=cc, score_code_generation=cg,
        score_reasoning=rs, score_chat=ch,
        score_debugging=db, score_refactoring=rf,
        score_documentation=dc, score_multifile=mf,
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
#          cc cg rs ch db rf dc mf | he mb sw gp mm lc |
#          ctx th tc st | gm gr tp vr | lic url avail tags
# ---------------------------------------------------------------------------

MODELS: list[TabnineModel] = [

    # ── Anthropic / Claude ──────────────────────────────────────────────────
    # Available in Tabnine today (confirmed in docs)
    _m("claude-4.8-opus",   "Claude 4.8 Opus",   "Anthropic","Claude","thinking",CLD,PRO,
       9.5,9.6,9.8,9.5,9.6,9.5,9.4,9.7, N,N,N,N,N,N,
       200000,YES,YES,YES, N,N,N,N, "Anthropic API",_D, YES, 3,
       ["thinking","flagship","newest"]),

    _m("claude-4.7-opus",   "Claude 4.7 Opus",   "Anthropic","Claude","thinking",CLD,PRO,
       9.4,9.5,9.7,9.4,9.5,9.4,9.3,9.6, N,N,N,N,N,N,
       200000,YES,YES,YES, N,N,N,N, "Anthropic API",_D, YES, 3,
       ["thinking","flagship"]),

    _m("claude-4.6-opus",   "Claude 4.6 Opus",   "Anthropic","Claude","thinking",CLD,PRO,
       9.3,9.4,9.6,9.3,9.4,9.3,9.2,9.5, N,N,N,N,N,N,
       200000,YES,YES,YES, N,N,N,N, "Anthropic API",_D, YES, 3,
       ["thinking","flagship"]),

    _m("claude-4.6-sonnet", "Claude 4.6 Sonnet", "Anthropic","Claude","thinking",CLD,PRO,
       9.2,9.3,9.5,9.3,9.3,9.2,9.2,9.4, N,N,N,N,N,N,
       200000,YES,YES,YES, N,N,N,N, "Anthropic API",_D, YES, 2,
       ["thinking","balanced"]),

    _m("claude-4.5-opus",   "Claude 4.5 Opus",   "Anthropic","Claude","thinking",CLD,PRO,
       9.2,9.3,9.5,9.2,9.3,9.2,9.1,9.4, N,N,N,N,N,N,
       200000,YES,YES,YES, N,N,N,N, "Anthropic API",_D, YES, 3,
       ["thinking"]),

    _m("claude-4.5-sonnet", "Claude 4.5 Sonnet", "Anthropic","Claude","thinking",CLD,PRO,
       9.1,9.2,9.4,9.2,9.2,9.1,9.1,9.3, N,N,N,N,N,N,
       200000,YES,YES,YES, N,N,N,N, "Anthropic API",_D, YES, 2,
       ["thinking","popular"]),

    _m("claude-4.5-haiku",  "Claude 4.5 Haiku",  "Anthropic","Claude","thinking",CLD,PRO,
       8.8,8.9,9.0,8.9,8.9,8.8,8.8,9.0, N,N,N,N,N,N,
       200000,YES,YES,YES, N,N,N,N, "Anthropic API",_D, YES, 1,
       ["thinking","fast","efficient"]),

    _m("claude-4-sonnet",   "Claude 4 Sonnet",   "Anthropic","Claude","thinking",CLD,PRO,
       9.0,9.1,9.3,9.1,9.1,9.0,9.0,9.2, 92.1,N,72.7,84.8,90.1,N,
       200000,YES,YES,YES, N,N,N,N, "Anthropic API",_D, YES, 2,
       ["thinking","proven"]),

    # Upcoming — not yet confirmed in Tabnine docs
    _m("claude-5-opus",     "Claude 5 Opus",     "Anthropic","Claude","thinking",CLD,PRO,
       9.8,9.9,9.9,9.7,9.8,9.8,9.7,9.9, N,N,N,N,N,N,
       200000,YES,YES,YES, N,N,N,N, "Anthropic API",_D, NO, 3,
       ["thinking","flagship","upcoming"]),

    # ── OpenAI / GPT ────────────────────────────────────────────────────────
    # Available in Tabnine today
    _m("gpt-5.5",      "GPT-5.5",      "OpenAI","GPT","thinking",CLD,PRO,
       9.5,9.6,9.7,9.5,9.6,9.5,9.4,9.6, N,N,N,N,N,N,
       1000000,YES,YES,YES, N,N,N,N, "OpenAI API",_D, YES, 3,
       ["thinking","flagship","newest","large-context"]),

    _m("gpt-5.4",      "GPT-5.4",      "OpenAI","GPT","thinking",CLD,PRO,
       9.4,9.5,9.6,9.4,9.5,9.4,9.3,9.5, N,N,N,N,N,N,
       1000000,YES,YES,YES, N,N,N,N, "OpenAI API",_D, YES, 3,
       ["thinking","flagship","large-context"]),

    _m("gpt-5.3-codex","GPT-5.3 Codex","OpenAI","GPT","coding",CLD,PRO,
       9.5,9.6,9.3,8.8,9.5,9.4,9.0,9.4, N,N,N,N,N,N,
       1000000,YES,YES,YES, N,N,N,N, "OpenAI API",_D, YES, 2,
       ["thinking","coding-specialist","large-context"]),

    _m("gpt-5.2-codex","GPT-5.2 Codex","OpenAI","GPT","coding",CLD,PRO,
       9.4,9.5,9.2,8.7,9.4,9.3,8.9,9.3, N,N,N,N,N,N,
       1000000,YES,YES,YES, N,N,N,N, "OpenAI API",_D, YES, 2,
       ["thinking","coding-specialist","large-context"]),

    _m("gpt-5.2",      "GPT-5.2",      "OpenAI","GPT","thinking",CLD,PRO,
       9.3,9.4,9.5,9.3,9.4,9.3,9.2,9.4, N,N,N,N,N,N,
       1000000,YES,YES,YES, N,N,N,N, "OpenAI API",_D, YES, 3,
       ["thinking","flagship","large-context"]),

    _m("gpt-5",        "GPT-5",        "OpenAI","GPT","thinking",CLD,PRO,
       9.2,9.3,9.5,9.3,9.3,9.2,9.2,9.3, N,N,N,N,N,N,
       1000000,YES,YES,YES, N,N,N,N, "OpenAI API",_D, YES, 3,
       ["thinking","flagship","large-context"]),

    _m("gpt-4o",       "GPT-4o",       "OpenAI","GPT","general",CLD,PRO,
       9.0,9.1,8.9,9.0,9.0,8.9,9.1,8.8, 90.2,87.5,N,N,88.7,N,
       128000,NO,YES,YES, N,N,N,N, "OpenAI API",_D, YES, 2,
       ["proven","multimodal","popular"]),

    # Upcoming
    _m("gpt-5.6",      "GPT-5.6",      "OpenAI","GPT","thinking",CLD,PRO,
       9.6,9.7,9.8,9.6,9.7,9.6,9.5,9.7, N,N,N,N,N,N,
       2000000,YES,YES,YES, N,N,N,N, "OpenAI API",_D, NO, 3,
       ["thinking","upcoming","large-context"]),

    _m("o3-pro",       "o3 Pro",       "OpenAI","GPT","thinking",CLD,PRO,
       9.5,9.6,9.9,9.2,9.7,9.5,9.0,9.6, 96.7,N,71.7,87.7,91.8,N,
       200000,YES,YES,YES, N,N,N,N, "OpenAI API",_D, NO, 3,
       ["thinking","upcoming","highest-reasoning"]),

    # ── Google / Gemini ─────────────────────────────────────────────────────
    # Available in Tabnine today
    _m("gemini-3.5-pro","Gemini 3.5 Pro","Google","Gemini","general",CLD,PRO,
       9.3,9.4,9.5,9.3,9.4,9.2,9.2,9.4, N,N,N,N,N,N,
       1000000,NO,YES,YES, N,N,N,N, "Google API",_D, YES, 2,
       ["flagship","huge-context","multimodal","newest"]),

    _m("gemini-3.1-pro","Gemini 3.1 Pro","Google","Gemini","general",CLD,PRO,
       9.1,9.2,9.3,9.1,9.2,9.0,9.0,9.2, N,N,N,N,N,N,
       1000000,NO,YES,YES, N,N,N,N, "Google API",_D, YES, 2,
       ["flagship","huge-context","multimodal"]),

    _m("gemini-3.0-pro","Gemini 3.0 Pro","Google","Gemini","thinking",CLD,PRO,
       9.1,9.3,9.6,9.2,9.3,9.1,9.0,9.4, N,N,63.8,84.0,89.0,N,
       1000000,YES,YES,YES, N,N,N,N, "Google API",_D, YES, 2,
       ["thinking","huge-context","multimodal"]),

    # Upcoming
    _m("gemini-ultra-3","Gemini Ultra 3","Google","Gemini","thinking",CLD,PRO,
       9.6,9.7,9.8,9.5,9.7,9.5,9.5,9.7, N,N,N,N,N,N,
       2000000,YES,YES,YES, N,N,N,N, "Google API",_D, NO, 3,
       ["thinking","upcoming","huge-context","multimodal"]),

    # ── Self-hosted / Open-weight ────────────────────────────────────────────
    # Available in Tabnine today
    _m("devstral-2-123b",    "Devstral 2 (123B)",     "Mistral","Devstral","coding",SH,ENT,
       9.1,9.3,8.5,8.0,9.2,9.0,8.5,9.1, N,N,56.0,N,N,N,
       128000,NO,YES,YES, "4x H100","4x B200",4,320,
       "Modified MIT",_H, YES, 2, ["self-hosted","coding-specialist","open-weights"]),

    _m("devstral-small-2-24b","Devstral Small 2 (24B)","Mistral","Devstral","coding",SH,ENT,
       8.7,8.9,7.8,7.5,8.8,8.6,8.0,8.7, N,N,46.8,N,N,N,
       128000,NO,YES,YES, "2x H100","2x B200",2,160,
       "Apache 2.0",_H, YES, 1, ["self-hosted","coding-specialist","open-weights","efficient"]),

    _m("minimax-m2.7",   "MiniMax-M2.7",   "MiniMax","MiniMax","general",SH,ENT,
       8.5,8.7,8.8,9.0,8.6,8.5,9.0,8.8, N,N,N,N,N,N,
       180000,YES,YES,YES, "8x H100","8x B200",8,640,
       "Open weights",_H, YES, 2, ["self-hosted","MoE","thinking","large-context"]),

    _m("minimax-m2.5",   "MiniMax-M2.5",   "MiniMax","MiniMax","general",SH,ENT,
       8.3,8.5,8.5,8.8,8.4,8.3,8.8,8.5, N,N,N,N,N,N,
       180000,NO,YES,YES, "2x H200","2x B200",2,160,
       "Open weights",_H, YES, 1, ["self-hosted","MoE"]),

    _m("laguna-xs2",     "Poolside Laguna XS.2",      "Poolside","Laguna","coding",SH,ENT,
       9.0,9.2,8.6,7.8,9.1,9.0,8.3,9.0, N,N,N,N,N,N,
       131072,YES,YES,YES, "2x H100","4x H100",4,320,
       "Proprietary",_H, YES, 2, ["self-hosted","coding-specialist","thinking"]),

    _m("laguna-xs2-fp8", "Poolside Laguna XS.2 FP8",  "Poolside","Laguna","coding",SH,ENT,
       8.9,9.1,8.5,7.7,9.0,8.9,8.2,8.9, N,N,N,N,N,N,
       131072,YES,YES,YES, "2x H100","4x H100",4,320,
       "Proprietary",_H, YES, 1, ["self-hosted","coding-specialist","FP8"]),

    _m("glm-4.7",        "GLM-4.7",           "Zhipu AI","GLM","general",SH,ENT,
       8.6,8.7,8.7,8.8,8.7,8.6,8.8,8.7, N,N,N,N,N,N,
       128000,NO,YES,YES, "8x H100","2x B200",2,160,
       "Open weights",_H, YES, 1, ["self-hosted","MoE"]),

    _m("qwen3-coder-480b","Qwen-3-Coder-480B","Alibaba","Qwen","coding",SH,ENT,
       9.2,9.4,8.8,8.5,9.3,9.2,8.7,9.2, N,N,N,N,N,N,
       131072,YES,YES,YES, "8x H100","8x B200",8,640,
       "Apache 2.0",_H, YES, 2, ["self-hosted","coding-specialist","open-weights","large"]),

    _m("qwen3-30b",      "Qwen-3-30B",        "Alibaba","Qwen","general",SH,ENT,
       8.5,8.7,8.8,8.6,8.7,8.5,8.7,8.6, N,N,N,N,N,N,
       131072,YES,YES,YES, "2x B200","2x B200",2,160,
       "Apache 2.0",_H, YES, 1, ["self-hosted","open-weights","efficient"]),

    _m("qwen3.6-27b",    "Qwen3.6-27B",       "Alibaba","Qwen","general",SH,ENT,
       8.4,8.6,8.6,8.5,8.6,8.4,8.6,8.5, N,N,N,N,N,N,
       131072,NO,YES,YES, "4x H100","2x B200",2,160,
       "Apache 2.0",_H, YES, 1, ["self-hosted","open-weights"]),

    # Upcoming self-hosted
    _m("llama-4-maverick","Llama 4 Maverick (17Bx128E)","Meta","Llama","general",SH,ENT,
       8.8,9.0,8.9,8.8,8.9,8.7,8.9,8.8, N,N,N,N,N,N,
       1000000,NO,YES,YES, "4x H100","4x B200",4,320,
       "Llama 4 Community License",_H, NO, 2,
       ["self-hosted","MoE","upcoming","large-context","open-weights"]),

    _m("llama-4-scout",   "Llama 4 Scout (17Bx16E)",  "Meta","Llama","coding",SH,ENT,
       8.5,8.7,8.5,8.4,8.6,8.4,8.5,8.5, N,N,N,N,N,N,
       10000000,NO,YES,YES, "2x H100","2x B200",2,160,
       "Llama 4 Community License",_H, NO, 1,
       ["self-hosted","MoE","upcoming","huge-context","efficient","open-weights"]),

    _m("deepseek-r2",     "DeepSeek R2",       "DeepSeek","DeepSeek","thinking",SH,ENT,
       9.3,9.5,9.7,8.8,9.4,9.3,8.9,9.4, N,N,N,N,N,N,
       128000,YES,YES,YES, "8x H100","8x B200",8,640,
       "MIT",_H, NO, 3,
       ["self-hosted","thinking","open-weights","upcoming","MoE"]),

    _m("qwen3-coder-32b", "Qwen3-Coder-32B",   "Alibaba","Qwen","coding",SH,ENT,
       8.9,9.1,8.7,8.4,9.0,8.9,8.6,8.9, N,N,N,N,N,N,
       131072,YES,YES,YES, "2x H100","2x B200",2,160,
       "Apache 2.0",_H, NO, 1,
       ["self-hosted","coding-specialist","open-weights","upcoming","efficient"]),

    _m("mistral-large-3", "Mistral Large 3",   "Mistral","Mistral","general",SH,ENT,
       8.8,9.0,8.9,8.8,8.9,8.8,8.9,8.8, N,N,N,N,N,N,
       128000,NO,YES,YES, "4x H100","4x B200",4,320,
       "MRL v2",_H, NO, 2,
       ["self-hosted","general","open-weights","upcoming"]),

    # ── Tabnine proprietary ──────────────────────────────────────────────────
    _m("tabnine-protected","Tabnine Protected Model","Tabnine","Tabnine","coding",CLD,PRO,
       8.5,8.5,7.5,7.8,8.4,8.4,8.2,8.3, N,N,N,N,N,N,
       100000,NO,YES,YES, N,N,N,N,
       "Tabnine proprietary — no code leaves Tabnine infrastructure",_D, YES, 1,
       ["privacy-first","no-data-sharing"]),
]


# ---------------------------------------------------------------------------
# Task → score column mapping
# ---------------------------------------------------------------------------
TASKS = {
    "Inline Code Completion":       "score_code_completion",
    "Code Generation":              "score_code_generation",
    "Complex Reasoning / Planning": "score_reasoning",
    "Chat / Q&A":                   "score_chat",
    "Debugging":                    "score_debugging",
    "Refactoring":                  "score_refactoring",
    "Documentation":                "score_documentation",
    "Multi-file / Agentic":         "score_multifile",
}

BENCHMARK_COLUMNS = {
    "HumanEval (pass@1 %)":   "bench_humaneval",
    "MBPP (pass@1 %)":        "bench_mbpp",
    "SWE-bench (% resolved)": "bench_swebench",
    "GPQA (%)":               "bench_gpqa",
    "MMLU (%)":               "bench_mmlu",
    "LiveCodeBench (%)":      "bench_livecodebench",
}

SCORE_COLUMNS = {v: k for k, v in TASKS.items()}
