from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TabnineModel:
    id: str
    display_name: str
    provider: str
    family: str
    category: str
    deployment: str
    plan: str
    score_code_completion: float
    score_code_generation: float
    score_reasoning: float
    score_chat: float
    score_debugging: float
    score_refactoring: float
    score_documentation: float
    score_multifile: float
    bench_humaneval: Optional[float]
    bench_mbpp: Optional[float]
    bench_swebench: Optional[float]
    bench_gpqa: Optional[float]
    bench_mmlu: Optional[float]
    bench_livecodebench: Optional[float]
    context_window: int
    thinking_mode: bool
    tool_calling: bool
    streaming: bool
    gpu_min: Optional[str]
    gpu_recommended: Optional[str]
    tensor_parallel: Optional[int]
    vram_gb: Optional[int]
    license_note: str
    doc_url: str
    tags: list = field(default_factory=list)

_D = "https://docs.tabnine.com/main/welcome/readme/ai-models"
_H = "https://docs.tabnine.com/main/welcome/readme/system-requirements/system-requirements"

def _c(id,name,prov,fam,cat,dep,plan,cc,cg,rs,ch,db,rf,dc,mf,he,mb,sw,gp,mm,lc,ctx,th,tc,st,gm,gr,tp,vr,lic,url,tags):
    return TabnineModel(id,name,prov,fam,cat,dep,plan,cc,cg,rs,ch,db,rf,dc,mf,he,mb,sw,gp,mm,lc,ctx,th,tc,st,gm,gr,tp,vr,lic,url,tags)

N=None
CLD="Cloud"
SH="Self Hosted"
PRO="Pro / Enterprise"
ENT="Enterprise (Private Mode)"

MODELS = [
    _c("claude-4.8-opus","Claude 4.8 Opus","Anthropic","Claude","thinking",CLD,PRO,9.5,9.6,9.8,9.5,9.6,9.5,9.4,9.7,N,N,N,N,N,N,200000,True,True,True,N,N,N,N,"Anthropic API",_D,["thinking","flagship","newest"]),
    _c("claude-4.7-opus","Claude 4.7 Opus","Anthropic","Claude","thinking",CLD,PRO,9.4,9.5,9.7,9.4,9.5,9.4,9.3,9.6,N,N,N,N,N,N,200000,True,True,True,N,N,N,N,"Anthropic API",_D,["thinking","flagship"]),
    _c("claude-4.6-opus","Claude 4.6 Opus","Anthropic","Claude","thinking",CLD,PRO,9.3,9.4,9.6,9.3,9.4,9.3,9.2,9.5,N,N,N,N,N,N,200000,True,True,True,N,N,N,N,"Anthropic API",_D,["thinking","flagship"]),
    _c("claude-4.6-sonnet","Claude 4.6 Sonnet","Anthropic","Claude","thinking",CLD,PRO,9.2,9.3,9.5,9.3,9.3,9.2,9.2,9.4,N,N,N,N,N,N,200000,True,True,True,N,N,N,N,"Anthropic API",_D,["thinking","balanced"]),
    _c("claude-4.5-opus","Claude 4.5 Opus","Anthropic","Claude","thinking",CLD,PRO,9.2,9.3,9.5,9.2,9.3,9.2,9.1,9.4,N,N,N,N,N,N,200000,True,True,True,N,N,N,N,"Anthropic API",_D,["thinking"]),
    _c("claude-4.5-sonnet","Claude 4.5 Sonnet","Anthropic","Claude","thinking",CLD,PRO,9.1,9.2,9.4,9.2,9.2,9.1,9.1,9.3,N,N,N,N,N,N,200000,True,True,True,N,N,N,N,"Anthropic API",_D,["thinking","popular"]),
    _c("claude-4.5-haiku","Claude 4.5 Haiku","Anthropic","Claude","thinking",CLD,PRO,8.8,8.9,9.0,8.9,8.9,8.8,8.8,9.0,N,N,N,N,N,N,200000,True,True,True,N,N,N,N,"Anthropic API",_D,["thinking","fast","efficient"]),
    _c("claude-4-sonnet","Claude 4 Sonnet","Anthropic","Claude","thinking",CLD,PRO,9.0,9.1,9.3,9.1,9.1,9.0,9.0,9.2,92.1,N,72.7,84.8,90.1,N,200000,True,True,True,N,N,N,N,"Anthropic API",_D,["thinking","proven"]),
    _c("gpt-5.5","GPT-5.5","OpenAI","GPT","thinking",CLD,PRO,9.5,9.6,9.7,9.5,9.6,9.5,9.4,9.6,N,N,N,N,N,N,1000000,True,True,True,N,N,N,N,"OpenAI API",_D,["thinking","flagship","newest","large-context"]),
    _c("gpt-5.4","GPT-5.4","OpenAI","GPT","thinking",CLD,PRO,9.4,9.5,9.6,9.4,9.5,9.4,9.3,9.5,N,N,N,N,N,N,1000000,True,True,True,N,N,N,N,"OpenAI API",_D,["thinking","flagship","large-context"]),
    _c("gpt-5.3-codex","GPT-5.3 Codex","OpenAI","GPT","coding",CLD,PRO,9.5,9.6,9.3,8.8,9.5,9.4,9.0,9.4,N,N,N,N,N,N,1000000,True,True,True,N,N,N,N,"OpenAI API",_D,["thinking","coding-specialist","large-context"]),
    _c("gpt-5.2-codex","GPT-5.2 Codex","OpenAI","GPT","coding",CLD,PRO,9.4,9.5,9.2,8.7,9.4,9.3,8.9,9.3,N,N,N,N,N,N,1000000,True,True,True,N,N,N,N,"OpenAI API",_D,["thinking","coding-specialist","large-context"]),
    _c("gpt-5.2","GPT-5.2","OpenAI","GPT","thinking",CLD,PRO,9.3,9.4,9.5,9.3,9.4,9.3,9.2,9.4,N,N,N,N,N,N,1000000,True,True,True,N,N,N,N,"OpenAI API",_D,["thinking","flagship","large-context"]),
    _c("gpt-5","GPT-5","OpenAI","GPT","thinking",CLD,PRO,9.2,9.3,9.5,9.3,9.3,9.2,9.2,9.3,N,N,N,N,N,N,1000000,True,True,True,N,N,N,N,"OpenAI API",_D,["thinking","flagship","large-context"]),
    _c("gpt-4o","GPT-4o","OpenAI","GPT","general",CLD,PRO,9.0,9.1,8.9,9.0,9.0,8.9,9.1,8.8,90.2,87.5,N,N,88.7,N,128000,False,True,True,N,N,N,N,"OpenAI API",_D,["proven","multimodal","popular"]),
    _c("gemini-3.5-pro","Gemini 3.5 Pro","Google","Gemini","general",CLD,PRO,9.3,9.4,9.5,9.3,9.4,9.2,9.2,9.4,N,N,N,N,N,N,1000000,False,True,True,N,N,N,N,"Google API",_D,["flagship","huge-context","multimodal","newest"]),
    _c("gemini-3.1-pro","Gemini 3.1 Pro","Google","Gemini","general",CLD,PRO,9.1,9.2,9.3,9.1,9.2,9.0,9.0,9.2,N,N,N,N,N,N,1000000,False,True,True,N,N,N,N,"Google API",_D,["flagship","huge-context","multimodal"]),
    _c("gemini-3.0-pro","Gemini 3.0 Pro","Google","Gemini","thinking",CLD,PRO,9.1,9.3,9.6,9.2,9.3,9.1,9.0,9.4,N,N,63.8,84.0,89.0,N,1000000,True,True,True,N,N,N,N,"Google API",_D,["thinking","huge-context","multimodal"]),
    _c("devstral-2-123b","Devstral 2 (123B)","Mistral","Devstral","coding",SH,ENT,9.1,9.3,8.5,8.0,9.2,9.0,8.5,9.1,N,N,56.0,N,N,N,128000,False,True,True,"4x H100","4x B200",4,320,"Modified MIT",_H,["self-hosted","coding-specialist","open-weights"]),
    _c("devstral-small-2-24b","Devstral Small 2 (24B)","Mistral","Devstral","coding",SH,ENT,8.7,8.9,7.8,7.5,8.8,8.6,8.0,8.7,N,N,46.8,N,N,N,128000,False,True,True,"2x H100","2x B200",2,160,"Apache 2.0",_H,["self-hosted","coding-specialist","open-weights","efficient"]),
    _c("minimax-m2.7","MiniMax-M2.7","MiniMax","MiniMax","general",SH,ENT,8.5,8.7,8.8,9.0,8.6,8.5,9.0,8.8,N,N,N,N,N,N,180000,True,True,True,"8x H100","8x B200",8,640,"Open weights",_H,["self-hosted","MoE","thinking","large-context"]),
    _c("minimax-m2.5","MiniMax-M2.5","MiniMax","MiniMax","general",SH,ENT,8.3,8.5,8.5,8.8,8.4,8.3,8.8,8.5,N,N,N,N,N,N,180000,False,True,True,"2x H200","2x B200",2,160,"Open weights",_H,["self-hosted","MoE"]),
    _c("laguna-xs2","Poolside Laguna XS.2","Poolside","Laguna","coding",SH,ENT,9.0,9.2,8.6,7.8,9.1,9.0,8.3,9.0,N,N,N,N,N,N,131072,True,True,True,"2x H100","4x H100",4,320,"Proprietary",_H,["self-hosted","coding-specialist","thinking"]),
    _c("laguna-xs2-fp8","Poolside Laguna XS.2 FP8","Poolside","Laguna","coding",SH,ENT,8.9,9.1,8.5,7.7,9.0,8.9,8.2,8.9,N,N,N,N,N,N,131072,True,True,True,"2x H100","4x H100",4,320,"Proprietary",_H,["self-hosted","coding-specialist","FP8"]),
    _c("glm-4.7","GLM-4.7","Zhipu AI","GLM","general",SH,ENT,8.6,8.7,8.7,8.8,8.7,8.6,8.8,8.7,N,N,N,N,N,N,128000,False,True,True,"8x H100","2x B200",2,160,"Open weights",_H,["self-hosted","MoE"]),
    _c("qwen3-coder-480b","Qwen-3-Coder-480B","Alibaba","Qwen","coding",SH,ENT,9.2,9.4,8.8,8.5,9.3,9.2,8.7,9.2,N,N,N,N,N,N,131072,True,True,True,"8x H100","8x B200",8,640,"Apache 2.0",_H,["self-hosted","coding-specialist","open-weights","large"]),
    _c("qwen3-30b","Qwen-3-30B","Alibaba","Qwen","general",SH,ENT,8.5,8.7,8.8,8.6,8.7,8.5,8.7,8.6,N,N,N,N,N,N,131072,True,True,True,"2x B200","2x B200",2,160,"Apache 2.0",_H,["self-hosted","open-weights","efficient"]),
    _c("qwen3.6-27b","Qwen3.6-27B","Alibaba","Qwen","general",SH,ENT,8.4,8.6,8.6,8.5,8.6,8.4,8.6,8.5,N,N,N,N,N,N,131072,False,True,True,"4x H100","2x B200",2,160,"Apache 2.0",_H,["self-hosted","open-weights"]),
    _c("tabnine-protected","Tabnine Protected Model","Tabnine","Tabnine","coding",CLD,PRO,8.5,8.5,7.5,7.8,8.4,8.4,8.2,8.3,N,N,N,N,N,N,100000,False,True,True,N,N,N,N,"Tabnine proprietary",_D,["privacy-first","no-data-sharing"]),
]

TASKS = {
    "Inline Code Completion": "score_code_completion",
    "Code Generation": "score_code_generation",
    "Complex Reasoning / Planning": "score_reasoning",
    "Chat / Q&A": "score_chat",
    "Debugging": "score_debugging",
    "Refactoring": "score_refactoring",
    "Documentation": "score_documentation",
    "Multi-file / Agentic": "score_multifile",
}

BENCHMARK_COLUMNS = {
    "HumanEval (pass@1 %)": "bench_humaneval",
    "MBPP (pass@1 %)": "bench_mbpp",
    "SWE-bench (% resolved)": "bench_swebench",
    "GPQA (%)": "bench_gpqa",
    "MMLU (%)": "bench_mmlu",
    "LiveCodeBench (%)": "bench_livecodebench",
}

SCORE_COLUMNS = {v: k for k, v in TASKS.items()}
