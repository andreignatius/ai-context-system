# configuration for the code-builder capstone:
# the LLM the agents share

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langfuse.langchain import CallbackHandler

load_dotenv()

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))

# which backend serves the model: "ollama" (local) or "deepinfra" (cloud)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# 0.2 (low, non-zero): concentrates on the most-likely-correct output in production (fewer tail draws like a
# 1000-day z-score window), but keeps enough variation for the self-heal loop to escape a bad attempt (temp=0
# makes retries barely diverge). Eval at THIS deployed temperature to measure real reliability - don't inflate
# it for "eval variation". Tunable via env.
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

def get_llm() -> BaseChatModel:
    # DeepInfra (hosted, per-token) or local Ollama - both behind one LangChain interface, both at TEMPERATURE.
    if LLM_PROVIDER == "deepinfra":
        # cloud: deepinfra's openai-compatible endpoint (per-token, hosted GPUs)
        # Qwen3 is a REASONING model. DEFAULT it to thinking-OFF so it behaves like the fast, non-
        # reasoning coder model the app was tuned for. The self-heal loop already provides the reasoning,
        # and the eval showed thinking adds NO accuracy (6/9 either way) at ~7x the latency (489s vs 66s)
        # + truncation risk. Set DEEPINFRA_THINK=true to re-enable (then also raise DEEPINFRA_MAX_TOKENS).
        extra = {}
        if os.getenv("DEEPINFRA_THINK", "false").lower() != "true":
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return ChatOpenAI(
            model=os.getenv("DEEPINFRA_MODEL", "Qwen/Qwen3-32B"),   # was Qwen2.5-Coder-32B-Instruct (deprecated; DeepInfra aliased it to Qwen3-32B)
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            temperature=TEMPERATURE,
            # PIN max output tokens: DeepInfra's default (65536) exceeds Qwen3-32B's max_total_tokens
            # (40960) -> a 400. With thinking OFF (default) 8192 is plenty; raise via env if you re-enable
            # thinking (reasoning alone can burn ~8k on the coder step).
            max_tokens=int(os.getenv("DEEPINFRA_MAX_TOKENS", "8192")),
            **extra,
        )
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=TEMPERATURE,
    )

def get_langfuse_handler() -> CallbackHandler:
    # langfuse callback - reads LANGFUSE_* from the env (langfuse 4.x)
    return CallbackHandler()
