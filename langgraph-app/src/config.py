"""Configuration: LLM and observability setup.

Centralising construction here keeps nodes free of wiring details and makes the
model/handler easy to swap or mock in tests.
"""

import os

import ollama
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

load_dotenv()

# system prompt: standing instructions for the agent (Milestone 1)
# short and specific, override via the SYSTEM_PROMPT env var
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a concise, friendly assistant. Answer in 2-3 sentences. "
    "If you are unsure, kindly say so rather than guessing."
)

# model + context window (milestone 6)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

# compression config (env-overridable), COMPRESS_AT must be > KEEP_RECENT
COMPRESS_AT = int(os.getenv("COMPRESS_AT", "12"))   # compress when len(messages) > this
KEEP_RECENT = int(os.getenv("KEEP_RECENT", "6"))    # keep system + this many recent msgs
NUM_CTX_CAP = int(os.getenv("NUM_CTX_CAP", "8192")) # cap the window (RAM guard)

def get_context_window(model: str, cap: int) -> int:
    """Detect the model's max context length at runtime (via Ollama), capped.

    Detect the MAX, then cap it - that capped value is the window we set on the LLM
    and use as the compression budget. Falls back to `cap` if Ollama is unreachable.
    """
    try:
        info = ollama.show(model)
        model_info = getattr(info, "modelinfo", {}) or {}
        for key, value in model_info.items():
            if key.endswith("context_length"):
                return min(int(value), cap)
    except Exception:
        pass
    return cap

# resolved once at import: the real budget for this run
CONTEXT_WINDOW = get_context_window(OLLAMA_MODEL, NUM_CTX_CAP)

def get_llm() -> ChatOllama:
    """Local Ollama chat model. Connection is lazy — no network call until invoke()
    
    num_ctx pins the effective context window (Ollama defaults it low otherwise)
    so CONTEXT_WINDOW is a real budget we can measure against
    """
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
        # model=os.getenv("OLLAMA_MODEL", "deepseek-r1:latest"),
        temperature=0.7,
        num_ctx=CONTEXT_WINDOW,
    )


def get_langfuse_handler() -> CallbackHandler:
    """Langfuse callback handler.

    In langfuse 4.x the handler reads LANGFUSE_* keys from the environment, so no
    arguments are needed. If keys are unset, tracing is simply a no-op.
    """
    return CallbackHandler()
