"""Configuration: LLM and observability setup.

Centralising construction here keeps nodes free of wiring details and makes the
model/handler easy to swap or mock in tests.
"""

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

load_dotenv()


def get_llm() -> ChatOllama:
    """Local Ollama chat model. Connection is lazy — no network call until invoke()."""
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
        temperature=0.7,
    )


def get_langfuse_handler() -> CallbackHandler:
    """Langfuse callback handler.

    In langfuse 4.x the handler reads LANGFUSE_* keys from the environment, so no
    arguments are needed. If keys are unset, tracing is simply a no-op.
    """
    return CallbackHandler()
