# configuration for the code-builder capstone:
# the LLM the agents share

import os

from langchain_ollama import ChatOllama

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))

def get_llm() -> ChatOllama:
    # local Ollama chat model
    # lower temperature than the chat agent
    # code wants determinism, not creativity
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.5,
    )