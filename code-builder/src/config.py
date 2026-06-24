# configuration for the code-builder capstone:
# the LLM the agents share

import os

from langchain_ollama import ChatOllama

def get_llm() -> ChatOllama:
    # local Ollama chat model
    # lower temperature than the chat agent
    # code wants determinism, not creativity
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
        temperature=0.2,
    )