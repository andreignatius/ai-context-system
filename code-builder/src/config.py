# configuration for the code-builder capstone:
# the LLM the agents share

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

load_dotenv()

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))

# which backend serves the model: "ollama" (local) or "deepinfra" (cloud)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

def get_llm() -> ChatOllama:
    # local Ollama chat model
    # lower temperature than the chat agent
    # code wants determinism, not creativity

    if LLM_PROVIDER == "deepinfra":
        # cloud: deepinfra's openai-compatible endpoint (per-token, hosted GPUs)
        return ChatOpenAI(
            model=os.getenv("DEEPINFRA_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            temperature=0.5,
        )
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.5,
    )

def get_langfuse_handler() -> CallbackHandler:
    # langfuse callback - reads LANGFUSE_* from the env (langfuse 4.x)
    return CallbackHandler()
