# AI Context Management System

## Overview
An in-house AI system for managing LLM context using LangGraph, with a focus on **Write, Select, Compress, Isolate** strategies. Built for local execution with Ollama.

Two parts:
- **Foundations** (`langgraph-app/`) — a chat/RAG agent that builds each context pillar (Write / Select / Compress / Isolate) end to end.
- **Capstone** (`code-builder/`) — a **multi-agent, self-healing code-builder** that applies the thesis: an orchestrator writes a spec, a QA agent writes tests *from the spec alone* (TDD), a coder implements it, a sandbox runs the tests, and an **auto-judge** routes each failure to the agent at fault (code / tests / spec) until it converges — or escalates to a human. Shipped as a FastAPI service, containerized with Docker, scored by an eval harness, and fully traced in Langfuse.

## Stack
- **Framework**: LangGraph + LangChain
- **LLM**: Ollama — Llama 3.2 (foundations), **qwen2.5-coder** (capstone builder + judge)
- **Embeddings**: Ollama (nomic-embed-text)
- **Vector store**: ChromaDB
- **Persistence**: SQLite (LangGraph checkpointer)
- **API**: FastAPI + uvicorn (capstone `/build` service)
- **Deploy**: Docker (containerized capstone)
- **Observability**: Langfuse (traces every agent call)
- **Sandbox / eval**: pytest in a subprocess (verifies *and* scores generated code)
- **Frontend**: Open WebUI / Swagger auto-docs
- **Language**: Python 3.11+

## Features (Planned)
Mapped to the Write / Select / Compress / Isolate thesis.
- [x] Local LLM setup (Ollama)
- [x] Open WebUI frontend
- [x] LangGraph agent with scratchpad
- [x] System prompt (persona; env-overridable)
- [x] Multi-turn chat loop
- [x] Persistent memory across restarts - SQLite checkpointer + thread_id (WRITE)
- [x] RAG retrieval over local docs - Chroma + embeddings (SELECT)
- [x] Agentic routing - the agent decides when to retrieve (SELECT)
- [x] Context compression - summary-buffer + RemoveMessage (COMPRESS)
- [x] Multi-agent / sub-agents (ISOLATE)
- [~] Guardrails (prompt injection protection)   <- partial: sandbox + brakes; no injection layer yet
- [x] Langfuse observability (wired; trace not yet confirmed)   <- CONFIRMED 26-Jun (Change 019/020)
- [x] Tests + eval, API server, deploy

### Capstone (code-builder) milestones   [26-Jun]
- [x] Multi-agent pipeline: orchestrator -> QA (TDD, spec-only) -> coder -> sandbox
- [x] Self-healing fix-loop + human-in-the-loop surgical edit (spec / tests / code)
- [x] Auto-judge: routes each failure to the agent at fault (code/tests/spec); 3 rounds then human
- [x] Eval harness: autonomous pass-rate (qwen2.5-coder ~80%; revealed the model is the ceiling)
- [x] FastAPI service (POST /build) + Docker container
- [x] Langfuse: every agent call traced (the whole build is a nested tree)
- [~] Tools (M8) / Guardrails (M9): spirit via the sandbox; model tool-calling + real isolation = future work

## Project Structure
```
ai-context-system/
├── README.md
├── docs/
│   ├── journal.md          # build log, design notes, curriculum
│   └── lessons.md          # concept write-ups (Lessons 001-009)
└── langgraph-app/
    ├── .env.example        # template for secrets (copy to .env)
    ├── requirements.txt    # active deps live; rest commented from the freeze
    ├── main.py             # chat loop (entry point)
    ├── ingest.py           # one-off: load docs into Chroma
    ├── simple_agent.py     # original single-file prototype (kept for reference)
    ├── data/
    │   └── facts.md        # sample document for RAG
    ├── src/
    │   ├── state.py        # AgentState (messages, scratchpad, query, context)
    │   ├── config.py       # LLM + embeddings + Langfuse + config constants
    │   ├── nodes.py        # process_input, retrieve, generate_response, compress, routers
    │   ├── graph.py        # builds + compiles the graph (conditional edges)
    │   └── rag.py          # embeddings + Chroma + ingest
    ├── demos/
    │   ├── reducer_demo.py     # Lesson 001 (messages grow via the reducer)
    │   └── retrieval_demo.py   # Lesson 006 (semantic retrieval in isolation)
    └── tests/
        └── test_graph.py
├── code-builder/           # CAPSTONE: multi-agent, self-healing code-builder
│   ├── Dockerfile          # containerize the API (M12)
│   ├── requirements.txt    # pinned deps for the image
│   └── src/
│       ├── state.py        # BuilderState + BuildEvent (the ledger)
│       ├── config.py       # LLM (qwen2.5-coder) + Langfuse handler
│       ├── agents.py       # orchestrator / QA / coder + the JUDGE
│       ├── sandbox.py      # temp dir + run pytest (the verifier)
│       ├── graph.py        # dispatcher + self-healing judge loop
│       ├── main.py         # CLI: interactive build + human intervention
│       ├── api.py          # FastAPI service: POST /build (M11)
│       ├── ui.py           # Streamlit web UI (chat + human-in-the-loop fix)
│       └── evals.py        # eval harness: autonomous pass-rate (M10)
# generated, gitignored: chroma_db/, checkpoints.sqlite, code-builder/.env
```

## Setup
1. Clone the repo
2. Create virtual environment: `python3.11 -m venv venv`
3. Install dependencies: `pip install -r langgraph-app/requirements.txt`
4. Ensure Ollama is running and pull the models: `ollama pull llama3.2` and `ollama pull nomic-embed-text`
5. (For RAG) ingest the sample doc once: `cd langgraph-app && python ingest.py`
6. Run: `python main.py`  (tip: `OLLAMA_MODEL=llama3.2:latest python main.py` for clean output)
7. Run tests: `cd langgraph-app && pytest`

### Capstone (code-builder)
```bash
ollama pull qwen2.5-coder                   # the builder + judge model
cd code-builder

# CLI - interactive build; human intervention when the auto-judge stays stuck
OLLAMA_MODEL=qwen2.5-coder:latest python -m src.main

# API service - interactive Swagger UI at http://localhost:8000/docs
uvicorn src.api:app --reload

# web UI - Streamlit chat with human-in-the-loop fix (needs the API running)
streamlit run src/ui.py

# eval harness - autonomous pass-rate over a fixed task suite
OLLAMA_MODEL=qwen2.5-coder:latest python -m src.evals

# Docker - containerized; reaches your host's Ollama
docker build -t code-builder .
docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 code-builder
```
Put `LANGFUSE_*` keys in `code-builder/.env` to trace every build in Langfuse.

## Config (env-overridable)
`OLLAMA_MODEL`, `SYSTEM_PROMPT`, `COMPRESS_AT`, `KEEP_RECENT`, `NUM_CTX_CAP`, `LANGFUSE_*` (see `.env.example`).

## Journal
See `docs/journal.md` for daily progress, notes, and blockers.

## Lessons
See `docs/lessons.md` for written-up concept notes (e.g. state & reducers).

## License
MIT