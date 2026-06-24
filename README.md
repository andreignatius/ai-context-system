# AI Context Management System

## Overview
An in-house AI system for managing LLM context using LangGraph, with a focus on **Write, Select, Compress, Isolate** strategies. Built for local execution with Ollama + DeepSeek/Llama.

## Stack
- **Framework**: LangGraph + LangChain
- **LLM**: Ollama (DeepSeek-R1, Llama 3.2)
- **Embeddings**: Ollama (nomic-embed-text)
- **Vector store**: ChromaDB
- **Persistence**: SQLite (LangGraph checkpointer)
- **Observability**: Langfuse
- **Frontend**: Open WebUI
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
- [ ] Multi-agent / sub-agents (ISOLATE)
- [ ] Guardrails (prompt injection protection)
- [ ] Langfuse observability (wired; trace not yet confirmed)
- [ ] Tests + eval, API server, deploy

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
# generated, gitignored: chroma_db/, checkpoints.sqlite
```

## Setup
1. Clone the repo
2. Create virtual environment: `python3.11 -m venv venv`
3. Install dependencies: `pip install -r langgraph-app/requirements.txt`
4. Ensure Ollama is running and pull the models: `ollama pull llama3.2` and `ollama pull nomic-embed-text`
5. (For RAG) ingest the sample doc once: `cd langgraph-app && python ingest.py`
6. Run: `python main.py`  (tip: `OLLAMA_MODEL=llama3.2:latest python main.py` for clean output)
7. Run tests: `cd langgraph-app && pytest`

## Config (env-overridable)
`OLLAMA_MODEL`, `SYSTEM_PROMPT`, `COMPRESS_AT`, `KEEP_RECENT`, `NUM_CTX_CAP`, `LANGFUSE_*` (see `.env.example`).

## Journal
See `docs/journal.md` for daily progress, notes, and blockers.

## Lessons
See `docs/lessons.md` for written-up concept notes (e.g. state & reducers).

## License
MIT