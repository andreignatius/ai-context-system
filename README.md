# AI Context Management System

## Overview
An in-house AI system for managing LLM context using LangGraph, with a focus on **Write, Select, Compress, Isolate** strategies. Built for local execution with Ollama + DeepSeek/Llama.

## Stack
- **Framework**: LangGraph + LangChain
- **LLM**: Ollama (DeepSeek-R1, Llama 3.2)
- **Observability**: Langfuse
- **Frontend**: Open WebUI
- **Language**: Python 3.11+

## Features (Planned)
- [x] Local LLM setup (Ollama)
- [x] Open WebUI frontend
- [ ] LangGraph agent with scratchpad
- [ ] Persistent memory (SQLite/Vector DB)
- [ ] RAG retrieval
- [ ] Guardrails (prompt injection protection)
- [ ] Langfuse observability

## Project Structure

## Setup
1. Clone the repo
2. Create virtual environment: `python3.11 -m venv venv`
3. Install dependencies: `pip install -r langgraph-app/requirements.txt`
4. Ensure Ollama is running with models pulled
5. Run: `python langgraph-app/simple_agent.py`

## Journal
See `docs/journal.md` for daily progress, notes, and blockers.

## License
MIT