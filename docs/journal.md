# AI Context System Journal

## Project Goal
Build an in-house AI system with LangGraph for managing context, with guardrails and observability (Langfuse).

## Setup Log

### 2026-06-22

#### Initial Setup
- Created project directory: `~/Documents/ai-context-system/`
- Created subfolders: `langgraph-app/`, `docs/`
- Using Python 3.11.15 (from Open WebUI setup)

#### Ollama & Open WebUI
- Models installed: `deepseek-r1:latest`, `llama3:latest`, `llama3.2:latest`
- Open WebUI running at: `http://localhost:8080`
- Database location: `/Users/andre/Documents/openwebui-env/lib/python3.11/site-packages/open_webui/data/webui.db`

#### Learning Notes
- Context engineering = managing what goes into the LLM's "short-term memory"
- Key strategies: Write, Select, Compress, Isolate
- LangGraph = orchestrates stateful AI workflows
- Langfuse = observability/tracing platform

### Next Steps
- [ ] Install LangGraph
- [ ] Set up Langfuse (cloud or self-hosted)
- [ ] Build a simple agent with memory scratchpad

### Questions/Blockers
- How to structure the LangGraph state?
- Which database for persistent memory?
