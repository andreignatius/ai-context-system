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

### Change 001: project review and refactor (22-Jun-2026)
Reviewed the repo structure and refactored the single-file prototype into a package.

- **Dependencies**: `requirements.txt` was a full `pip freeze` of the Open WebUI env
  (265 pkgs). Kept the freeze for reference but commented out everything except the
  5 direct deps: `langgraph`, `langchain-ollama`, `langchain-core`, `langfuse`,
  `python-dotenv` (+ `pytest` for dev). pip resolves transitive deps from those.
- **Secrets**: added `.env.example` template (Langfuse keys + `OLLAMA_MODEL`).
- **Refactor**: split `simple_agent.py` into a package under `langgraph-app/src/`:
  - `state.py` - `AgentState` now uses the `add_messages` reducer instead of
    mutating the messages list in place.
  - `config.py` - LLM + Langfuse construction, isolated for easy swap/mock.
  - `nodes.py` - **Langfuse is now actually wired in** via
    `config={"callbacks": [handler]}` (before, the handler was created but never
    passed to `invoke`, so nothing was traced). Scratchpad now accumulates instead
    of overwriting each turn.
  - `graph.py` - `build_graph()` assembles + compiles.
  - `main.py` - entry point.
  - `tests/test_graph.py` - 3 offline tests (graph compiles, state keys, node output).
- **Kept** `simple_agent.py` as the original prototype for reference.
- Verified via import smoke test (pytest not yet installed in the env).

#### Learning Notes
- `add_messages` reducer = a fold/accumulator over state: nodes return only the NEW
  messages, LangGraph appends them. Avoids manual in-place `.append()`.
- Observability only works if the callback handler is passed to `.invoke()` - having
  the library imported is not enough.

### Change 002: Milestone 1 - system prompt (23-Jun-2026)
Added a system prompt to the agent (build milestone #1). Typed by hand to learn it.

- `config.py` - added `SYSTEM_PROMPT`, read via `os.getenv("SYSTEM_PROMPT", default)`
  so it can be overridden without editing code (config over hardcoding).
- `nodes.py` - `process_input` now prepends a `SystemMessage` at the FRONT of the
  conversation, but only ONCE: an `any(isinstance(m, SystemMessage) ...)` guard stops
  a duplicate system prompt being added every turn (matters for multi-turn later).
- `tests/test_graph.py` - updated to expect `[System, Human]` on the first turn, and
  added an edge-case test that a second system message is NOT added.
- Verified live against Ollama: same question + same code, but
  `SYSTEM_PROMPT="...pirate..." python main.py` flipped the personality, then a plain
  run returned to the friendly default. Proved both (a) system prompts steer behaviour
  and (b) the env var overrides the default temporarily.

#### Learning Notes
- A "system prompt" is just a `SystemMessage` placed first; models are trained to read
  rules-before-dialogue, so order matters. (See Lesson 002.)
- The context window is a fixed token budget; the system prompt is an always-on cost
  paid every call -> keep it lean, push question-specific info to retrieval later.

### Next Steps
- [x] Install LangGraph
- [ ] Set up Langfuse (cloud or self-hosted) and confirm a trace appears
- [ ] Build a simple agent with memory scratchpad
- [ ] Install pytest and run the real test suite (`pip install pytest && pytest`)
- [ ] Add a conditional edge / loop (move beyond linear flow)
- [ ] Add a checkpointer for persistent state across runs

### Questions/Blockers
- How to structure the LangGraph state? -> started: typed `AgentState` + reducers.
- Which database for persistent memory? -> likely SQLite checkpointer first, vector DB
  (Chroma) later for RAG.

---

## Learning Curriculum

A progressive checklist from fundamentals -> the project's goals
(Write / Select / Compress / Isolate). Check off as concepts genuinely click -
the goal is understanding, not just running code.
Concept write-ups live in `docs/lessons.md`.

Two linked views of the same journey:
- CONCEPTS  = "do I understand this?" (read/learn)
- MILESTONES = "have I shipped this?" (build)

Rule of thumb: each milestone is done when it is BUILT and explained in
`docs/lessons.md`. Ship + explain = the loop that actually upskills.
Time ranges = focused second pass -> first-time-while-learning. The slow pass
is where depth happens; that is fine.


### A. Concepts checklist

### 1. LangGraph fundamentals
- [ ] **State & reducers** - `TypedDict`, `Annotated[list, add_messages]`, why
      reducers beat in-place mutation
- [ ] **Nodes** - pure-ish functions that return partial state updates
- [ ] **Edges** - `set_entry_point`, `add_edge`, `END`
- [ ] **Compile & invoke** - what `builder.compile()` produces; `.invoke()` vs `.stream()`
- [ ] **Conditional edges** - `add_conditional_edges`, routing on state
- [ ] **Cycles / loops** - building an agent that iterates until a condition

### 2. Working with LLMs in the graph
- [ ] **Message types** - `HumanMessage`, `AIMessage`, `SystemMessage`
- [ ] **Prompt construction** - passing message history vs. f-string stuffing
- [ ] **Tool calling** - binding tools, `ToolNode`, the ReAct pattern
- [ ] **Structured output** - forcing JSON / schema-validated responses

### 3. Persistence & memory
- [ ] **Checkpointers** - `MemorySaver`, then `SqliteSaver`
- [ ] **Threads** - `thread_id`, resuming conversations
- [ ] **Short-term vs long-term memory** - scratchpad vs. durable store

### 4. Context engineering (the project's core thesis)
- [ ] **Write** - persisting state/notes outside the context window (scratchpad, memory)
- [ ] **Select** - retrieving only what's relevant (RAG, BM25, embeddings)
- [ ] **Compress** - summarization nodes, trimming history, token budgeting
- [ ] **Isolate** - sub-agents / sandboxing context per task

### 5. RAG retrieval
- [ ] **Embeddings** - `sentence-transformers`, vector similarity
- [ ] **Vector store** - Chroma: ingest, chunk (`text-splitters`), query
- [ ] **Hybrid retrieval** - combining BM25 + dense (`rank-bm25`)
- [ ] **Retrieval as a graph node** - wiring Select into the flow

### 6. Guardrails & robustness
- [ ] **Prompt injection** - detection / mitigation basics
- [ ] **Input/output validation** - schema checks, refusals
- [ ] **Error handling & retries** - `tenacity`, graceful node failures

### 7. Observability & evaluation
- [ ] **Langfuse traces** - confirm spans appear; read a trace
- [ ] **Metrics** - latency, token counts, cost per run
- [ ] **Evaluation** - scoring outputs, regression checks on prompts

### 8. Quality habits (carry over from your quant projects)
- [ ] **Tests** - unit tests per node, integration test behind an "Ollama up" marker
- [ ] **Config over hardcoding** - env vars, no magic strings
- [ ] **Git hygiene** - let git hold history (avoid `_OLD` / `_v2` files)


### B. Build milestones

Numbered 1-12 in build order. Each lists the concepts it exercises.
(Estimates are focused build time; expect longer on the first pass while learning.)

Progress: [x] 1 system prompt (23-Jun) | [ ] 2-12 pending

**Milestone 0 - Foundations** (quick wins, build confidence)

| # | Task                       | Why it matters                          | Concepts | Est.  |
|---|----------------------------|-----------------------------------------|----------|-------|
| 1 | Add a system prompt        | Controls agent personality/behavior     | 2.1      | 10 m  |
| 2 | Multi-turn chat loop       | Real conversations, not single Q&A      | 1.5, 3.2 | 20 m  |
| - | Confirm a Langfuse trace   | Proves observability is actually wired  | 7.1      | 15 m  |

**Milestone 1 - WRITE**

| # | Task                       | Why it matters                          | Concepts | Est.  |
|---|----------------------------|-----------------------------------------|----------|-------|
| 3 | Persistent memory          | Stops forgetting between sessions       | 3.1, 3.2 | 30 m  |

**Milestone 2 - SELECT**

| # | Task                       | Why it matters                          | Concepts | Est.  |
|---|----------------------------|-----------------------------------------|----------|-------|
| 4 | Add ChromaDB (vectors)     | Stores embeddings of your documents     | 5.2      | 45 m  |
| 5 | Build a RAG pipeline       | Answer questions about your docs        | 5.x      | 1 h   |

**Milestone 3 - COMPRESS** (was missing from the original roadmap)

| # | Task                       | Why it matters                          | Concepts   | Est. |
|---|----------------------------|-----------------------------------------|------------|------|
| 6 | Summarize / trim history   | Keep within context window; the core    | 4.Compress | 1 h  |
|   |                            | "context management" idea in the name   |            |      |

**Milestone 4 - ISOLATE**

| # | Task                       | Why it matters                          | Concepts  | Est.  |
|---|----------------------------|-----------------------------------------|-----------|-------|
| 7 | Multi-agent system         | Specialized agents per task             | 4.Isolate | 2 h+  |

**Milestone 5 - Harden & ship** (portfolio / demo-day items)

| #  | Task                      | Why it matters                          | Concepts | Est.  |
|----|---------------------------|-----------------------------------------|----------|-------|
| 8  | Add tools (calc, search)  | Agent takes actions, not just chat      | 2.3      | 1 h   |
| 9  | Add guardrails            | Prevent prompt injection / abuse        | 6.x      | 1 h   |
| 10 | Tests + a small eval      | Catch regressions; quant-style rigor    | 7.3, 8   | 1 h   |
| 11 | API server (FastAPI)      | Expose agent as a web service           | -        | 1 h   |
| 12 | Deploy to cloud           | Run 24/7; the interview artifact        | -        | 2 h+  |

Target story for an interviewer:
"A context-management agent that Writes, Selects, Compresses and Isolates -
observable (Langfuse), tested, and deployed."
