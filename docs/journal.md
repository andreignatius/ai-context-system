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

### Change 003: Milestone 2 - multi-turn chat loop (23-Jun-2026)
Turned the single-shot entry point into a real conversation. Typed by hand.

- `main.py` - added a `chat()` function: a `while True` loop that reads input, calls
  `app.invoke(...)`, prints the reply, then saves the result back via
  `messages = result["messages"]` so the NEXT turn starts where this one ended.
  Kept the old `main()` (commented out) for reference.
- Key distinction learned: the `add_messages` reducer appends WITHIN one graph run
  (across nodes); the chat loop carries state BETWEEN graph runs (across turns).
  A 3-turn conversation = 3 graph runs wrapped in 1 chat loop.
- Verified live: told it "my name is Andre, favourite number 42", then a later turn
  correctly recalled both, and multiplied 42 x 2 = 84. The single-shot version could
  not have remembered (it starts each run from `messages = []`).
- The system-prompt-once guard (Milestone 1) paid off: one system message stayed at
  the front across all turns, no duplicates.
- "Break it on purpose" experiment: commented out `messages = result["messages"]` and
  the agent forgot everything (Turn 2: "our conversation just started"). Confirmed
  memory lives in that single carry-forward line. Also confirmed the `scratchpad`
  save-back is a red herring here - scratchpad is not fed to the LLM yet. (See
  Lesson 003.)

#### Learning Notes
- Graph run = one `app.invoke(...)` = one question answered.
- Chat loop = the `while` loop that strings many graph runs together, feeding memory
  forward. Memory lives in the variable carried across iterations, not in the graph.
- Memory is currently IN-PROCESS only: quit the script and it is gone. Durable memory
  across sessions = a checkpointer (Milestone 3 / WRITE).
- Full write-up in Lesson 003 (graph runs vs the chat loop).

### Reference: where Open WebUI fits (23-Jun-2026)
Mental map of the local stack, and how our project relates to Open WebUI.

What `open-webui serve` starts (it is a finished PRODUCT):
- Frontend: a Svelte web app (the nice UI in the browser).
- Backend: a FastAPI (Python) server via uvicorn, on port 8080.
- Database: SQLite (`webui.db`) - users, chat history, settings.
- Built-ins: auth/login, RAG (doc upload + embeddings + Chroma), tools, web search.

Key point: Open WebUI is a CLIENT of Ollama, exactly like our `main.py` is. Both talk
to the same Ollama engine on `localhost:11434` and run the same models
(llama3.2, deepseek-r1). Open WebUI just wraps it in a polished app.

The stack as layers:
```
  Browser UI        <- Open WebUI's Svelte frontend (the "face")
      |
  Agent logic       <- OUR LangGraph app (the "brain": context mgmt) / Open WebUI backend
      |
  Ollama engine     <- runs the models (both clients share this)
```

How OUR project differs: Open WebUI is the car you drive; we are building an engine to
learn how cars work. It already HAS most of our milestones (RAG, tools, API, UI) as
features - but using a feature is not understanding it. We build the agent brain from
scratch so we understand the machinery Open WebUI hides. Milestone 11 (FastAPI) is the
point where our brain could get a face like Open WebUI's.

Can we point the Open WebUI UI at OUR agent instead of its built-ins? Yes - two routes:
1. OpenAI-compatible API: expose our agent as `POST /v1/chat/completions` (+ `/v1/models`)
   with SSE streaming, then add it under Open WebUI Settings -> Connections as an
   OpenAI API base URL. Our agent then appears as a selectable "model".
2. Open WebUI "Pipelines"/"Functions": their official plugin framework for injecting
   custom Python (including a LangGraph agent) as a model in the UI.
Either way it is Milestone 11 plus a compatibility shim (map OpenAI-style messages
<-> our AgentState). Noted as a future option, not a near-term task.

### Change 004: Milestone 3 - persistent memory (23-Jun-2026)
Memory now survives a full restart, via a LangGraph checkpointer. Typed by hand.

- Installed `langgraph-checkpoint-sqlite==3.1.0` (pulls in `langgraph-checkpoint`
  and `aiosqlite`); uncommented/added those three in `requirements.txt`.
- `graph.py` - `build_graph(checkpointer=None)` and `compile(checkpointer=...)`. The
  default `None` keeps the old behaviour (and the existing tests) working.
- `main.py` - `chat()` now:
  - builds the graph with a checkpointer,
  - uses a `thread_id` ("save slot") in `config={"configurable": {"thread_id": ...}}`,
  - passes ONLY `{"query": user_input}` to invoke - the checkpointer reloads prior
    messages automatically. The manual `messages = result["messages"]` carrying is gone.
- Two stages: `MemorySaver` (RAM, proves the mechanism within one run) -> `SqliteSaver`
  (disk, `checkpoints.sqlite`, survives a restart). `*.sqlite` is already gitignored.
- Verified across two SEPARATE process launches: told it name+number in run A, quit,
  asked in run B, and it answered correctly from disk. Also confirmed at the storage
  layer by reading 5 saved messages back out of `checkpoints.sqlite` for thread andre-1.

#### Learning Notes
- Checkpointer = auto-save of graph state after every step. thread_id = which save slot.
  Same thread_id across runs -> the conversation resumes; new thread_id -> fresh slot.
- Swapping `MemorySaver` -> `SqliteSaver` changed only the storage line; the loop,
  thread_id, and invoke were identical. Backends are interchangeable behind one interface.
- DEBUGGING LESSON: separate the system from the model. The chat output looked like
  "it forgot", but the database proved the history WAS saved+reloaded - the deepseek-r1
  reasoning model just gave a confused answer. Check the layer that owns the behaviour;
  do not trust the surface.
- CONTEXT-ENGINEERING LESSON (on-thesis): persisting EVERYTHING verbatim backfired - the
  model's own earlier "I don't remember across restarts" denials got checkpointed, then
  reloaded, then reinforced (a feedback loop). This is exactly why Compress/Select exist:
  curate what stays in context, don't hoard it. Fixed by starting a fresh thread_id and
  using llama3.2. Full write-up in Lesson 004.

### Change 005: Milestone 4 - RAG ingest + retrieval (SELECT) (23-Jun-2026)
Built the INGEST half of RAG and verified retrieval in isolation. Code typed by hand.

- Installed `langchain-chroma==1.1.0`; pulled the Ollama embedding model
  `nomic-embed-text`. Uncommented chromadb + langchain-text-splitters and added
  langchain-chroma in requirements.txt. Added `**/chroma_db/` to .gitignore.
- `data/facts.md` - synthetic test doc (fictional "Project Zephyr") full of UNKNOWABLE
  facts, so a correct answer proves retrieval rather than the model guessing.
- `src/rag.py` - `get_embeddings` (OllamaEmbeddings, nomic-embed-text),
  `get_vectorstore` (Chroma persisted to `chroma_db/`), `ingest` (read -> split with
  RecursiveCharacterTextSplitter chunk_size=300/overlap=50 -> embed -> add_texts).
- `ingest.py` - one-off ingest script (the offline phase). Ingested 4 chunks.
- `demos/retrieval_demo.py` - `similarity_search(question, k=2)` verified retrieval in
  ISOLATION (no LLM). Asked "lead engineer + budget"; got back the People chunk
  (Mei Tanaka) and Numbers chunk (4.2M SGD) - semantic match, the question never used
  those words.

#### Learning Notes
- Same embedding model for ingest AND query (nomic-embed-text both sides) or vectors
  are not comparable.
- Build/verify the retrieval piece in ISOLATION before wiring it into the graph (that
  is Milestone 5).
- PYTHON IMPORT GOTCHA: `python demos/x.py` puts the SCRIPT's folder on sys.path, not the
  cwd - so `from src...` fails from a subfolder. `main.py`/`ingest.py` work because they
  sit at the root. Fix: run as a module from the root -> `python -m demos.retrieval_demo`.
- Full concept write-up: Lesson 006.

### Change 006: Milestone 5 - RAG wired into the graph (SELECT complete) (23-Jun-2026)
Added a retrieve node so the agent answers from ingested docs. First multi-file feature.

- `state.py` - added `context: str` (no reducer -> REPLACE: fresh chunks each turn, no
  pollution). The ABSENCE of a reducer is the design choice (Lesson 001).
- `nodes.py` - new `retrieve` node: `similarity_search(query, k=3)` -> joined `context`.
  `generate_response` now builds a TRANSIENT prompt = [context as SystemMessage] + messages
  for the LLM call ONLY; it is NOT returned into messages, so retrieved chunks never get
  saved to history (ephemeral context - Lesson 004 "don't hoard").
- `graph.py` - flow is now process_input -> retrieve -> generate_response -> END.
- `tests/test_graph.py` - updated expected state keys to include `context`. 4 tests pass.
- Verified live: "lead engineer + budget of Project Zephyr?" -> "Mei Tanaka" + "4.2M SGD";
  "team mascot?" -> "Pascal the capybara". Facts that exist ONLY in `data/facts.md` - proof
  the agent answered from retrieval, not training.

#### Learning Notes
- SELECT pillar complete (Milestones 4+5). WRITE (M3) + SELECT now done.
- Wiring RAG in was pure nodes+state+edges (Lessons 001/003) - no new framework. The
  fundamentals paid off: a new capability was just one node + two edges + one state field.
- NUANCE (motivates conditional edges / agentic RAG, bonus #16): `retrieve` ALWAYS runs,
  even for off-topic questions. Asked "who won the 2022 World Cup?", the agent got 3
  irrelevant Zephyr chunks + the "say if unsure" rule and DECLINED rather than answering
  from training. Real RAG systems often DECIDE when to retrieve (a conditional edge)
  instead of retrieving unconditionally.

### Change 007: conditional edges / agentic RAG (concept 1.5) (23-Jun-2026)
Made retrieval conditional - the agent now DECIDES whether to look up the docs.

- `nodes.py` - added `route_after_input(state) -> str`: a small LLM call that classifies
  the query as 'retrieve' (needs the Zephyr KB) or 'skip' (general knowledge / chit-chat).
  Lenient parse; prints `[router] <choice>` for visibility.
- `graph.py` - replaced the static `process_input -> retrieve` edge with
  `add_conditional_edges("process_input", route_after_input,
  {"retrieve": "retrieve", "skip": "generate_response"})`. Flow now BRANCHES.
- 4 tests still pass (the change is wiring, not new state).
- Verified: "lead engineer of Project Zephyr?" -> retrieve -> Mei Tanaka; "what is 2+2?" /
  "capital of France?" -> skip -> answered directly. The always-retrieve over-grounding gone.

#### Learning Notes
- Static edge = fixed track; conditional edge = a router function reads state and PICKS the
  next node. The same mechanism pointed BACKWARD makes loops (basis of iterating agents).
- KEY LESSON (on-thesis): the router is itself an LLM call, so its decision is only as good
  as the CONTEXT we give it. We passed only state["query"], so the ambiguous follow-up
  "what is the team mascot?" misrouted to skip (no way to know "team" = Zephyr). Saying it
  explicitly ("...mascot for Project Zephyr?") routed correctly -> Pascal. Bad context in ->
  bad decision out, applied to the router itself.
- TRADEOFF (no free lunch): always-retrieve is robust but wasteful / over-grounds; agentic
  routing is efficient but can misroute ambiguous queries. Choose per app. Deeper fixes
  (conversation-aware routers, rerankers) = bonus #16 (agentic RAG).

### Design note: Milestone 6 (Compress) - agreed design (23-Jun-2026)
Decisions settled in discussion before building. (Build pending - paused.)

- TRIGGER (v1): message count. Compress when len(messages) > COMPRESS_AT (~12). Later:
  promote to TOKEN-based (the deferred Lesson 002 "measure tokens"). The trigger lives in
  ONE router function, so swapping the metric is a one-function change.
- TOKEN GAUGE (now): print tokens used each turn from
  `response.usage_metadata["input_tokens"]` (Ollama's real prompt_eval_count), plus % vs
  the window. A dashboard now; promote it to the trigger later.
- WINDOW (num_ctx): do NOT hardcode. Two numbers exist:
  (1) model MAX context - queryable at runtime via `ollama.show(model)` -> a
      `*.context_length` field (llama3.2 = 131072);
  (2) EFFECTIVE num_ctx - what Ollama actually runs with; OUR choice, and if unset it
      defaults LOW (~4096), NOT the model max.
  The budget that matters is (2). Plan: at startup detect the max (1) at runtime, then set
  `num_ctx = min(max, a configurable cap)` on ChatOllama and use that as the budget -
  runtime-derived, not hardcoded, but known. Only needed for the gauge / token trigger.
- SUMMARY STORE: keep the existing `scratchpad` field (NO rename). Understand that it now
  holds a running summary that the compress node refreshes periodically. generate_response
  injects it into the prompt and stops its old accumulate-responses line.
- KEEP VERBATIM: system + last KEEP_RECENT messages, KEEP_RECENT a CONFIG constant
  (env-overridable), default 6 (= 3 turns). COUPLING: the trigger threshold must be GREATER
  than the keep window, or compression keeps everything and does nothing (hysteresis). So
  COMPRESS_AT (~12) > KEEP_RECENT (6).
- CONFIG CONSTANTS (config.py, env-overridable): COMPRESS_AT, KEEP_RECENT, num_ctx cap.

Build steps (when resumed): A) rewrite generate_response (inject summary, drop old
scratchpad line, add token gauge); B) add compress node (summarise + RemoveMessage);
C) add route_compress router; D) graph: compress node + 2nd conditional edge; E) run a
long chat, watch the message count shrink and the summary preserve early facts.
Full concept: Lesson 008.

### Change 008: Milestone 6 - COMPRESS (summary-buffer) (24-Jun-2026)
History now self-compresses: summarise old turns into scratchpad, delete the originals,
keep the agent within budget. The project's namesake pillar. Typed by hand.

- `config.py`: runtime context-window detection (`ollama.show` -> `*.context_length`,
  capped at NUM_CTX_CAP=8192) set as `num_ctx` on the LLM; config constants COMPRESS_AT=12,
  KEEP_RECENT=6 (env-overridable). The detected CONTEXT_WINDOW is the gauge denominator.
- `nodes.py`:
  - generate_response: injects the scratchpad SUMMARY + RAG context as transient system
    messages (not saved); prints a [tokens] gauge (usage_metadata input_tokens / window);
    stopped owning scratchpad.
  - compress: summarise messages[1:-KEEP_RECENT] (+ previous summary) -> scratchpad;
    RemoveMessage the old ones. Keeps system + last KEEP_RECENT verbatim.
  - route_compress: 'compress' if len(messages) > COMPRESS_AT else 'end' (the v1
    message-count trigger; promote to token-based later = one-function change).
- `graph.py`: added the compress node + a 2nd conditional edge after generate_response.
- Verified live (7-turn chat): at 13 messages compress fired, removed 6, count dropped to
  9; turn 7 "what is my name + number?" -> "Andre, 42" recalled FROM THE SUMMARY though the
  originals were deleted. Write/Select/Compress trilogy all working.

#### Learning Notes
- The compress ENGINEERING worked first try (summary captured "Andre / 42", removed old,
  count shrank) - but recall first FAILED. Debugging (reading the saved scratchpad) proved
  the summary was correct and injected; the MODEL disowned it, reading "Andre's name" as
  third-party trivia ("someone named Andre... I'm starting fresh").
- FIX = FRAMING, not engineering: wrapping the summary as "this is an ongoing conversation;
  treat the summary as established facts about the user you are talking to" flipped llama3.2
  from disclaiming to answering "Andre, 42". Same data, different frame, opposite behaviour.
- Context engineering is not only WHAT is in context but HOW it is framed so the model knows
  how to use it. (See Lesson 008 addendum.)
- Two more "replace became add" slips (the graph edge; the context branch overwriting
  extras) - a recurring personal error pattern; worth a quick re-read after every refactor.

### Next Steps
- [x] Install LangGraph
- [ ] Set up Langfuse (cloud or self-hosted) and confirm a trace appears
- [ ] Build a simple agent with memory scratchpad
- [x] Install pytest and run the real test suite (4 tests passing, 23-Jun)
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

Progress: [x] 1 system prompt | [x] 2 multi-turn loop | [x] 3 persistent memory | [x] 4 ChromaDB ingest+retrieval | [x] 5 RAG pipeline | [x] 6 compress (24-Jun) | [ ] 7-12 pending  [+ conditional edges done, see Change 007]

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

**Milestone 6 - Bonus: Modernization** (optional, AFTER the core 1-12 ship)

The 2024-2026 paradigms the core path does not cover (prompted by a "your AI infra is
a bit 2023/24" nudge). These BUILD ON the fundamentals above - they do not replace them,
so do the core first. "Type": gap = not covered at all; evolves = newer flavour of an
existing item. Not part of the 1-12 progress tracker.

| #  | Task                       | What it is (one line)                                            | Relates to        | Type            |
|----|----------------------------|------------------------------------------------------------------|-------------------|-----------------|
| 13 | MCP (Model Context Protocol)| Universal "USB-C" standard to plug tools/data into any model    | Tools (#8)        | gap             |
| 14 | Prompt caching             | Cache the static prefix so you don't re-pay it every call        | Lesson 002 budget | gap (cost lever)|
| 15 | Long-context vs RAG        | ~1M-token windows mean "put it all in context" can beat RAG      | Select (#4-5)     | reframes        |
| 16 | Agentic RAG + rerankers    | Agent decides WHAT to retrieve; reranker re-scores chunks        | Select (#4-5)     | evolves         |
| 17 | LLM-as-judge evals         | Use an LLM to SCORE another LLM's output, systematically         | Eval (#10)        | evolves         |
| 18 | Server-side compaction     | Provider auto-compacts context (managed version of Compress)     | Compress (#6)     | evolves         |
| 19 | Multimodal (vision)        | Models that take IMAGES (screenshots, PDFs, charts), not just text| -                | gap             |
| 20 | Computer use / GUI agents  | Agents that operate a screen (click, type, screenshot)           | beyond Tools (#8) | gap             |
| 21 | Agent Skills               | Package task-specific instructions loaded only WHEN relevant     | scaling prompts   | gap             |

Priority if revisiting: 14 (prompt caching) and 15 (long-context vs RAG) are the two with
the most immediate practical payoff; 13 (MCP) is the headline "modern" one. Rest are
breadth. Full radar discussion: see the 23-Jun chat / Lesson 006 context.


### C. Capstone project: multi-agent code-builder

Reframing (24-Jun): Milestones 1-6 were FOUNDATIONS - learning each context pillar on a
simple chat/RAG agent (langgraph-app/). The capstone APPLIES them: a multi-agent system
that writes + tests Python code. It does not replace the thesis (context engineering) - it
FULFILS it, and it absorbs the remaining milestones into one coherent product:

| Milestone        | Role in the capstone                          |
|------------------|-----------------------------------------------|
| 7  ISOLATE       | the multi-agent architecture itself           |
| 8  Tools         | agents run code / read + write files          |
| 9  Guardrails    | sandbox for executing LLM-written code        |
| 10 Tests + eval  | the QA agent's tests ARE the eval             |
| 11 API server    | expose the builder as a service               |
| 12 Deploy        | run it for real                               |

The capstone is a NEW app (new module) that reuses the patterns proven in langgraph-app.
Built in tight stages:
- v1 (crawl): sequential single pass - orchestrator -> coder -> QA -> report. No loop.
- v2 (walk):  add the fix-loop (QA fails -> orchestrator -> coder iterates) + max-iteration
              brake + user-intervention when stuck.
- v3 (run):   parallelise coder/QA from the spec; add a research/spec agent.

Domain port (later): the same skeleton -> a BACKTESTER agent (strategy + instrument in,
data via yfinance, QA computes Sharpe / max drawdown / turnover + correctness tests).
Generic builder FIRST because the success signal (tests pass) is cleanest; the skeleton +
debugging lessons port to the backtester with little friction.
Key boundary: "correct" (the loop's job) is NOT "profitable" (never the loop's job).

Detailed design (agents, boundaries, sandbox, state, decisions): see docs/capstone.md.
