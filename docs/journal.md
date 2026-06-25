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

### Change 009: Capstone v1 Phase A - the three agents (24-Jun-2026)
Scaffolded the code-builder capstone and built its three ISOLATED-context agents. Typed by hand.

- New module `code-builder/` (separate app, reuses langgraph-app patterns):
  - `src/state.py` - `BuilderState` (request/spec/code/tests/test_result/status). Note: NO
    `messages` field - the agents do NOT share a conversation; each builds its own. That
    absence IS the isolation.
  - `src/config.py` - `get_llm()` at temperature 0.2 (code wants determinism, not creativity).
  - `src/agents.py` - orchestrator (request->spec), coder (spec->code), QA (spec->tests).
    Each builds its OWN message list (own prompt + scoped input) and returns ONLY its artifact.
- DESIGN DECISION (Andre): QA writes tests from the SPEC ALONE (TDD), not spec+code - so the
  tests verify CORRECTNESS independently and can catch coder bugs (spec+code would bias tests
  to confirm the code). Better than the original draft. Updated capstone.md.
- Verified each agent in isolation (`python -m src.agents`):
  - orchestrator -> a clean CONTRACT (name/signature/edge cases/examples), no leaked code.
  - coder -> clean fence-free Python.
  - QA -> an independent pytest suite importing `from solution import ...`.
- THE PAYOFF (bug caught before we even ran tests): the code returns False for "" (because
  "".isalnum() is False), but the spec + QA tests say "" is a palindrome (True). The
  independent tests WILL catch the coder bug - the exact scenario the architecture exists for.
  Predicted: 5 pass / 1 fail (empty string) when we run the sandbox (Step 5).

#### Learning Notes (full write-up: Lesson 011)
- TDD-from-spec makes QA an ADVERSARY of the code, both answering to the spec -> catches bugs.
- Independent code + tests can DISAGREE; spec ambiguity becomes real divergence -> spec
  precision matters even more (Lesson 010 IN-boundary, sharpened).
- POSITIVE prompts beat NEGATIVE ("output ONLY these 4 fields" stopped the orchestrator
  leaking an implementation; "do NOT write code" did not - Lesson 002).
- LLM output needs CLEANING before execution (strip ```fences``` or it is a syntax error).
- RELATIVE imports (`from .config`) need `python -m src.agents`, not `python agents.py`.
- Build slips: string-literal "ORCHESTRATOR_PROMPT", missing comma, print-without-payload.

### Change 010: Capstone v1 LOCKED - sequential graph + user input (24-Jun-2026)
v1 (crawl) complete: a sequential multi-agent code-builder, wired as a LangGraph graph,
taking a USER request and producing tested code with an objective verdict.

- `src/graph.py` - `build_graph()`: nodes write_spec -> write_code -> write_tests ->
  run_sandbox -> END (run_sandbox is a non-LLM node wrapping the pytest runner; the 3 LLM
  agents are already graph nodes - state -> dict). Plain sequential edges, no checkpointer.
- `main.py` - entry point: request via `input()` (user-driven, with a default fallback),
  `app.invoke({"request": ...})`, prints spec/code/tests/status from the FINAL state.
- INSIGHT: `invoke()` returns the accumulated FINAL STATE, which holds EVERY artifact each
  node wrote (request/spec/code/tests/test_result/status) - no need to capture step-by-step
  like the manual chain. BuilderState collects them; the graph hands you one object. (This
  also justified the graph over manual chaining - v2's fix-LOOP is one conditional edge,
  vs a hand-rolled while-loop. Lesson 007: a loop is a conditional edge pointing back.)
- Verified live: typed "please write function to check if string is palindrome" ->
  spec -> code (`return s == s[::-1]`) -> tests -> sandbox -> passed: True, status: ok.

v1 = crawl DONE. Next: v2 (walk) - the fix-loop (QA fails -> coder iterates) via a
conditional edge + max-iteration brake + user intervention when stuck.

### Change 011: Capstone v2 - fix-loop wired (25-Jun-2026)
The coder now ITERATES on test failures (the orchestration loop) with a brake. Typed by hand.

- `state.py`: added `attempts: int`. `config.py`: `MAX_ATTEMPTS=3` (env-overridable) = the brake.
- `agents.py`: write_code is now RETRY-AWARE - on a retry it gets spec + its previous code +
  the failures, increments `attempts`, returns {code, attempts}. (Failures flow
  run_sandbox -> state -> write_code: the feedback path that makes the loop converge.)
- `graph.py`: REORDERED to write_spec -> write_tests -> write_code -> run_sandbox (tests
  FIRST = TDD, fixed target), then a conditional edge `route_after_tests`:
  passed -> done(END); failed & attempts>=MAX -> done (give up); else -> retry(write_code).
  The conditional edge pointing BACK to write_code IS the loop (Lesson 007). v2 was one
  router + one conditional edge + a reorder - NOT a rewrite. The graph investment paid off.
- Bug caught (the recurring "computed but not wired to output" pattern): write_code built
  `task` and `attempts` but still passed `state["spec"]` and returned only `{code}` - fixed
  to pass `task` and return `attempts`.

Agreed plan: 1) test the loop (trigger a retry); 2) add user-intervention on stuck (the last
v2 piece - currently "give up" just logs + ENDs + reports failure, no interactive ask);
3) add a while True (multi-build) + maybe a checkpointer.
Design note: the builder is a ONE-SHOT TASK (run-to-completion), not a conversation - so the
while-loop + checkpointer are OPTIONAL enhancements, not required. (See Lesson 012.)

### Change 012: Capstone v2 - tested the loop; the debug saga (25-Jun-2026)
Step 1 of the v2 plan DONE: ran the builder on "int -> roman numeral" to trigger a retry.
What the 3 live runs ACTUALLY verified (Andre's catch - the first draft overclaimed): the
loop's PLUMBING only - retry fires, `attempts` increments, the brake stops at MAX_ATTEMPTS.
They did NOT verify the loop's VALUE (a coder taking a real failure and converging
red -> green): runs 1-2 fed it unfixable/mis-routed bugs, run 3 one-shot (the loop never
engaged). It took 3 runs to go green, and EVERY failure was UPSTREAM of the coder - the
coder's code was correct on all 3 runs.
The VALUE was proven SEPARATELY with a deterministic test (check_fixloop.py): plant a known
buggy solution + good tests, drive the retry branch directly -> the coder read the logs and
fixed it (planted False -> True, even added n<2 + sqrt). Plumbing verified live; convergence
verified by the planted-bug test. (See Lesson 013, esp. Lesson D: validating a loop means
separating "the wiring runs" from "the wiring produces value".)

- Run 1 (red, gave up at 3): (a) the SPEC hallucinated an out-of-range edge case
  `0 -> ''` that contradicts the "1..3999" range; (b) the QA used `pytest.raises` but never
  wrote `import pytest` - the test file would not even load.
- Run 2 (red, gave up at 3): fixes a+b worked (spec consistent, `import pytest` present),
  but the QA found new ways to break the file: (c) no `from solution import int_to_roman`
  (NameError); (d) it QUOTED the call - `assert 'int_to_roman(1)' == 'I'` (compares text).
- Run 3 (GREEN on attempt 1): genuine pass - 0/4000 raise ValueError, 3.14 raises TypeError,
  numerals correct.

3 prompt fixes, ALL upstream (orchestrator + QA), NONE to the coder:
- orchestrator: edge cases MUST be consistent with the input range (no invented
  out-of-range return values).
- QA: include EVERY import; imperative "Begin the file with `import pytest` +
  `from solution import <fn>`"; plus a POSITIVE assertion example.

Findings (-> Lesson 013):
- "red != broken code" - a red verdict is only as trustworthy as the spec + the harness.
  Here the harness author (QA) was the weak link: writing a RUNNABLE test file is
  structurally harder than the function (one bad import/typo fails the whole suite to load).
- weak-verb bug: "Assume `from solution import`" was read as "may assume it's available";
  "Begin the file with ..." fixed it. Imperatives + positive examples beat
  assumptions/negatives (the positive-prompt rule, confirmed a 3rd time).
- coverage still light (no big composite like 3999 -> MMMCMXCIX) = Lesson 011 still lurks.

Next: step 2 (user-intervention on stuck), then step 3 (while-True multi-build).

### Design note: v2 plan refined - the build ledger + decidable routing (25-Jun-2026)
A discussion (no code yet) that reshaped the v2 roadmap. Full design in capstone.md; concept
in Lesson 013 (+ its decidability addendum). The key moves:
- BUILD LEDGER (Andre's idea): a state field recording who-wrote-what-and-what-resulted, so a
  failure traces to its author. In v2 it fuels a TRUTHFUL stuck-report to the HUMAN (the
  current state OVERWRITES code/test_result each retry, losing the per-attempt trajectory -
  the ledger keeps it; e.g. "same error 3x" = not the code's fault). Reused in v3 for
  auto-routing. NOT a checkpointer (one-shot task, Lesson 012) - an explicit in-state ledger.
- PRINCIPLE (Andre): the system ASSISTS, does not GUARANTEE; on giving up, be truthful about
  what it tried and hand control to the user. The human is the judge in v2.
- DECIDABILITY is the routing rule, not the v2/v3 label: a test file that won't LOAD (pytest
  exit 2/5) is DECIDABLE -> auto-route to the QA (v2, no judge, no ping-pong). An assertion
  failure (exit 1) is UNDECIDABLE (code/test/spec?) -> v2 escalates to a human, v3 builds a
  judge. Ping-pong + losing the fixed-target TDD property are costs of the v3 judge ONLY.
- Refined v2 steps: 2) ledger + user-intervention on stuck; 2.5) decidable QA-route; 3)
  while-True multi-build. (Corrected a mis-scope: the ledger is v2, not v3.)

### Change 013: Capstone v2 step 2 - ledger + user-intervention DONE (25-Jun-2026)
The builder no longer dies silently when stuck: it shows the full trajectory and lets the
user (the judge) steer. Typed by hand. Full concept: Lesson 014.

- LEDGER: `state.py` adds `ledger: Annotated[list, add]` (operator.add = list concat = append;
  the humble cousin of add_messages, Lesson 001) + a `BuildEvent` dataclass
  (author/artifact/content). Each node appends ONE event (write_spec/write_tests/write_code/
  run_sandbox). This keeps the per-attempt trajectory the final state used to OVERWRITE.
- INTERVENTION LOOP: `main.py` wraps the graph in an outer `while True`. The graph stays a
  pure one-shot invoke (Lesson 012 holds - the loop lives in plain Python, not the graph).
  On a stuck build (status=="failed" after the brake) it prints a truthful stuck-report FROM
  the ledger (spec, tests, each coder attempt + each sandbox verdict), then asks the user:
  feedback / retry / quit.
- FEEDBACK BUG + FIX: the first cut checked `if choice == "e"`, so free-text typed at the
  prompt was DISCARDED -> blind retries; the apparent "improvement" was non-determinism, not
  steering (Lesson 009: a green build masked a DEAD intervention). Fixed by ACCUMULATING
  free-text feedback and AUGMENTING the request each round (keep the task, add corrections).
- VALIDATED live: a 3-round whack-a-mole on int->roman that converged to a GENUINE green
  (real attempt1->2). Each correction killed its targeted bug the next round -> feedback now
  genuinely steers.
- BUG FIXED (prompt-echo): the orchestrator copied ORCHESTRATOR_PROMPT bullet-3 verbatim INTO
  the spec. Cause: a rule was INLINED in the field list, so the model echoed the how into the
  what. Fix: separate FORMAT (sections 1-4) from GUIDANCE (notes), lean on the POSITIVE
  "Output ONLY sections 1-4" (Lesson 011: positive > negative).
- KNOWN ISSUE PARKED: `input()` reads ONE line, so multi-line feedback truncates + the
  remainder leaks into the next prompt (the `^R` mangling). Workaround: single-line feedback.
  Real fix (deferred to step 3, when we are in main.py anyway): read-until-blank-line.
- TEST TECHNIQUE: to exercise the STUCK path you must manufacture failures - weaken an agent
  prompt (Andre temporarily regressed QA_PROMPT; since restored) or set MAX_ATTEMPTS=1. A
  healthy pipeline one-shots and never shows the path you are trying to test (Lesson 013-D).

### Change 014: Capstone v3 step 1 + 3 - surgical, stateful build loop (26-Jun-2026)
The intervention loop stopped whack-a-moling: it now PERSISTS the build and edits ONLY the
artifact the human targets. Typed by hand. Concepts: Lessons 015 (memory+targeting), 016
(model ceiling). Full design: capstone.md "v3".

- STATE: added `fix_target: str` ("" fresh | spec | tests | code) + `feedback: str`.
- GRAPH: a DISPATCHER conditional edge from START routes on fix_target (Lesson 007), replacing
  the fixed entry; an after_write_tests fork sends a TESTS fix straight to run_sandbox (REUSE
  the code) else on to write_code. route_after_tests now LOGS its decision + the failing tests.
- AGENTS: write_tests is EDIT-aware (fix_target=="tests": prev tests + feedback -> edit, keep
  the good ones); write_code now also reads the human feedback (fix_target=="code"). [plan]
  lines announce which agent edits which file.
- MEMORY (main.py): the intervention loop carries {spec,tests,code,test_result} into the next
  invoke (the Lesson 003 chat-loop pattern, for build-state) + asks "fix which?
  [spec/tests/code]". Feedback is per-fix now (the artifacts persist, so no text re-accumulation).
- PROVEN: the TESTS path converges - spec stays BYTE-IDENTICAL (pinned), the QA edits the suite
  surgically, the coder cascades only if the corrected tests expose a code bug. The
  "v2 couldn't, v3 can" moment (Lesson 015 dissolved).
- OPEN: step 2 (spec EDIT-aware) NOT built - a "spec" fix still regenerates. The CODE-feedback
  path is wired + confirmed reaching the coder (debug-print method, Lesson 016) but never
  converged end-to-end (3B too weak; 8B already correct, so no code-fix was needed).
- MODEL CEILING (Lesson 016): the bottleneck is now MODEL quality PER AGENT, not the wiring.
  3B coder too weak; swapping to llama3 8B FIXED the coder but made the orchestrator chatty and
  leak the code into the spec. -> points at PER-AGENT models as the next high-value move.

### Change 015: Capstone v3 step 2 - spec edit-aware; V3 COMPLETE (26-Jun-2026)
The orchestrator is now EDIT-aware too, completing the surgical trilogy (all three agents edit
their OWN previous artifact + feedback). Typed by hand. Concept: Lesson 015 addendum.

- agents.py: write_spec EDIT mode (fix_target=="spec": prev spec + feedback -> edit, keep the
  good parts) - mirrors write_tests/write_code. Also removed the temp CODER-TASK debug print
  (served its purpose: proved the wiring, Lesson 016).
- PROVEN live: a contradictory-spec build (empty = palindrome AND raises) -> picked "spec" ->
  "[orchestrator] editing spec (human-directed)" -> the new spec was NEAR-IDENTICAL to the old
  with only the TARGETED rule changed (the proof it EDITED, not regenerated) -> cascaded ->
  green on attempt 1.
- V3 COMPLETE: all three fix_targets are surgical; the human-targeting + memory loop converges
  across spec/tests/code. The Lesson 015 whack-a-mole is dissolved.
- CAVEAT (Lesson 016): edit QUALITY is model-gated - the orchestrator only PARTIALLY applied
  the feedback (kept one contradictory line); it converged only because no test discriminated
  on it (Lesson 011). Wiring proven; quality wants a stronger / per-agent model.

Next: C (ship-it) - M11 FastAPI wrapper + M12 deploy + confirm Langfuse.

### Change 016: M10 eval + import-guard + the self-healing supervisor (26-Jun-2026)
Built the eval (M10), used it to diagnose, then built the auto-judge supervisor (v3+). Typed by
hand. Concepts: Lessons 016 (model ceiling), 017 (the judge).

- EVAL (M10): `src/evals.py` - a "backtest" running a fixed task suite ONE-SHOT (no human),
  reporting pass-rate + attempts. First numbers were poor: llama3=0%, llama3.2=20%.
- DIAGNOSIS: nearly all failures were the QA's TEST FILE (won't-load, missing import, invented
  requirements), not the code - and the coder-only loop can't fix QA bugs (Lesson 013-C).
- STRUCTURAL GUARD (enforce, don't ask): `sandbox.py` auto-prepends `from solution import *` to
  the test file if the QA forgot the import - killed the NameError class deterministically.
- MODEL CEILING (Lesson 016): same architecture, swap to qwen2.5-coder -> 80% (100% on a lucky
  run). PROVED the bottleneck was model capability, not the design. qwen2.5-coder set as default.
- AUTO-JUDGE SUPERVISOR (v3+, Lesson 017): a `judge` node (decidable load-error tier + LLM tier)
  writes fix_target+feedback; graph rewired to run_sandbox -> route_after_sandbox
  {pass/brake -> done, else -> judge} -> dispatch -> the culprit agent -> loop. ONE brake
  (`attempts`, now incremented in the JUDGE not the coder) = 3 auto-rounds, then escalate to human.
- VALIDATED: judge classifies code vs tests (check_judge.py); routed -> code on a real code bug
  (gcd, escalated after 3); routed -> tests on a QA bug (dropped `import pytest`) and the QA
  RECOVERED it autonomously in 1 round - the value the coder-only loop never had.
- NOTE: the high temperature + loose QA prompt used to TRIGGER failures are TEST FIXTURES -
  restore temp 0.2 + the strong QA_PROMPT for production.

### Next Steps
- [x] Install LangGraph
- [ ] Set up Langfuse (cloud or self-hosted) and confirm a trace appears   <- still open
- [x] Build a simple agent with memory scratchpad (done - foundations agent)
- [x] Install pytest and run the real test suite (4 tests passing, 23-Jun)
- [x] Add a conditional edge / loop (Change 007; capstone uses them too)
- [x] Add a checkpointer for persistent state across runs (Change 004)
- [x] Capstone v2: the fix-loop (wired Change 011; tested + validated Change 012)
- [x] Capstone v2 step 2: ledger + user-intervention on stuck (Change 013, Lesson 014)
- [x] Capstone v2 step 3: while-True multi-build (run_one_build + "build another?")
- [x] Capstone v2 step 2.5 (attempt): surgical feedback-to-QA -> revealed Lesson 015
- [x] Capstone v3 step 1 (tests path) + step 3 (code feedback): BUILT; tests-path PROVEN (Change 014)
- [x] Capstone v3 step 2 (spec EDIT-aware): V3 COMPLETE - all 3 fix_targets surgical (Change 015)
- [x] M10 eval harness + structural import-guard; model-ceiling found, qwen2.5-coder=80% (Change 016)
- [x] v3+ self-healing supervisor: auto-judge routes each failure to the culprit agent (Change 016, Lesson 017)
- [ ] SHIP-IT (C): M11 FastAPI wrapper + M12 deploy + confirm Langfuse   <- NEXT
- [ ] restore production settings: temp 0.2 + strong QA_PROMPT (the test fixtures)
- [ ] (later) PER-AGENT models (the Lesson 016 model-ceiling fix: strong coder, terse orchestrator)

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

Progress - FOUNDATIONS (langgraph-app): [x] 1 system prompt | [x] 2 multi-turn | [x] 3 memory
| [x] 4 ChromaDB | [x] 5 RAG | [x] 6 compress | [x] conditional edges (Changes 002-008).
  -> All four thesis pillars (Write/Select/Compress/Isolate) now built.
Progress - CAPSTONE (code-builder, fulfils milestones 7-12): [x] 7 ISOLATE = multi-agent v1
(Change 010); v2 COMPLETE - fix-loop (Changes 011-012), ledger + user-intervention
(Change 013), multi-build (step 3), surgical feedback-to-QA (step 2.5 attempt -> Lesson 015).
v3 COMPLETE (Changes 014-015) - all 3 fix_targets surgical (tests/code/spec edit-aware),
converges. Also DONE (Change 016): M10 eval + structural guard + v3+ SELF-HEALING supervisor
(auto-judge routes to the culprit); qwen2.5-coder = 80-100% (Lessons 016-017). 10 Tests+eval
is now solidly covered. Next: SHIP-IT (M11 API + M12 deploy + Langfuse). Deferred: PER-AGENT
models (the model-quality ceiling).
[x] 10 Tests+eval = QA tests + foundations unit tests + the M10 eval harness (src/evals.py).
Partial: 8 Tools (code exec via sandbox, no ToolNode/ReAct), 9 Guardrails (sandbox + brake, no
injection/validation layer). Pending: 11 API, 12 Deploy.
Loose ends parked: Langfuse trace unconfirmed; compress token-trigger; summary/context test.

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
