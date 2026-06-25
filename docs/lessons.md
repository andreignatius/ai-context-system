# Lessons

A running set of concept notes, written up as we learn them.
Each lesson is self-contained so it can be re-read later.

---

## Lesson 001: LangGraph state & reducers (22-Jun-2026)

### The big picture
- **State** = a typed dictionary that flows from node to node through the graph.
  Ours (`langgraph-app/src/state.py`):
  ```python
  class AgentState(TypedDict):
      messages: Annotated[list, add_messages]
      scratchpad: str
      query: str
  ```
- **Node** = just a function. It receives the state and returns ONLY the piece it
  changed (a "partial update"), not the whole state.
- **Reducer** = a small function that decides how to merge a node's new piece into
  the old state, field by field.

### messages vs. scratchpad
Both are "memory", but they hold different things:

| | `messages` | `scratchpad` |
|---|---|---|
| Holds | Full conversation, turn by turn | Short working notes |
| Grows by | Adding each new message | Whatever we decide to jot |
| Analogy | The meeting transcript | The post-it note |

- `messages` keeps the full history, verbatim.
- `scratchpad` is short working memory (placeholder for now; later it is where the
  "Compress" idea lives - summarising the long transcript into a short note).

### What a reducer is
A reducer always has this shape:
```python
def reducer(old_value, new_value):
    return combined_value
```
Two inputs (what was there, what just came in), one output (what to keep).
It is the same idea as a fold / `functools.reduce` over a series.

For `messages` we want to APPEND, so the reducer is basically:
```python
def add_messages(old_list, new_list):
    return old_list + new_list   # old first, then new (plus some smart extras)
```
`add_messages` also: coerces dicts/strings into Message objects, and updates a
message in place if a new one shares the same id.

### How LangGraph knows which reducer to use
You tell it in the state definition with `Annotated`:
```python
messages: Annotated[list, add_messages]   # field is a list; merge with add_messages
scratchpad: str                           # no reducer -> default = REPLACE
```
- Reducer attached  -> that function decides the merge (here: append).
- No reducer        -> default behaviour = overwrite (last write wins).

### The key "aha": the old+new step is hidden
A node returns ONLY the new piece. It does NOT contain a
`messages = messages + new` line. LangGraph runs the reducer for you, invisibly,
the moment the node returns.

OLD style (we wrote the merge by hand, in `simple_agent.py`):
```python
def process_input(state):
    messages = state.get("messages", [])   # get the OLD list
    messages.append(state["query"])        # we did old + new here
    return {"messages": messages}          # return the whole thing
```

NEW style (the reducer does old+new for us, so the node shrinks):
```python
def process_input(state):
    return {"messages": [HumanMessage(content=state["query"])]}   # just the NEW piece
    # LangGraph runs add_messages(old, new) behind the scenes
```
The old+new did not disappear - it moved OUT of the function and INTO LangGraph.

### Full workflow trace
Running `main.py` with
`{"query": "What is 2+2?", "messages": [], "scratchpad": ""}`.
The hidden reducer step is marked [R].

```
START
  messages   = []
  scratchpad = ""

Step 1 - node process_input
  returns: {"messages": [Human("What is 2+2?")]}
  [R] messages   : add_messages([], [Human]) -> [Human]     (append)
  [R] scratchpad : not returned -> unchanged -> ""
  state: messages = [Human]        scratchpad = ""

Step 2 - node generate_response
  calls the LLM, gets AI("4")
  returns: {"messages": [AI("4")], "scratchpad": "4"}
  [R] messages   : add_messages([Human], [AI]) -> [Human, AI]   (append)
  [R] scratchpad : no reducer -> REPLACE -> "4"
  state: messages = [Human, AI]    scratchpad = "4"

END -> main.py prints messages[-1].content -> "4"
```

Same workflow, two different rules:

| After   | messages (reducer = append) | scratchpad (no reducer = replace) |
|---------|-----------------------------|-----------------------------------|
| Start   | []                          | ""                                |
| Step 1  | [Human]                     | "" (untouched)                    |
| Step 2  | [Human, AI]                 | "4" (overwritten)                 |

`messages` keeps GROWING; `scratchpad` only holds the LATEST value.

### One-sentence summary
A node returns only the new piece; a reducer is the little function that merges it
into the old state per field - `messages` is set to append (so history grows),
`scratchpad` has no reducer so it is replaced.

### Why reducers matter beyond convenience
- Avoids manual read-modify-write in every node.
- Works correctly when two nodes write the same field (e.g. parallel branches):
  LangGraph folds both updates through the reducer deterministically, instead of
  one mutation racing another.
- Replayable / mergeable updates are what later make checkpointing and
  time-travel possible.

### To revisit / try next
- [x] Run a tiny demo that prints the messages list growing each step.
      (Done: `langgraph-app/demos/reducer_demo.py`.)
- [ ] Give `scratchpad` its own reducer so it accumulates instead of replacing.
- [ ] See it "break and heal": a parallel node writing `messages` without vs. with
      a reducer.

---

## Lesson 002: message types & the system prompt (23-Jun-2026)

### The big picture
- Every message has a ROLE. There are three main types:
  | Type            | Role                  | Who "speaks"        | Example                         |
  |-----------------|-----------------------|---------------------|---------------------------------|
  | `SystemMessage` | setup / instructions  | you, the developer  | "You are a concise math tutor." |
  | `HumanMessage`  | the user's input      | the user            | "What is 2+2?"                  |
  | `AIMessage`     | the model's reply     | the LLM             | "4"                             |
- A "system prompt" is just a `SystemMessage` placed at the FRONT of the messages
  list. It sets the agent's role, rules, and output format for the whole session.
- We already use `HumanMessage` and `AIMessage` in `nodes.py`. `SystemMessage` is
  the one not used yet - adding it IS Milestone 1.

### The model doesn't see "objects" - it sees one string
The list of message objects is flattened by a CHAT TEMPLATE into a single block of
text, with special role-marker tokens. For a Llama-style model:
```
<|system|>
You are a math tutor.
<|user|>
What is 2+2?
<|assistant|>
```
So a "role" is just a label wrapped around your text so the model knows who said
what. The whole conversation becomes one long string; the model predicts what comes
after the final `<|assistant|>`.

### Why order matters (system goes FIRST)
- Position = meaning, because it is read top-to-bottom as one string.
- Models are TRAINED with system instructions at the very top, before any user turn.
- Put the system message last and (a) it is a shape the model rarely saw in training,
  so it is followed less reliably, and (b) it is backwards - giving the rules after
  the user already spoke.
- Rule: system first, then human/AI turns in chronological order.

### Standing vs. spoken
- `SystemMessage` = STANDING instructions: set once, applies to the whole session,
  usually one message at the front that never repeats. The user never sees it.
- `HumanMessage` / `AIMessage` = the SPOKEN dialogue: each is one turn, and they keep
  ACCUMULATING (via the `add_messages` reducer from Lesson 001).

### Token budgeting: what belongs in the system prompt?
- The context window is a FIXED budget (tokens; ~3/4 word each). Everything competes:
  ```
  [system prompt] + [history] + [retrieved docs] + [user question] + [ROOM FOR ANSWER]
  ```
- The system prompt is an ALWAYS-ON cost: re-sent on every call, paid every time.
- Decision rule (this maps onto the project thesis):
  | If the info is...                         | Put it in...          | Pillar   |
  |-------------------------------------------|-----------------------|----------|
  | ALWAYS needed (role, rules, format)       | the system prompt     | Write    |
  | SOMETIMES needed (facts, docs)            | retrieved on demand   | Select   |
  | OLD / stale (early history)               | summarized or dropped | Compress |
- Lesson: don't permanently carry what you can fetch on demand. Keep the system
  prompt for what is UNIVERSALLY true about the agent.

### Failure modes (find the sweet spot, not the maximum)
- Case A - NO system prompt: generic, inconsistent tone, no sense of role/format.
  (This is what the agent is now.)
- Case B - GOOD system prompt: short, specific, every line earns its place
  (role + format + a safety rule). ~40 tokens, big behavior change.
- Case C - BLOATED system prompt: backfires.
  - Instruction dilution: more rules -> FEWER reliably followed.
  - "Lost in the middle": models attend most to the start/end, weakest in the middle,
    so buried rules get ignored.
  - Crowds out history: a huge prompt leaves less room, so the agent FORGETS earlier
    turns - it gets "dumber" about the actual conversation.
  - Slower + costlier; long prompts also accumulate contradictions.

### Not a security boundary
A system prompt STEERS behavior but does not HARD-ENFORCE it; a clever user can talk
the model out of it. That is why Guardrails (Milestone 9) is a separate layer.

### One-sentence summary
A system prompt is a `SystemMessage` at the front of the messages list - your
standing, always-on instructions; keep it lean (it is paid every call) and push
question-specific info to retrieval.

### To revisit / try next
- [x] Build Milestone 1: add a small, Case-B-quality system prompt to the agent.
      (Done 23-Jun; verified live with a pirate-vs-friendly override.)
- [ ] Learn to actually MEASURE tokens (so budget sizes stop being abstract).

---

## Lesson 003: graph runs vs the chat loop - memory across turns (23-Jun-2026)

### The big picture: two timescales
There are TWO different "lifetimes" for the messages list, and it is easy to mix them:
- WITHIN one graph run: the `add_messages` reducer (Lesson 001) appends across NODES.
  messages grows [] -> [System, Human] -> [System, Human, AI] during one `invoke`.
- ACROSS graph runs: each new `invoke` starts from whatever messages you hand it.
  The chat loop is what carries the result of one run into the next.

The reducer accumulates inside one trip through the graph. The chat loop accumulates
across trips. Different scopes.

### Definitions
- Graph run = ONE `app.invoke(...)` = the nodes run once = one question answered.
- Chat loop = the `while True:` loop that calls `invoke` over and over, feeding state
  forward, until you type quit.

A chat loop CONTAINS many graph runs:
```
CHAT LOOP (while-loop in main.py)
|
+-- turn 1 --> ONE GRAPH RUN (app.invoke)
|               +-- node process_input
|               +-- node generate_response
+-- turn 2 --> ONE GRAPH RUN (app.invoke)
+-- turn 3 --> ONE GRAPH RUN (app.invoke)
```
So a 3-turn conversation = 3 graph runs wrapped in 1 chat loop.

Analogy: a graph run is one round trip to the kitchen (take order, cook, serve). The
chat loop is the waiter working the whole evening, remembering what the table already
ordered so the next dish fits. The kitchen only handles one order at a time.

### The line that IS the memory
In `main.py`'s `chat()`:
```python
result = app.invoke({"query": user_input, "messages": messages, "scratchpad": scratchpad})
...
messages = result["messages"]      # <-- save it back, so the NEXT run starts here
```
Feeding `messages` IN gives this run the prior conversation; saving it back OUT lets
the next run continue from here. That one assignment strings the trips together.

### The "break it on purpose" experiment
Comment out the save-back line and run the same memory test:
```python
# messages = result["messages"]
```
Result: the agent FORGETS. Turn 2 ("what is my name?") replies "our conversation just
started", because messages stays [] forever -> every turn is a fresh graph run with no
history. Same graph, same nodes, same system prompt; removing ONE line removes memory.

Conclusion: memory is not in the graph, the nodes, or the system prompt. It is the
single variable carried from one graph run to the next.

### Subtle: it is `messages`, not `scratchpad`
Of the two save-back lines, only `messages = result["messages"]` controls memory.
`scratchpad` is NOT sent to the LLM yet - `generate_response` calls
`llm.invoke(state["messages"], ...)`, so only messages reaches the model. Commenting
the scratchpad line has zero effect on this test. (scratchpad gets used later, in
the Compress milestone.) Good debugging instinct: find the exact line responsible.

### In-process vs durable
This memory is IN-PROCESS only: it lives in a Python variable, so quitting the script
wipes it. Run `python main.py` again and the agent does not know you. Remembering
ACROSS sessions needs a checkpointer that saves state to memory/SQLite, plus a
`thread_id` to resume a conversation - that is Milestone 3 (WRITE).

### One-sentence summary
The reducer appends messages within one graph run; the chat loop carries messages
between graph runs - remove that carry and every turn is a stranger; persist it to
disk (a checkpointer) and the agent remembers across sessions.

### To revisit / try next
- [x] Build Milestone 3: swap the manual `messages` variable for a checkpointer
      + `thread_id` so memory survives a restart. (Done 23-Jun - see Lesson 004.)

---

## Lesson 004: persistent memory - checkpointers & thread_id (23-Jun-2026)

### The problem
Milestone 2's chat loop kept memory in a Python variable (`messages`). That is
IN-PROCESS only: quit the script and it dies. We want memory that survives a restart.

### The two ideas
- Checkpointer = automatic save. Hand it to the graph at compile time and LangGraph
  saves the whole state after every step, to a store. You stop managing `messages`.
- thread_id = the "save slot". Each conversation is filed under its thread_id. Invoke
  with the same thread_id again and LangGraph reloads that conversation's state first.

Analogy: checkpointer = a video game's auto-save; thread_id = which save slot to load.

### How it changed the code
`graph.py`:
```python
def build_graph(checkpointer=None):
    ...
    return builder.compile(checkpointer=checkpointer)   # None = no persistence
```
`main.py` (the manual carrying is GONE):
```python
with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    app = build_graph(checkpointer)
    config = {"configurable": {"thread_id": "andre-1"}}
    ...
    result = app.invoke({"query": user_input}, config)   # only the query!
```
LangGraph reloads prior messages (keyed by thread_id), runs the turn, saves the new
state - all automatically. We deleted `messages = result["messages"]`.

### Two flavours (the two-stage build)
| | MemorySaver | SqliteSaver |
|---|---|---|
| Saves to | RAM | a `.sqlite` file on disk |
| Survives restart? | No | YES |
- Swapping one for the other changed ONLY the storage line. The loop, thread_id, and
  invoke were identical. Backends are interchangeable behind one interface (same idea
  as ChatOllama vs ChatAnthropic).

### The `with` block
`SqliteSaver.from_conn_string(...)` is a context manager, so it is used with `with`:
it opens the DB connection and GUARANTEES it is closed cleanly when the block ends.
Same pattern as `with open(file) as f:`. Good hygiene - never leak a DB connection.

### Debugging lesson: separate the SYSTEM from the MODEL
Live test: told it name+number (run A), quit, asked (run B). The chat output looked
like it FORGOT. But:
- the model's own `<think>` referenced "Andre mentioned 42", and
- reading `checkpoints.sqlite` showed 5 saved messages across both runs.
So persistence WAS working - the deepseek-r1 reasoning model just gave a confused
answer (over-thinking, primed by "I restarted my computer"). If we trusted the surface
we would have debugged the wrong layer. Check the layer that owns the behaviour.

### Context-engineering lesson (on-thesis): don't hoard context
After a few messy turns, the assistant's OWN earlier denials ("I don't remember across
restarts", "each interaction is independent") were checkpointed, then reloaded, then
reinforced - a feedback loop that made it keep denying. Persisting everything verbatim
backfired. This is exactly why the COMPRESS and SELECT pillars exist: curate what stays
in context, don't blindly keep it all. Fixed by a fresh thread_id + llama3.2.

### One-sentence summary
A checkpointer auto-saves state per step and a thread_id is the slot it saves under;
SqliteSaver makes that durable across restarts - but what you persist matters, because
the model reads its own saved history back as context.

### To revisit / try next
- [x] Time travel: use the checkpointer to rewind a conversation to an earlier state.
      (Covered conceptually - see Lesson 005.)
- [ ] Give each user their own thread_id (multi-user memory).
- [ ] Eventually: a Compress step so old/!useful turns do not pollute the context.

---

## Lesson 005: time travel - checkpoints as a git-like history (23-Jun-2026)
(Conceptual only - no build yet. Anchored to Milestone 3's checkpointer.)

### The realization
The checkpointer does not save ONE state per thread. It saves a SNAPSHOT after every
step, each with its own `checkpoint_id`. So a thread is not a single save file - it is
a whole TIMELINE of snapshots:
```
thread "andre-1":
  checkpoint 1  (after process_input, turn 1)
  checkpoint 2  (after generate_response, turn 1)
  checkpoint 3  (after process_input, turn 2)
  checkpoint 4  (after generate_response, turn 2)
  ...
```

### The git analogy (the whole concept)
| Git | LangGraph checkpointer |
|---|---|
| a commit | a checkpoint (state after one step) |
| a branch | a thread (`thread_id`) |
| `git log` | `app.get_state_history(config)` |
| `git checkout <commit>` | invoke with a past `checkpoint_id` |
| branching from an old commit | resuming from a past checkpoint |

A thread is a branch; each step is a commit on it. And like git, you can check out an
old commit and start a NEW branch from there. That is time travel.

### The two methods
- `app.get_state(config)`        -> the LATEST checkpoint. Like `git show HEAD`.
- `app.get_state_history(config)` -> EVERY checkpoint, newest first. Like `git log`.
  Each carries the full state at that moment plus its `checkpoint_id`.

To rewind, put a past checkpoint_id in the config:
```python
past = {"configurable": {"thread_id": "andre-1", "checkpoint_id": "<old-id>"}}
old_state = app.get_state(past)   # the state as it was, back then
```

### The powerful part: rewind AND branch
Resuming an invoke from a past checkpoint does NOT overwrite the future - it FORKS a new
branch from that point (like `git checkout <old-commit>` then committing). So you can ask
"what if turn 3 had gone differently?" without destroying what actually happened.

This unlocks four things:
1. Inspect/debug - exactly what was the state after step N? (We did a weak version of
   this by reading checkpoints.sqlite to debug the deepseek confusion.)
2. Undo - roll a conversation back one turn after a bad response.
3. "What if" / retry - rewind to before a bad answer, feed a different input, get a new
   branch. (Could rewind PAST the polluted denials from Lesson 004.)
4. Human-in-the-loop - pause at a checkpoint, let a human edit state, then resume.
   This is how serious agents add approval gates.

### Why it matters for this project
Time travel is infrastructure for the COMPRESS and ISOLATE pillars: when we build a
Compress step we will inspect and rewrite past state, and checkpoints make that safe and
reversible. It also directly answers the Lesson 004 pollution bug - rewind past the bad
turns instead of being stuck with them.

### One-sentence summary
A checkpointer saves a snapshot after EVERY step, so a thread is a git-like history of
states; `get_state_history` is your `git log`, and rewinding to a `checkpoint_id` then
resuming forks a new branch - giving inspect, undo, what-if retries, and human-in-the-loop.

### To revisit / try next
- [ ] Run `get_state_history` on thread andre-1 to SEE the real timeline.
- [ ] Rewind to a past checkpoint and resume with a different input (make a branch).

---

## Lesson 006: RAG and embeddings - the SELECT pillar (23-Jun-2026)
(Conceptual, before building Milestone 4.)

### The problem RAG solves
The agent only knows its training data + what is in the context window. It does NOT
know YOUR documents (notes, PDFs, research). You also cannot stuff them all into the
system prompt - that blows the token budget (Lesson 002). Lesson 002's table already
named the fix: info that is SOMETIMES needed should be RETRIEVED on demand = SELECT.

### RAG in one sentence
Retrieval-Augmented Generation = before answering, RETRIEVE the few most-relevant
document chunks and inject only those into the prompt, then GENERATE. Fetch what you
need instead of carrying everything.

### The magic ingredient: embeddings
An embedding turns text into a VECTOR (a list of numbers, e.g. 384 dims) that captures
MEANING. Similar meaning -> nearby vectors; unrelated -> far apart. Quant framing:
points in high-dimensional space, closeness = cosine similarity (angle between vectors).
"car" and "automobile" end up nearly parallel; "car" and "banana" nearly orthogonal.
So semantic search = nearest-neighbour search in vector space.

### The vector store (ChromaDB)
A database built for this: stores all chunks as embeddings and answers "give me the k
chunks whose vectors are closest to THIS query vector". The fast nearest-neighbour index.

### The pipeline has two phases
```
INGEST (offline, once):
  documents -> split into chunks -> embed each chunk -> store vectors in ChromaDB
QUERY (online, every question):
  question -> embed it -> find k nearest chunks -> put them in the prompt -> generate
```

### The three quality levers (where RAG lives or dies)
1. CHUNKING - how you split the docs. Too big = imprecise + wastes budget; too small =
   fragments ideas. Use OVERLAP (~50 tokens) so ideas spanning a boundary survive.
   This is the #1 lever - most "RAG doesn't work" is actually bad chunking.
2. EMBEDDING MODEL - a separate small model (e.g. nomic-embed-text via Ollama, or
   sentence-transformers). CRITICAL: use the SAME model for ingest AND query - vectors
   are only comparable within one model's space (else it is miles vs kilometres).
3. k - how many chunks to retrieve. Too small -> miss the answer; too large -> dilute
   with noise, blow the budget, trigger "lost in the middle" (Lesson 002). Sweet spot
   k=3-5. Same "find the sweet spot, not the maximum" shape as system-prompt sizing.

### How it slots into our graph
It is just a NEW NODE. Today: process_input -> generate_response. RAG adds a retrieve
node in the middle:
  process_input -> retrieve -> generate_response
`retrieve` embeds the question, pulls top-k chunks from Chroma, adds them to state;
`generate_response` then answers WITH those chunks in context. Same nodes/state/edges
from Lessons 001-003 - RAG is not a new framework, just one more node.

### One-sentence summary
RAG = the Select pillar: embed your docs into meaning-vectors, store them in a vector
DB (Chroma), and at query time fetch only the few NEAREST chunks to inject - added as a
retrieve node before generate_response. Quality = good chunking + one embedding model +
small k.

### To revisit / try next
- [x] Build Milestone 4: set up Chroma + an embedding model, ingest a document.
      (Done 23-Jun: data/facts.md -> 4 chunks; retrieval verified in isolation.)
- [x] Build Milestone 5: wire a retrieve node into the graph (the full RAG pipeline).
      (Done 23-Jun: process_input -> retrieve -> generate_response. Answers from docs.)

---

## Lesson 007: conditional edges & agentic routing (23-Jun-2026)

### Static vs conditional edges
- Static edge (`add_edge`): fixed path, always the same next node. A train on one track.
- Conditional edge (`add_conditional_edges`): a ROUTER function reads the state and PICKS
  the next node. A railway switch.

API:
```
builder.add_conditional_edges(source_node, router_fn, path_map)
```
- `router_fn(state) -> str`  returns a KEY (a short string)
- `path_map  {key: next_node_name}`  translates the key to a node

Our graph now branches:
```
process_input --(router)--> "retrieve" -> retrieve -> generate_response -> END
                       \--> "skip" ---------------> generate_response -> END
```
Both paths converge at generate_response; a skipped turn just has empty context (which
generate_response already handles).

### Agentic RAG = the LLM is the router
`route_after_input` is a small LLM call: "does this need the Zephyr KB? reply retrieve or
skip". The AGENT decides its own path -> "agentic". KB question -> retrieve; chit-chat /
general knowledge -> skip and answer directly.

### The big unlock: loops
A conditional edge that points BACK to an earlier node is a CYCLE. Cycles are how agents
iterate (think -> act -> observe -> route back to think). The ReAct loop is literally a
conditional edge pointing backward. We built the simplest case (a one-time branch) but it
is the same tool.

### KEY LESSON: the router is only as good as its context
The router is an LLM call, so its decision quality depends on the CONTEXT we feed it. We
passed ONLY state["query"]. Results:
- "what is 2+2?"                                -> skip     -> "4"          (correct)
- "capital of France?"                          -> skip     -> "Paris"      (correct)
- "lead engineer of Project Zephyr?"            -> retrieve -> Mei Tanaka   (correct)
- "what is the team mascot?"                    -> skip     -> FAILED       (WRONG)
- "what is the team mascot for Project Zephyr?" -> retrieve -> Pascal       (correct)
The failing one is ambiguous WITHOUT context ("which team?"). The router had no
conversation history, so it could not know "team" = Zephyr. Bad/insufficient context in ->
bad decision out - the project thesis, biting the router itself.

### The tradeoff (no free lunch)
| Approach        | Pro                                       | Con                               |
|-----------------|-------------------------------------------|-----------------------------------|
| Always retrieve | never misses a docs question              | wastes a call; can over-ground    |
| Agentic routing | skips pointless retrieval; answers general| can misroute ambiguous follow-ups |
Choose per app. A pure docs-bot may prefer always-retrieve; a general assistant prefers
routing. Better routers (conversation-aware, rerankers) = bonus #16 (agentic RAG).

### One-sentence summary
A conditional edge adds a router function that picks the next node from the state -
enabling branching and (pointed backward) loops; making the router an LLM call gives
"agentic RAG", but the router's decisions are only as good as the context you feed it.

### To revisit / try next
- [ ] Give the router recent conversation context so ambiguous follow-ups route correctly.
- [ ] Use a conditional edge that points BACKWARD to build a real loop (concept 1.6).

---

## Lesson 008: COMPRESS - summarize AND remove (23-Jun-2026)
(Conceptual, before building Milestone 6. The pillar the project is named after.)

### The problem (Lesson 002, coming due)
Every turn sends the WHOLE message history to the LLM. It grows unbounded:
turn 1 [System,H1,A1] -> turn 50 [System,H1,A1,...,H50,A50]. Eventually it blows the
fixed context window: errors, rising cost per turn, "lost in the middle". Unbounded
history is a time bomb.

### Compress = TWO steps: summarize AND remove
The crux (and the easy thing to miss): summarising ALONE makes the history BIGGER.
The messages LIST is what gets sent. Summarising creates a new short text, but the
originals are still in the list:
```
BEFORE:                  [old 50%]                 + [recent 20%] = 70% used
SUMMARISE (no remove):   [old 50%] + [summary 10%] + [recent 20%] = 80%  <- BIGGER!
SUMMARISE then REMOVE:               [summary 10%] + [recent 20%] = 30%  <- target
```
So compression is a PAIR:
1. Summarise  -> produce the 10% gist (stored in scratchpad).
2. Remove     -> delete the 50% of verbatim originals (this reclaims the budget).
The REMOVE is what actually frees space; summarise alone just adds text.

Analogy: a 50-page transcript + a 1-page summary kept together = 51 pages (bigger).
To save shelf space you SHRED the 50 pages and keep the 1-page summary.
Summarise = write the CliffsNotes; Remove = shred the original book. Need both.

### The policy in percentages (Andre's mental model - it is textbook)
A clean summary-buffer policy, in numbers:
- TRIGGER at ~70% of the window (with headroom - do not wait for 100%).
- SUMMARISE the oldest ~50% down to a ~10% note (-> scratchpad).
- KEEP the latest ~20% verbatim (fresh, high fidelity).
- REMOVE the old ~50% of originals.
- RESULT: ~10% summary + ~20% recent = ~30% used, ~70% FREE again - room to grow until
  the next trigger.

The same numbers, as the three states:
```
BEFORE (trigger):        old 50%               + recent 20%  = 70% used
SUMMARISE, no remove:    old 50% + summary 10% + recent 20%  = 80% used  (WORSE)
SUMMARISE + REMOVE:                summary 10% + recent 20%  = 30% used  (target, 70% free)
```
One line: at 70%, summarise the old 50% into a 10% note, then RemoveMessage the old 50%
from the list -> 30% used, 70% free. The REMOVE is what reclaims the budget; summarising
without removing pushes you to 80% and makes it worse.

### WHEN to trigger (size threshold, with headroom + hysteresis)
- Measure "too big" by message COUNT (simple, crude) or TOKEN count (precise - this is
  the "measure tokens" item deferred from Lesson 002, now coming due).
- Trigger with HEADROOM (e.g. at 70% of budget, not 100%) so you compress before the
  window overflows.
- HYSTERESIS: when you compress, compress GENEROUSLY (many old turns at once) so you do
  not trip the threshold again next turn and compress every single turn.

### WHAT to keep vs summarise (the "summary buffer" pattern)
Not all history is equal:
```
[System prompt]        -> ALWAYS keep verbatim (standing instructions)
[Summary of old turns] -> distant past, COMPRESSED into scratchpad
[Last few turns]       -> recent past, kept VERBATIM (high fidelity)
[current question]
```
- System prompt: never summarise (losing it changes behaviour).
- Recent turns: keep verbatim - the model needs PRECISE recent wording.
- Old turns: summarise - only the gist matters; drop "hello/thanks" filler.
What goes in the summary = durable facts + decisions, not pleasantries (Lesson 004's
"don't hoard", made active - the summary is curation). Caveats: summarisation is LOSSY
(an LLM call that can drop detail); re-summarise-the-window is simpler than incremental
(which compounds errors like telephone).

### HOW RemoveMessage works (Lesson 001 reducer, again)
`messages` is governed by the `add_messages` reducer, which APPENDS. To DELETE, return a
special `RemoveMessage(id=...)`; the reducer dispatches on TYPE - a normal message ->
append; a RemoveMessage -> delete the message with that id.
```python
from langchain_core.messages import RemoveMessage

def compress(state):
    old = state["messages"][1:-4]                 # keep system + last 4
    summary = llm.invoke([... "summarise these" ...]).content
    removals = [RemoveMessage(id=m.id) for m in old]
    return {"messages": removals, "scratchpad": summary}
```
The SAME reducer both appends and deletes, based on what you hand it. The summary lives
in `scratchpad`; `generate_response` must INJECT it into the prompt (like the RAG
context), or you deleted the history and kept the summary somewhere the model never sees.

### How it assembles
```
generate_response -> (router: history long?) --no--> END
                                          \--yes--> compress -> END
```
compress: summarise old -> scratchpad ; RemoveMessage the old turns. Next turn,
generate_response injects scratchpad summary + the kept recent turns.

### Everything converges here
- Lesson 001 (reducers) -> RemoveMessage is the reducer deleting.
- Lesson 002 (budget)   -> the WHY + the trigger threshold.
- Lesson 004 (don't hoard) -> the summary is active curation.
- Lesson 005 (checkpoints) -> compression is forward-looking; full history still lives in
  past checkpoints, so time-travel can recover it. You do not truly lose it.
- Lesson 007 (conditional edges) -> the "history long?" trigger is a router.
- scratchpad -> finally used, as the summary store (promised back in Lesson 001).

### One-sentence summary
COMPRESS keeps history within budget by SUMMARISING the distant past into scratchpad AND
REMOVING the verbatim originals with RemoveMessage (summarise alone makes it bigger; the
remove reclaims the space), keeping system + recent turns verbatim, triggered by a
conditional edge only when history is long.

### Addendum (24-Jun): FRAMING the injected summary (a real gotcha we hit)
The compress engineering worked first try, but recall still FAILED: asked "what is my
name?", the agent said "I don't have that information". Reading the saved scratchpad proved
the summary was correct ("Andre's name and favourite number (42)") AND injected. So it was
not an engineering bug - the MODEL disowned the summary, reading "Andre's name" as
third-party trivia: "someone named Andre... I'm starting fresh".

Fix = FRAMING, not code. Wrapping the summary as:
  "This is an ongoing conversation with the user. Treat the summary below as established
   facts about the user you are talking to (e.g. their name, preferences): <summary>"
flipped llama3.2 from disclaiming to "Your name is Andre, and your favourite number is 42".
Same data, different frame, opposite behaviour.

Lesson: context engineering is not only WHAT is in context, but HOW it is framed so the
model knows how to USE it. Injected context needs a clear role ("this is the user", "use
this to answer"), or the model may treat it as detached notes.

### To revisit / try next
- [x] Build Milestone 6: a compress node (summarise + RemoveMessage) + a conditional
      trigger; inject the scratchpad summary in generate_response; watch the count shrink.
      (Done 24-Jun: count shrank 13 -> 9; recall "Andre, 42" from the summary.)
- [ ] Upgrade the trigger from message-count to real TOKEN counting.

---

## Lesson 009: a debugging methodology for LLM apps (24-Jun-2026)
(Distilled from the Milestone 6 recall bug + the Lesson 004 deepseek bug. Process, not concept.)

### The trap
When an LLM app misbehaves, the instinct is to blame the most visible component
("compression is broken!") and start changing it. That is usually the WRONG layer.
The output is a surface; the bug lives in one specific layer beneath it.

### The four layers a failure can live in
1. DATA / STATE   - is the saved/loaded data actually correct? (checkpoint, scratchpad)
2. WIRING / CODE  - do the nodes/edges/returns do what you think? (a real code bug)
3. PROMPT / FRAMING - is the right info present but framed so the model misreads it?
4. MODEL          - info present + well framed, but the model still fumbles (small model).
Locate WHICH layer owns the failure BEFORE fixing anything.

### The playbook (what we actually did)
1. REPRODUCE reliably (the 7-turn script).
2. INSPECT THE STATE, not the surface. We read the saved `scratchpad` from the
   checkpoint - it held the correct summary ("Andre / 42"). That ruled OUT layers 1+2
   (data correct, compress code worked). Same move as Lesson 004 (read checkpoints.sqlite).
   => "separate the SYSTEM from the MODEL".
3. NARROW THE LAYER. Summary correct + injected => the bug is downstream (prompt or model),
   not in compression. Binary-search the pipeline instead of guessing.
4. ISOLATE with a MINIMAL REPRO. We left the graph entirely and ran a tiny script that
   invoked the LLM with just the prompt structures (2 system msgs vs merged). Stripping the
   system down removes variables and makes the test fast + deterministic.
5. READ THE MODEL'S OWN WORDS for clues. It said "someone named Andre... I'm starting fresh"
   - that revealed it read the summary as THIRD-PARTY trivia. The model told us the bug.
6. HYPOTHESIS -> TEST ONE THING -> APPLY. Hypothesis: framing. Changed ONLY the framing in
   isolation -> it worked -> applied to the node -> re-verified live.

### Two worked examples
- Lesson 004 (deepseek "forgot"): read checkpoint -> 5 messages saved -> layer = MODEL
  (reasoning model over-thought), not persistence. Fix was understanding, not code.
- Lesson 008 (compress recall): read scratchpad -> summary correct -> isolate prompt ->
  model reads as third party -> layer = PROMPT/FRAMING. Fix was wording, not engineering.

### Bonus: a recurring PERSONAL error pattern
"Replace became add" - several times an instruction to REPLACE a line became an ADD,
leaving the old line in (graph edge; the context branch; the SqliteSaver import). After any
refactor, RE-READ the changed region and check the old version is actually gone.

### One-sentence summary
Do not fix the loudest component; locate the layer (data / code / framing / model) by
INSPECTING state and ISOLATING a minimal repro, let the model's own words point at the
cause, then change ONE thing and re-verify.

### To revisit / try next
- [ ] When a future bug appears, write down which of the 4 layers it was - build the muscle.

---

## Lesson 010: ISOLATE - multi-agent (supervisor + specialists) (24-Jun-2026)
(Conceptual, before building Milestone 7. The 4th pillar: Write / Select / Compress / ISOLATE.)

### The leap: shared context -> separate contexts
Everything in our graph so far (process_input, retrieve, generate_response, compress)
SHARES one AgentState - one context passed around. ISOLATE breaks that: each specialist
gets its OWN separate context (own state, own messages, own system prompt), walled off
from the main one. That separation IS the pillar.

### The pattern: supervisor + specialists
```
  MAIN context (supervisor)              ISOLATED contexts (specialists)
  +-----------------------+
  | goal + conversation   |-- task ----> [ researcher: own prompt, own messages ]
  | (stays clean)         |<- result --- [ (messy work hidden) ]
  +-----------------------+
```
- SUPERVISOR: holds the goal + conversation; decides which specialist; hands a SCOPED task
  (not the whole history); merges results. Memory lives here.
- SPECIALIST: works in its OWN isolated context (focused prompt + the task only). Can do
  messy/expensive work (many retrievals, several steps) privately.
- HANDOFF: the specialist returns ONLY a clean result. Its internal mess never enters the
  main context.

### Why it is a CONTEXT pillar (the payoff)
A researcher might do 3 retrievals + 2 reasoning steps. With one shared context, all that
noise piles into messages and pollutes everything after (Lesson 004). With ISOLATE it
happens in the specialist's PRIVATE context, and the main conversation sees only the clean
answer. Two wins: clean main context (no bloat/pollution) + better quality (a focused
specialist beats a generalist juggling everything).

### The context BOUNDARY (where multi-agent lives or dies)
Two crossings, each a context-engineering decision:

IN - what the supervisor passes (the task spec):
- The specialist is isolated (no main history), so the task MUST be SELF-CONTAINED.
- Naive pass fails: user (after 4 Zephyr turns) says "what about the mascot?"; the isolated
  specialist has no history -> "which mascot?" -> FAIL. This is the Lesson 007 router bug,
  at the agent boundary.
- Fix = QUERY REWRITING: supervisor rewrites "what about the mascot?" -> "What is the mascot
  of Project Zephyr?". A self-contained instruction.
- Central tension: ISOLATION vs SUFFICIENCY. Too little context -> specialist fails; too
  much (whole history) -> defeats isolation, re-pollutes. Aim for MINIMAL SUFFICIENT context.

OUT - what the specialist returns (the result):
- Only a clean result, not its working. Form: plain text (simple) or STRUCTURED
  ({answer, sources, confidence} - lets the supervisor reason about how to use it).
- FRAME it on the way back, or the main agent misreads it (the Lesson 008 framing bug).

Failure modes at the boundary (all old friends):
| Boundary           | Failure                       | Seen in            |
| IN under-specified | specialist can't resolve task | Lesson 007         |
| OUT lossy/unframed | result degrades / misread     | Lesson 008 + drift |
Plus the TELEPHONE problem: every handoff is lossy, so chaining many agents degrades info
(like incremental-summary drift). Fewer hops = less drift.

META-INSIGHT: every agent boundary is a context filter you DESIGN ("what does this agent
need IN, produce OUT?"). Multi-agent is not a new skill - it is context engineering with
more seams. And bugs live at the seams.

### How the supervisor DECIDES (3 levels)
- ROUTE: pick ONE specialist, hand off, done. (route_after_input, but to agents.) 1 decision.
- PLAN: break the goal into a fixed SEQUENCE of subtasks up front (research -> write -> edit).
- ORCHESTRATE: iterate - call a specialist, READ the result, decide the next step, repeat.
  This is the ReAct loop at the multi-agent level = a conditional edge pointing BACKWARD
  (Lesson 007's cycle). Most power, most cost. Escalate only as needed.

### Sequential vs parallel + merge (Lesson 001 callback)
- SEQUENTIAL (A->B->C): output feeds the next; needed when there is a dependency. Latency adds.
- PARALLEL: independent specialists run concurrently; gather. Latency = the slowest, not the sum.
- MERGE (fan-in): concatenate / synthesize (LLM combine) / vote-pick-best.
- THE PAYOFF: Lesson 001 said reducers matter "beyond convenience" because two nodes can
  write the same field in parallel and the reducer folds them deterministically. THIS is
  that - parallel specialists writing results -> the reducer merges them, no race. The merge
  is itself a Compress-like act (combine N contexts, decide what to keep).

### Two realities
- Specialists are usually STATELESS: fresh isolated context per call, no memory between
  calls. They are pure functions (task in, result out). Memory lives with the SUPERVISOR
  (main thread + checkpointer). Start stateless.
- The loop needs a BRAKE: an orchestrating supervisor can loop forever; cap it (max N
  specialist calls, then answer with what you have). Same instinct as compression hysteresis.

### The big theme: multi-agent REUSES the primitives
- supervisor loop      = Lesson 007 cycles
- parallel merge       = Lesson 001 reducers
- boundary bugs        = Lessons 007/008 context bugs
- stateless vs memory  = Lessons 003/004
You are recombining what you already built, not learning a new system.

### The tradeoff (no free lunch)
More LLM calls, more latency, coordination + failure handling. ISOLATE is for tasks that
genuinely benefit from a focused, isolated worker - not for everything. A simple chat does
not need a committee.

### One-sentence summary
ISOLATE = a supervisor hands isolated, usually-stateless specialists a SELF-CONTAINED task
(query rewriting, or it fails like Lesson 007) and gets back a CLEAN, FRAMED result (or it
is misread like Lesson 008); decisions scale route->plan->orchestrate, parallel results
fan in through a reducer (Lesson 001), and the win is a lean main context because the messy
work stays private.

### To revisit / try next
- [ ] Scope + build Milestone 7: a supervisor + 1-2 stateless specialists with isolated
      context; verify the specialist's working never enters the main messages.

---

## Lesson 011: TDD agents - independent verification (24-Jun-2026)
(From building the capstone code-builder v1, Phase A. The QA agent design.)

### The decision: QA writes tests from the SPEC ALONE
Two ways to wire a QA / test agent:
- spec + code -> tests tend to CONFIRM what the code does (biased to the implementation).
  A buggy code + matching tests = FALSE GREEN (looks passing, is wrong).
- spec ONLY -> tests encode what is CORRECT (from the contract), independent of the code.
  If the code is wrong, the independent tests CATCH it. This is true TDD.
Choose spec-only: the QA becomes a genuine ADVERSARY of the code, both answering to the spec.

### The payoff (seen live, before we even ran the tests)
Code + tests now derive INDEPENDENTLY from the spec, so they can DISAGREE - and the
disagreement is the QA doing its job. Example: the coder returned False for "" (because
`"".isalnum()` is False), but the spec + QA tests say "" is a palindrome (True). The
independent test catches a real coder bug. That mismatch is the entire reason the
architecture exists (in v2 the fix-loop feeds it back to the coder).

### The flip side: spec ambiguity becomes divergence
With independent code + tests, the SPEC is the SOLE shared contract. A vague spec -> coder
and QA resolve the ambiguity differently -> false failures (code "right" by one reading,
tests by another). So spec precision matters even MORE (Lesson 010 IN-boundary, sharpened).

### Tactical notes from the build
- POSITIVE prompts beat NEGATIVE: "Output ONLY these 4 fields" stopped the orchestrator
  leaking an implementation; "do NOT write code" had not (Lesson 002 - negative instructions
  are weakly followed, especially by small models).
- LLM output needs CLEANING before execution: models wrap code in ```fences```; strip them
  before writing to a .py file or it is a syntax error. A cleanup layer always sits between
  the LLM and execution (same theme as JSON parsing in RAG, framing in compress).
- RELATIVE imports need a module run: a file with `from .config` cannot be run as
  `python agents.py` (no package context -> "attempted relative import with no known parent
  package"); run `python -m src.agents` from the parent dir.

### One-sentence summary
Feed the QA/test agent the SPEC ALONE (not the code) so its tests verify CORRECTNESS
independently and catch coder bugs - true TDD; the cost is the spec must be precise, because
it is now the only shared contract between the two independent agents.

### Addendum (24-Jun): green != correct - test COVERAGE gaps
v1 ran end-to-end and reported `passed: True` - but the green was MISLEADING. The spec said
"non-alphanumeric (except spaces) strings are NOT palindromes" (so "a!a" -> False), but the
coder STRIPPED non-alnum ("a!a" -> "aa" -> True). They DISAGREE on "a!a". Yet the QA's
non-alnum test used "a!b" - which is False under BOTH readings ("ab" is not a palindrome
anyway) - a NON-DISCRIMINATING input. So it passed for the wrong reason and never exercised
the divergence. Green build, latent bug.

Lesson: TDD-from-spec gives an INDEPENDENT check, but a check with GAPS still gives false
confidence. "Passed" means "passes THESE tests", not "is correct". Good QA must test
DISCRIMINATING / adversarial cases - the inputs where interpretations diverge - or green is
just green. (Refinement for v2+: prompt the QA to include adversarial cases. Also note: the
spec's "reject non-alnum" was an UNUSUAL convention - the common one is to strip - so the
spec itself wobbled. Garbage spec -> the whole chain wobbles, Lessons 010/011.)

Also: LLM pipelines are NON-DETERMINISTIC - each run produced a different spec/code/tests.
The predicted empty-string failure appeared in one run, not another. Expect variance.

### To revisit / try next
- [x] Step 5: run the tests in a sandbox (temp dir + subprocess + timeout). Works; returns
      {passed, failures}. (Specific failures vary run-to-run; coverage gaps can mask bugs.)
- [x] v2: the fix-loop feeds a failure back to the coder to iterate (+ max-iteration brake).
      (Wired 25-Jun, Change 011; testing next.)
- [ ] QA prompt upgrade: require discriminating/adversarial test cases.

---

## Lesson 012: task pipeline vs conversational agent (25-Jun-2026)
(Surfaced building the capstone v2 - "where's the while-True loop + checkpointer?")

### Two SHAPES of agent system, with different memory/control needs
CONVERSATIONAL AGENT (the foundations langgraph-app):
- A back-and-forth that CONTINUES over many turns.
- Needs a `while True` loop (keep talking until quit) AND a CHECKPOINTER + thread_id to
  remember ACROSS turns and restarts (Lessons 003/004).
- Memory lives BETWEEN invokes - each turn is one invoke; the checkpointer carries state over.

TASK PIPELINE (the code-builder):
- A ONE-SHOT job: input -> run to completion -> output -> exit.
- Runs in a SINGLE invoke; needs no memory BETWEEN runs.
- The working memory it DOES need (e.g. the retry feedback in the fix-loop) lives WITHIN that
  one invoke, in the loop's state. So no `while True`, no checkpointer - by design.

### Why it matters
Do not bolt a chat loop / checkpointer onto a one-shot task out of habit. Ask: does this
system CONTINUE across turns (conversation) or RUN to completion (task)? Match the machinery
to the shape. The foundations agent and the code-builder use the SAME primitives (graph,
state, nodes, edges) but different control/memory shapes.

### Optional (not required) for a task pipeline
- `while True` -> build MANY tasks in one session (UX).
- checkpointer -> resumable builds + time-travel the internal loop (inspect each attempt,
  Lesson 005).
Nice-to-haves, not correctness needs.

### One-sentence summary
A conversational agent needs a loop + checkpointer to persist memory ACROSS turns; a one-shot
task pipeline runs in a single invoke and keeps its working memory (e.g. retry feedback)
WITHIN that invoke - so the loop/checkpointer are optional, not required.

### To revisit / try next
- [ ] (optional) wrap the builder in a while-True for multi-build sessions.
- [ ] (optional) add a checkpointer to make builds resumable + inspect the retry history.

---

## Lesson 013: a red verdict does not say WHO is wrong (25-Jun-2026)
(From testing the capstone v2 fix-loop on "int -> roman numeral": 3 runs, 5 bugs, and
EVERY bug was upstream of the coder. The coder's code was correct on all 3 runs.)

### What happened (the saga, compressed)
- Run 1 (red, hit the brake at 3): (a) the SPEC hallucinated `0 -> ''`, contradicting the
  "1..3999" range; (b) the QA used `pytest.raises` but never wrote `import pytest` - the
  test file would not even load.
- Run 2 (red, hit the brake at 3): a+b fixed, but the QA found NEW ways to break the file:
  (c) no `from solution import int_to_roman` (NameError); (d) it QUOTED the call,
  `assert 'int_to_roman(1)' == 'I'` (compares the literal text).
- Run 3 (GREEN on attempt 1): genuine pass once the spec was consistent and the test file
  ran. 3 prompt fixes total, ALL upstream (orchestrator + QA), NONE to the coder.

### Lesson A: "red != broken code" (the mirror of Lesson 011)
Lesson 011 said green != correct (a passing suite can have coverage gaps). The mirror is
just as true: a FAILING suite does not mean the code is broken. A red verdict is only as
trustworthy as the SPEC and the HARNESS that produced it. Here the code was right every
time; the red came from a contradictory spec and a non-loading test file. Trust the verdict
only as much as you trust the contract + the test runner behind it.

### Lesson B: the QA is the structural weak link
Across 3 runs the bottleneck was ALWAYS the QA, never the coder. That is not luck - writing
a RUNNABLE test file is structurally harder than writing the function: it needs the right
imports, no typos, and valid assertion syntax, or the WHOLE suite fails to load (one bad
line zeroes out the entire harness). The function only has to be correct; the test file has
to be correct AND loadable. More ways to fail = the weak link.

### Lesson C: the loop only routes to the CODER (fire-and-forget QA)
The v2 fix-loop's conditional edge is `{"retry": write_code, "done": END}`. It loops back to
the CODER only. `write_tests` (the QA) runs ONCE at the start and is never re-invoked - it
never sees a single failure log. So when the logs said `NameError: name 'pytest' is not
defined` (a QA bug), they were delivered to the coder's desk - and the coder can only
rewrite `solution.py`, never `test_solution.py`. The runs did not just get UNFIXABLE
garbage, they got MIS-ROUTED garbage: a QA bug sent to the one agent powerless to fix it.
The QA did not "fail to understand" the feedback - it was never asked.

  write_spec -> write_tests -> write_code -> run_sandbox
                    ^                             |
                    |  NO edge back here          | retry
                    X                             v
                                  write_code <----+   (only the coder is in the loop)

### Lesson D: validating a LOOP - plumbing vs value
We almost declared victory too early. The 3 runs proved the loop's PLUMBING (retry fires,
`attempts` increments, the brake stops at MAX) - but NOT its VALUE (a coder taking a real
failure and converging red -> green), because runs 1-2 fed it unfixable bugs and run 3
one-shot (loop never engaged). A live run could not be relied on to produce a natural,
fixable coder bug (non-deterministic - might one-shot, might never converge).
Fix = a DETERMINISTIC isolated test: plant a known buggy `solution.py` against a known-good
test, drive the retry branch directly (`check_fixloop.py`), confirm it converges. Result:
planted False -> coder read the logs -> fixed (and improved n<2 + sqrt) -> True. THEN the
claim was earned. Lesson: to validate a loop, separate "the wiring runs" from "the wiring
produces value", and manufacture the input the live system will not reliably give you.

### Lesson E: the deeper limit - an assertion failure is UN-ATTRIBUTABLE
Andre's instinct: classify the failure and route it back to the agent who authored the
broken script. The clean part works - pytest's EXIT CODE distinguishes "tests ran and
asserted-False" (code suspect) from "tests never loaded" (test file broken), so a
collection error can route to the QA. But the hard part is a genuine limit:
- An ASSERTION failure does NOT say whose fault it is. Exit code 1 = "code and tests
  DISAGREE", not "code is wrong". The `0 -> ''` bug was an assertion failure whose real
  cause was the SPEC. A returncode-classifier would have shipped it to the coder forever.
- There are THREE authors, not two: orchestrator (spec), QA (tests), coder (code) - but
  only TWO scripts. The orchestrator writes no file, yet its spec is upstream of both, and
  spec bugs surface as exactly these ambiguous assertion failures.
- "Which script failed to LOAD" is decidable from logs. "Which author is RIGHT when code
  and tests disagree" is NOT - that needs a judge, not a classifier.

### Lesson F: tactical - the weak-verb prompt bug
The QA's missing `from solution import` traced to a weak verb: the prompt said "Assume
`from solution import <function>`", which the model read as "you MAY assume it is available"
rather than "WRITE this line". Changing it to "Begin the file with these import lines: ..."
plus a POSITIVE assertion example fixed it in one shot. Imperatives + positive examples beat
assumptions + negatives (Lesson 002 / Lesson 011, confirmed a 3rd time). When an agent omits
something, check whether the instruction was a suggestion dressed as a command.

### Addendum: decidable vs undecidable failures - the real routing rule (25-Jun)
Lesson E said an assertion failure is un-attributable. The sharper framing (Andre's): split
failures by DECIDABILITY, and let THAT - not the v2/v3 label - decide who handles each.
- DECIDABLE (test file won't LOAD, pytest exit 2/5): the exit code NAMES the culprit = the
  QA. No judge needed - the SYSTEM can auto-route this back to the QA even in v2.
- UNDECIDABLE (assertion fails, exit 1): code and tests DISAGREE; the culprit could be the
  code, the test, OR the spec. THIS is the only case that needs a judge.
Crucial: decidable routing does NOT ping-pong, because each failure has exactly ONE correct
home (exit 2/5 -> QA; exit 1 -> coder). PING-PONG (an endless code <-> test tug-of-war, each
edit making the other side look broken) only arises in the UNDECIDABLE case, where you GUESS
who to blame and the other side re-breaks. Letting the QA rewrite tests mid-loop also gives
up v2's fixed-target TDD property (tests written once = a stationary target the coder can
converge on). So the rule:
  decidable   -> auto-route (cheap, safe, no judge) - fine in v2.
  undecidable -> escalate to a HUMAN (v2) or build a JUDGE (v3).
The judge + ping-pong problems belong ONLY to the undecidable branch - it was a mistake to
attach them to ALL QA-routing.

### One-sentence summary
A red verdict tells you code and tests disagree, not WHO is wrong - and our v2 loop assumes
it is always the coder (the QA is fire-and-forget), so QA bugs get mis-routed to an agent
that cannot fix them; "which file failed to load" is decidable (route it to its author) but
"who is right when they disagree" is not (it needs a judge, because the spec is a hidden
third author).

### To revisit / try next (this is the v3 "supervisor" design)
- [ ] PROVENANCE / build-ledger (Andre's idea): a data structure recording, per attempt,
      WHICH agent wrote WHICH script at WHICH step, and WHICH output log resulted - so a
      failure can be traced to its author. Note: the CHECKPOINTER (Lesson 005) already gives
      a timestamped per-step history for free - get_state_history is half of this ledger.
      Caveat: provenance answers WHO AUTHORED, not WHO IS AT FAULT (Lesson E) - necessary
      but not sufficient for routing.
- [ ] CLASSIFY in run_sandbox: read pytest's exit code -> {load-error -> QA, assertion ->
      coder/judge}. Make write_tests retry-aware (like write_code) + its own brake.
      (The DECIDABLE part - load-error -> QA - is v2-doable; only the judge is v3.)
- [ ] A JUDGE for assertion disagreements (code vs test vs spec) - the un-decidable case (v3).
- [ ] Guard the timeout path: on TimeoutExpired we discard pytest's partial output
      (e.stdout/e.stderr) - keep it for debugging.

---

## Lesson 014: human-in-the-loop intervention - the RIGHT boundary (25-Jun-2026)
(From building capstone v2 step 2: the build ledger + user-intervention on stuck.)

### The setup
When the fix-loop gives up (brake hit), instead of dying silently the builder hands the
HUMAN a truthful report and lets them steer. The human is the judge (the v2 principle: the
system ASSISTS, it does not GUARANTEE; be truthful about what was tried). The graph stays a
one-shot task; the "ask and retry" lives in an outer `while True` in main.py (Lesson 012).

### The ledger is what makes judging possible
State fields get OVERWRITTEN each retry (no reducer -> replace), so the final state only holds
the LAST attempt. The ledger (an append-only `Annotated[list, add]` of BuildEvents) keeps the
whole TRAJECTORY: spec, tests, and every (coder attempt -> sandbox verdict) pair. The
trajectory is itself diagnostic - e.g. the SAME error across all 3 attempts means the coder
changed the code and the error never budged => the bug is NOT in the code (it is in the
test/spec). You can only see that if you kept the history.

### THE BIG ONE: feedback must enter at the RIGHT boundary (the coarse-lever problem)
Our intervention augments the REQUEST, which enters at request -> orchestrator. But in testing,
the bugs were all in the QA (which reads only the SPEC, never the feedback). So a correction
about a test bug is a BANK-SHOT: human -> request -> orchestrator -> spec -> QA. The
non-deterministic QA then rolls a BRAND-NEW bug each round, so convergence is WHACK-A-MOLE
(it took 3 rounds; one round even REGRESSED to an earlier bug). Each correction did kill its
targeted bug next round (so feedback genuinely steers - unlike the first cut, which silently
discarded it), but slowly, because it acts on the wrong agent.
LESSON: route a correction to the agent that can ACT on it, not to an upstream proxy. The
real cure is a SURGICAL lever (edit the tests directly / let the QA see the feedback), not
"better feedback" through the orchestrator. (This is the IN-boundary design of Lesson 010,
applied to HUMAN feedback: minimal-sufficient context delivered to the right place.)

### Two tactical lessons
- input() READS ONE LINE. Multi-line human input truncates, and the remainder leaks into the
  next read (terminal `^R` mangling). If you want rich human input, read until a terminator
  (e.g. a blank line), do not assume one input() call captures a paste.
- DO NOT inline meta-instructions into an output template. The orchestrator echoed its own
  rule ("edge cases MUST be consistent...") verbatim into the spec, because the rule sat
  INSIDE the "3. Edge cases" field. Separate FORMAT (what to output) from GUIDANCE (how to do
  it), and prefer a POSITIVE "Output ONLY sections 1-4" over a negative "do not copy this"
  (Lesson 011: small models follow positives more reliably). Put the format constraint LAST
  for recency (Lesson 002: models attend most to start/end).

### Test technique: manufacture the failure
To test a STUCK / error path you must induce failures a healthy pipeline will not produce -
weaken an agent prompt, or set MAX_ATTEMPTS=1. A good pipeline one-shots and never walks the
path you are trying to test. (Same principle as Lesson 013-D's planted-bug test: manufacture
the input the live system will not reliably give you.)

### One-sentence summary
Hand the human a truthful TRAJECTORY (the ledger) so they can judge a stuck build - but route
their correction to the agent that can ACT on it: feeding QA-bug feedback through the
orchestrator is a coarse bank-shot that converges by whack-a-mole, which is the live case for
a surgical (edit-tests / QA-visible) feedback lever.

### Addendum: the is_palindrome case study - nobody's feedback reaches the QA (26-Jun)
Ran is_palindrome (the canonical AMBIGUOUS spec - Lesson 011's own example) through the v2
intervention loop. 3 builds, 2 human corrections, never converged - and it laid the
wrong-boundary problem bare. In ALL 3 builds the CODER's code was essentially correct; the
QA's tests were the bug:
- build 1: QA wrote CONTRADICTORY tests - "...Panama" == True (non-alnum IGNORED) AND
  is_palindrome("a!b!") raises ValueError (non-alnum REJECTED). No code satisfies both.
- build 2: contradiction again - "...Panama" == True (case-INSENSITIVE) AND "Madam" == False
  (case-SENSITIVE). Mutually exclusive.
- build 3: MALFORMED test - `assert pytest.raises(ValueError, is_palindrome)` (calls with no
  arg -> TypeError; misuses pytest.raises).
The coder produced IDENTICAL code across attempts each time = the Lesson 013 "same error 3x =
not the code's fault" signal.

THE SMOKING GUN - who gets which feedback:
| who is WRONG | who the pytest log reaches | who the human feedback reaches |
|--------------|---------------------------|--------------------------------|
| the QA       | the CODER (retry branch)  | the ORCHESTRATOR (request->spec)|
The QA reads the spec ONLY and runs ONCE (Lesson 013-C), so NOBODY's correction ever reaches
the agent that is actually wrong. The human even typed "the test is wrong" - it went to the
orchestrator, not the QA. That is why it is whack-a-mole: the loop is STRUCTURALLY unable to
fix a QA bug. Not a regression - the documented coarse-lever limit, biting hard.

Takeaways:
- ambiguous specs are a TRAP for multi-agent (orchestrator + QA resolve conventions
  DIFFERENTLY -> contradictory tests). Use UNAMBIGUOUS tasks (int->roman, is_prime) to test
  the machinery; save is_palindrome as a stress test.
- "quality regressed" was a misread: earlier one-shots were luck + lenient tests (Lesson 011
  variance). The system behaved exactly as documented.
- the cure is step 2.5 + surgical feedback: re-invoke the QA on ITS OWN failures, and let
  human feedback reach the QA, not just the orchestrator.

### To revisit / try next
- [ ] SURGICAL feedback: edit the failing tests/spec directly, or pass feedback into the QA's
      own context - so a correction reaches the agent that owns the bug.
- [ ] multi-line feedback reader (read until a blank line) - the deferred input() fix.
- [ ] accumulated feedback could bloat the request over many rounds - cap or summarise it.

---

## Lesson 015: regeneration vs editing - convergence needs MEMORY + TARGETING (26-Jun-2026)
(From testing v2 surgical-feedback on is_palindrome: feedback reached the QA, but it STILL
whack-a-moled. Andre diagnosed the real architectural flaw.)

### The flaw: every intervention round is a fresh random draw
Each round, main.py calls `app.invoke({request, feedback})`, which runs
write_spec -> write_tests -> write_code FROM SCRATCH. The invoke is STATELESS (Lesson 012),
so every round DISCARDS the previous spec/tests/code and regenerates all three. The only
thing carried forward is the feedback TEXT. So the system has NO MEMORY of its own artifacts.

Seen in the log (5 rounds on is_palindrome):
- round N feedback fixes the last bug, but the non-deterministic QA regenerates a BRAND-NEW
  contradiction (e.g. "!@#$" -> False while "" -> True, though both clean to ""; or
  asserting "a, b!@#" is a palindrome when "ab" is not).
- the target is RE-RANDOMISED every round, so it cannot converge - good work is thrown away
  and new bugs appear. That IS the whack-a-mole.

### The asymmetry that gives it away
The CODER is the ONLY agent that EDITS: on a retry it receives its PREVIOUS code + the
failures, so it refines toward a fixed target (this is why the coder loop converges - Lesson
013-D). The orchestrator and QA REGENERATE BLIND each round (they get only request/spec, never
their own previous output). So two of three agents have no continuity. Convergence requires
all the artifacts to hold still except the one being fixed.

### The two missing capabilities (Andre's words: "keep track of past iterations and know
### which agent to edit which script")
1. MEMORY - persist {spec, tests, code} across intervention rounds; keep the parts that were
   RIGHT instead of regenerating them.
2. TARGETING - identify WHICH artifact is wrong and edit ONLY that one (surgically), pinning
   the rest. If spec+code are fine and one test assertion is wrong, re-run ONLY the QA to edit
   THAT test.

We currently have NEITHER. The build ledger RECORDS the trajectory but nothing USES it to
target. The checkpointer (Lesson 005/012, dismissed as "optional") is exactly the mechanism
that would persist artifacts across rounds. Surgical feedback (Lesson 014) is the targeting.
This finding unifies all those deferred threads: convergence = memory + targeting.

### Caveat: is_palindrome is also a pathological spec
Independent of the architecture, is_palindrome is genuinely self-contradictory: after
stripping non-alnum, "", "   ", and "!@#$" ALL collapse to "" - so "empty raises / whitespace
False / non-alnum False / empty-is-palindrome True" cannot all hold unless you check BEFORE
cleaning. Even a human team regenerating from scratch each round would spin. Fighting a weak
architecture AND the worst-case task at once. Use unambiguous tasks to test; the architecture
is the real lesson.

### One-sentence summary
The v2 loop whack-a-moles because each intervention round regenerates the WHOLE pipeline from
scratch (no memory) and sprays feedback at every agent (no targeting), so the target is
re-randomised every round; convergence needs to PERSIST the artifacts and SURGICALLY edit only
the wrong one - which is the v3 redesign.

### Addendum: v3 BUILT + PROVEN - the loop converges (26-Jun)
The three "to revisit" items below are now all DONE (Changes 014-015):
- MEMORY: main.py carries {spec,tests,code,test_result} into the next invoke (Lesson 003 pattern).
- TARGET: a `fix_target` + a dispatcher (Lesson 007) send a fix to ONE agent; the rest are pinned.
- EDIT-AWARE: all three agents (orchestrator/QA/coder) EDIT their own previous artifact +
  feedback instead of regenerating.
PROVEN live: a tests-fix converges (spec byte-identical/pinned, QA edits the suite surgically);
a spec-fix EDITS the spec (near-identical, only the targeted rule changed) and cascades to green.
The whack-a-mole is GONE - convergence where v2 re-randomised. CAVEAT: edit QUALITY is still
model-gated (llama3.2 only partially applied a spec edit) - the WIRING is what was proven here;
quality wants a stronger / per-agent model (Lesson 016).

### To revisit / try next (= the v3 design)
- [ ] PERSIST artifacts across rounds (carry {spec,tests,code} forward, or a checkpointer).
- [ ] Make the QA + orchestrator RETRY-AWARE (previous output + feedback -> EDIT, like the coder).
- [ ] TARGET: route a fix to ONE agent (re-run only it + downstream), pinning the rest.

---

## Lesson 016: agent quality is capped by MODEL capability (26-Jun-2026)
(From testing v3 step-3, the "code" fix path: the human's feedback was CONFIRMED reaching the
coder, and it STILL could not write correct code.)

### The diagnostic that settled it: print the actual prompt
The coder kept ignoring the human's "raise ValueError on non-alnum" feedback. Two hypotheses:
a wiring bug (feedback not reaching the coder) or a model limit. To decide, we added a temp
debug dump of the EXACT task string the coder receives (Lesson 009: inspect the data, do not
trust the surface). The dump showed the feedback VERBATIM in the prompt:
    "A human reviewer says, specifically about the CODE: <feedback> ... Apply this."
=> wiring is fine. This technique ISOLATES wiring from capability - always print what the
agent actually sees before blaming the agent.

### Evidence it is the MODEL, not wiring or temperature
- WIRING: ruled out - the debug dump proved the feedback was in the prompt.
- TEMPERATURE: ruled out - the coder's output VARIED across attempts (added empty-checks,
  len-checks), so it was not stuck on one deterministic answer. Variation did not help.
- THE SMOKING GUN: when the coder finally tried to use the feedback it wrote the raise
  BACKWARDS - `if s != s[::-1]: raise ValueError(...); return False` - i.e. raise when NOT a
  palindrome, return False when it IS. It is not refusing the instruction; it genuinely does
  not UNDERSTAND it. A 3B model (llama3.2) has a strong "palindrome = strip + compare" prior
  that overrides the spec, the failing test, AND the human feedback.

### The asymmetry: editing a test is EASY, writing code is HARD
The SAME architecture converged a "tests" fix in the same session (flip `== True` to
`== False` - a tiny edit any model can do) but failed every "code" fix (writing a correct
implementation needs real coding ability). So the human-targeting + memory loop (v3) is SOUND
- it is specifically the CODER's code-quality that is capped by the model. The weakest agent's
model is the system's ceiling.

### The lesson
No orchestration cleverness fixes a model that cannot follow the prompt. Match the model to
the task: a cheap small model is fine for easy edits (QA test tweaks, the orchestrator's
spec), but the CODER needs a capable / code-specialised model (e.g. llama3 8B, or
qwen2.5-coder) or the whole pipeline stalls on it. This also argues for PER-AGENT models
(cheap where easy, strong where hard) rather than one shared LLM (today config.get_llm() is
shared by all three agents).

### One-sentence summary
The v3 wiring was correct (proved by printing the coder's actual prompt) - it was the 3B model
that could not implement "raise on non-alnum" (it even wrote the raise backwards); agent
quality is capped by model capability, and since editing a test is easy but writing code is
hard, the CODER needs a stronger/code-specialised model than the rest.

### Addendum: the model swap is NON-UNIFORM (26-Jun)
Swapped OLLAMA_MODEL to llama3:latest (8B) and re-ran. The bigger model did NOT improve the
system uniformly - it shifted WHERE it fails:
- CODER: BETTER - wrote correct standard-palindrome code on attempt 1 and held it stable (vs
  the 3B flailing / writing the raise backwards). Confirms the capability point indirectly.
- ORCHESTRATOR: WORSE - ignored "EXACTLY four sections, output ONLY sections 1-4" and reverted
  to a chatty assistant tutorial ("Here's a simple function... Hope that helps!") AND LEAKED
  the full implementation INTO the spec - regressing the no-code-leak we had tuned. The spec
  stopped being a clean contract; it now contained the answer, quietly breaking TDD-INDEPENDENCE
  (the QA no longer derives tests from a contract alone - Lesson 011).
- QA: still wrong - confidently asserted a REAL palindrome ("Was it a car or a cat I saw?") was
  == False.
The build still CONVERGED, but via two human "tests" fixes - the CODE-feedback path was never
exercised (the 8B coder nailed the code, so no code fix was needed). So Lesson 016's exact
prediction (stronger coder acts on code-feedback) stays INDIRECTLY confirmed, not directly tested.

COROLLARY: "use a bigger model" is NOT a uniform win - it trades one failure mode for another
(better code, chattier/leakier spec). This sharpens the PER-AGENT-MODELS case: put the muscle on
the CODER, but the orchestrator wants a TERSE model (or a re-hardened prompt). One shared LLM
cannot be optimal for agents with OPPOSITE needs (terse-structured spec vs creative-correct code).

### To revisit / try next
- [~] Re-run the "code" fix with a stronger coder (llama3:latest tried; coder IS more capable,
      but the code-feedback path itself is still untested - need a task where 8B code is wrong).
- [ ] PER-AGENT models: let config pick a different model per role (cheap QA/orchestrator,
      strong coder) instead of one shared llm.
- [ ] Remove the temp CODER-TASK debug print now it has served its purpose.

### Addendum 2: qwen2.5-coder CONFIRMS the ceiling - 20% -> 80% (26-Jun)
The clean A/B that settles it: same architecture, same eval, same tasks - only the model
changes. llama3=0%, llama3.2=20%, qwen2.5-coder=80% (4/5 one-shot; 100% on a lucky run). So the
poor autonomous pass-rate was ALWAYS the model, never the design - every v1/v2/v3 decision was
sound. The one remaining failure (gcd) was spec ambiguity (Lesson 011), the human-in-the-loop
case. Set qwen2.5-coder as the default. Lesson confirmed with a number, not a hunch.

---

## Lesson 017: the self-healing supervisor - an auto-judge that routes to the author (26-Jun-2026)
(Built after the eval (M10) showed the autonomous floor was gated by QA reliability. Andre's
idea: "route each error to the responsible agent" - i.e. automate the human's fix_target.)

### The idea: automate the human's fix_target with a JUDGE
v3 already lets a HUMAN pick which artifact to fix (fix_target) when stuck; the edit-aware
agents + dispatcher do the rest. The self-healing supervisor just replaces the human with a
JUDGE for N rounds, then escalates to the human. The judge's `culprit` IS the human's
`fix_target`; everything downstream was already built - the only new piece is the judge + the
loop wiring. (ISOLATE at its fullest: a supervisor that triages and routes to specialists.)

### The judge - two tiers (Lesson 013 decidable/undecidable)
- DECIDABLE (mechanical, no LLM): the test file did not LOAD (collection error) -> the QA's
  file is broken -> route to tests. Free, certain.
- UNDECIDABLE (LLM): an assertion failed -> a focused judge call reads spec+tests+code+failure
  and returns CULPRIT (code|tests|spec) + REASON. culprit -> fix_target, reason -> feedback.
  Lenient parse, default to 'code' (the old safe behaviour).

### The brake: ONE counter, incremented in the JUDGE
The loop now passes through the judge every round (any agent may act), so the brake must count
ROUNDS, not coder-calls. Insight (Andre): reuse the existing `attempts` counter, but MOVE the
increment from write_code to the JUDGE - the node that runs once per round. MAX_ATTEMPTS=3
rounds, then route to the human (the existing stuck report). One counter, one brake - no
second `rounds` field needed.

### Validated three ways
1. check_judge.py (isolation): correctly fingers a code bug as 'code', a test bug as 'tests'.
2. gcd, a real CODE bug (qwen wrote the LCM formula + oscillated on the import): judge routed
   -> code every round (correct), the coder could not fix it, escalated to the human after 3.
   The whole loop worked: judge -> route -> brake -> human.
3. gcd, a TEST bug (the QA dropped `import pytest`): judge routed -> tests, the QA edited the
   import back in, GREEN in ONE round - spec + code pinned. The coder-only loop could NEVER do
   this (it cannot touch the test file). The judge's value, demonstrated.

### The KEY insight: the value is routing AWAY from the coder
On a TEST or SPEC bug the judge routes to the QA/orchestrator - something the old coder-only
loop structurally could not do (Lesson 013-C). THAT is its value-add. On a pure CODE bug it
correctly routes to the coder = the same as the old loop, so it offers no shortcut for a stuck
coder (just one extra LLM call to confirm 'code'). So the supervisor earns its keep
specifically on the QA/spec failures the eval showed dominate autonomous runs.

### Honest limits
- The judge is MODEL-GATED (Lesson 016): on a weak model it would mis-attribute. qwen2.5-coder
  makes it reliable; llama3.2 would not.
- PING-PONG (judge flips code <-> tests) is bounded by the combined brake (3 -> human).
- Edit/coder QUALITY is still model + TEMPERATURE gated - keep the coder at temp 0.2; high temp
  makes it oscillate (the import/LCM flip-flop on gcd). High temp + a loose QA prompt are useful
  to TRIGGER failures for testing, but are test fixtures - restore for production.

### One-sentence summary
The self-healing supervisor replaces the human's fix_target with a JUDGE (decidable load-errors
mechanically, undecidable assertions by LLM) that routes each failure to the agent who can fix
it, looping a combined-braked 3 rounds then escalating to the human - and its real value is
routing AWAY from the coder to recover the QA/spec bugs the old coder-only loop never could.

### To revisit / try next
- [ ] PER-AGENT models/temps (the eval + gcd both point here): a strong/deterministic coder and
      a separate judge model - config.get_llm(role) instead of one shared llm.
- [ ] NON-PROGRESS detection: if the same failure repeats, stop trusting the judge and escalate
      early (cheaper than burning all 3 rounds; guards ping-pong).
- [ ] the judge could append its verdicts to the LEDGER for a transparent stuck-report.

---

## Lesson 018: an API exposes YOUR agent logic, not a generic chat (26-Jun-2026)
(From M11 - and the "why not just open-webui serve?" question.)

### The layers
```
  Browser UI   <- the FACE (borrowed: Swagger /docs, Streamlit, or Open WebUI)
  Agent logic  <- the BRAIN (what YOU built: orchestrator/QA/coder/sandbox/judge)
  Ollama       <- the engine (runs the model)
```
`open-webui serve` ships a GENERIC chat brain (forward prompt -> model -> reply, + built-in
RAG/tools). Your `api.py` ships YOUR brain - a specific, verified, self-healing workflow. Same
engine underneath; different logic layer.

### The differentiator: VERIFIED output
A chat server returns whatever the model SAID (unverified text). Your `/build` returns code that
was EXECUTED against generated tests in a sandbox - `status=="ok"` is a real pass/fail signal,
not a vibe. That verification is the value your agent adds over a raw model call.

### The frontend is BORROWED, never hand-built
The project is the brain, not the face. M11's auto-generated Swagger `/docs` is already a usable
UI; Streamlit/Gradio (~20 lines of Python) or pointing Open WebUI at the API (OpenAI-compat shim
/ Pipelines) are the upgrade paths. Do not code a bespoke frontend - off-thesis.

### One-sentence summary
An API server exposes the agent LOGIC you built (a verified, self-healing multi-agent workflow),
not a generic chat over a raw model - which is why you build your own API instead of running
open-webui serve; the UI is borrowed (Swagger now, Streamlit/Open WebUI later).

---

## Lesson 019: containerize for reproducibility - and the localhost-in-a-container gotcha (26-Jun-2026)
(From M12 - Dockerizing the builder API.)

### What Docker buys: "runs the same everywhere"
A Dockerfile is a RECIPE; `docker build` bakes it into an IMAGE (app + pinned deps + runtime,
frozen); `docker run` starts a CONTAINER (a running instance of the image). The win is
REPRODUCIBILITY - the image runs identically on any machine with Docker, killing "works on my
machine". A container is lighter than a VM (shares the host kernel, isolates just the app).
PIN your deps (==versions) or the image is only as reproducible as whatever "latest" happened
to be that day (Andre's catch).

### The gotcha: `localhost` inside a container is the CONTAINER, not the host
Our app calls Ollama. Inside the container, `localhost:11434` points at the container itself -
nothing is listening there; Ollama runs on the HOST. Fix: reach the host via the special DNS
name `host.docker.internal:11434` (Docker Desktop), passed in as `OLLAMA_BASE_URL`. GENERAL
RULE: a container is ISOLATED by default - anything outside it (host services, other containers)
is reached by an explicit address, never `localhost`.

### Keep the image LEAN
Do NOT `pip freeze` a fat env into the image (our openwebui-env has ~265 packages). Pin the few
DIRECT deps; pip resolves the rest. A `.dockerignore` keeps junk (logs, __pycache__, *.sqlite,
chroma_db) out of the build context.

### Two daemon facts that bit us
- `docker --version` works with NO daemon (it only checks the client). `docker build/run` need
  the DAEMON (the engine) running - on a Mac that means launching Docker Desktop. Confirm with
  `docker info` (a `Server:` section = daemon up).

### One-sentence summary
Docker packages the app + its PINNED environment into a portable IMAGE that runs the same
anywhere; the classic gotcha is that `localhost` inside a container is the container itself, so
host services (like Ollama) are reached via host.docker.internal, not localhost.

---

## Lesson 020: observability - one callback at the top traces the WHOLE graph (26-Jun-2026)
(From wiring Langfuse into the capstone.)

### What tracing buys
An LLM app is a black box - print() is crude + ephemeral. A tracing platform (Langfuse) records
every LLM call (prompt, output, latency, tokens) AND the structure of a run, in a searchable
dashboard. Trace = one run; Observation/Span = a step within it. Quant framing: an execution
blotter + latency/cost analytics for your model calls.

### The key mechanic: callbacks PROPAGATE down the graph
Attach a Langfuse CallbackHandler and pass it ONCE to the top-level invoke
(`config={"callbacks":[handler]}`). LangChain propagates it through the run via context, so EVERY
nested call - each LangGraph node AND each agent's `llm.invoke` - is captured automatically, with
NO per-agent wiring. One code-builder build -> a `LangGraph` root trace with the orchestrator, QA,
coder, and judge generations nested under it (incl. the judge's "CULPRIT: tests" verdict and the
self-heal). The whole multi-agent pipeline becomes a visible, timed, token-counted tree.

### Setup gotchas (from getting it working)
- The handler reads LANGFUSE_* from the ENVIRONMENT (langfuse 4.x), so the app must
  `load_dotenv()` and the .env must hold the keys (gitignored - they are secrets).
- langfuse 4.x reads `LANGFUSE_HOST`, not `LANGFUSE_BASE_URL`; a wrong name silently defaults to
  the EU cloud (fine only if that is your region).
- Traces flush ASYNC on a background thread - exit the program CLEANLY (not kill) so the queue
  flushes; add `get_client().flush()` if a short script exits too fast.

### One-sentence summary
Tracing makes the agent black box visible; with LangChain/LangGraph you attach ONE Langfuse
callback to the top-level invoke and it propagates to every node + every agent LLM call, so a
whole multi-agent build (including the self-healing judge) shows up as a nested, timed,
token-counted tree.
