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

### To revisit / try next
- [ ] Step 5: run the tests in a sandbox; watch the empty-string test FAIL (QA catching the bug).
- [ ] v2: the fix-loop feeds that failure back to the coder to iterate.
