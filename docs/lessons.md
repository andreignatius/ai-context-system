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
- [ ] Run a tiny demo that prints the messages list growing each step.
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
- [ ] Build Milestone 1: add a small, Case-B-quality system prompt to the agent.
- [ ] Learn to actually MEASURE tokens (so budget sizes stop being abstract).
