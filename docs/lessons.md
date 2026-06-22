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
