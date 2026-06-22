"""Demo: watch the `messages` list GROW via the add_messages reducer,
and watch `scratchpad` get REPLACED because it has no reducer.

This is offline - it does NOT call the LLM/Ollama. It just shows what
LangGraph does for you behind the scenes when a node returns a partial update.

Run from the langgraph-app/ directory:
    python demos/reducer_demo.py
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages


def show(step, messages, scratchpad):
    # Print the messages as short "Role(text)" labels so it's easy to read.
    labels = [f"{type(m).__name__.replace('Message', '')}({m.content!r})" for m in messages]
    print(f"{step:<26} messages={labels}")
    print(f"{'':<26} scratchpad={scratchpad!r}\n")


# --- Initial state (same as main.py's invoke) ------------------------------
messages = []
scratchpad = ""
show("START", messages, scratchpad)

# --- Step 1: process_input returns ONLY the new piece ----------------------
# The node would `return {"messages": [HumanMessage("What is 2+2?")]}`.
# LangGraph then merges it for us. We do that merge here, by hand, to SEE it:
new_piece = [HumanMessage("What is 2+2?")]
messages = add_messages(messages, new_piece)   # <-- the hidden old + new step
# scratchpad was not returned, so it stays the same.
show("after process_input", messages, scratchpad)

# --- Step 2: generate_response returns its new pieces ----------------------
# The node would `return {"messages": [AIMessage("4")], "scratchpad": "4"}`.
new_piece = [AIMessage("4")]
messages = add_messages(messages, new_piece)   # messages: append (has reducer)
scratchpad = "4"                               # scratchpad: REPLACE (no reducer)
show("after generate_response", messages, scratchpad)

# --- What main.py prints at the end ----------------------------------------
print("FINAL answer (messages[-1].content):", messages[-1].content)

print(
    "\nNotice: messages went [] -> [Human] -> [Human, AI] (it GREW),\n"
    "while scratchpad just got overwritten to its latest value."
)
