"""Graph nodes.

Each node takes the current state and returns ONLY the fields it changes.
LangGraph merges the returned partial into the full state (using each field's
reducer, e.g. `add_messages` for `messages`).
"""

from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage

from .config import get_langfuse_handler, get_llm, SYSTEM_PROMPT, CONTEXT_WINDOW, KEEP_RECENT, COMPRESS_AT
from .rag import get_vectorstore

llm = get_llm()
langfuse_handler = get_langfuse_handler()


def process_input(state) -> dict:
    """Turn the raw query into a HumanMessage and append it to the history."""
    # return {"messages": [HumanMessage(content=state["query"])]}
    new_messages = []
    # add system prompt ONCE, at the very front of the conversation
    already_has_system = any(
        isinstance(m, SystemMessage) for m in state.get("messages", [])
    )
    if not already_has_system:
        new_messages.append(SystemMessage(content=SYSTEM_PROMPT))
    # then append user actual question
    new_messages.append(HumanMessage(content=state["query"]))
    
    return {"messages": new_messages}

def route_after_input(state) -> str:
    """Agentic router: decide whether the question needs the knowledge base.

    Returns "retrieve" (look up docs) or "skip" (answer directly).
    """
    decision = llm.invoke([
        SystemMessage(content=(
            "You are a routing classifier. Decide whether answering the user's "
            "question requires looking up the internal Project Zephyr knowledge base "
            "(a private company doc). Reply with ONLY one word: 'retrieve' or 'skip'. "
            "Use 'retrieve' for anything about Project Zephyr, Calderwood Capital, or its "
            "people/budget/details. Use 'skip' for general knowledge, math, or chit-chat."
        )),
        HumanMessage(content=state["query"]),
    ])
    choice = "retrieve" if "retrieve" in decision.content.lower() else "skip"
    print(f"[router] {choice} (for: {state['query'][:50]})")
    return choice

def route_compress(state) -> str:
    # after generating: 'compress' if history is too long else 'end'
    n = len(state["messages"])
    decision = "compress" if n > COMPRESS_AT else "end"
    print(f"[compress-router] {decision} (history = {n} messages, threshold {COMPRESS_AT})")
    return decision

def retrieve(state) -> dict:
    # select: fetch the top-k chunks most relevant to the current query
    docs = get_vectorstore().similarity_search(state["query"], k=3)
    context = "\n\n".join(d.page_content for d in docs)
    return {"context": context}


def generate_response(state) -> dict:
    # call the LLM on the history, injecting the summary + retrieved context
    summary = state.get("scratchpad", "") # running summary (compress writes this)
    context = state.get("context", "")
    messages = state["messages"]

    # transient prompt: prepend summary + context as system instructions for this call
    # not returned into state["messages"], so they are never saved to history
    extras = []
    if summary:
        extras.append(SystemMessage(content=(
            "This is an ongoing conversation with the user. The summary below captures "
            "earlier parts of THIS SAME conversation - treat it as established facts about "
            "the user you are talking to (e.g. their name, preferences):\n\n"
            f"{summary}"
        )))

    if context:
        # give model the retrieved chunks as system instruction for this call only
        # this list is not returned into state["messages"], so it is never saved to history
        extras.append(SystemMessage(content=f"Use this retrieved context to answer:\n\n{context}"))
    
    prompt = extras + messages

    response = llm.invoke(
        prompt,
        config={"callbacks": [langfuse_handler]},  # <-- this is what actually enables tracing
    )

    # token gauge: check how full is the window this current turn
    used = (response.usage_metadata or {}).get("input_tokens", 0)
    pct = 100 * used / CONTEXT_WINDOW if CONTEXT_WINDOW else 0
    print(f"[tokens] {used}/{CONTEXT_WINDOW} ({pct:.0f}% of window)")

    # # Accumulate working memory instead of overwriting it each turn.
    # scratchpad = (state.get("scratchpad", "") + "\n" + response.content).strip()

    # return {"messages": [response], "scratchpad": scratchpad}

    # note: scratchpad is now owned by the compress node
    # we do not write it here
    return {"messages": [response]}

def compress(state) -> dict:
    # summarize the old turns into scratchpad then delete them with RemoveMessage
    messages = state["messages"]

    # keep messages[0] (system) + the last KEEP_SECRET verbatim
    # summarize the middle
    old = messages[1:-KEEP_RECENT]
    if not old:
        return {} # nothing old enough to compress <- no op
    
    # fold old turns (plus any existing summary) into short note of durable facts
    previous = state.get("scratchpad", "")
    transcript = "\n".join(f"{type(m).__name__}: {m.content}" for m in old)
    summary = llm.invoke([
        SystemMessage(content=(
            "Summarise the conversation into a short note of DURABLE facts and decisions "
            "(names, numbers, preferences, goals). Drop pleasantries. If an existing "
            "summary is given, merge the new content into it."
        )),
        HumanMessage(content=f"Existing summary:\n{previous}\n\nNew turns to fold in:\n{transcript}"),
    ]).content

    removals = [RemoveMessage(id=m.id) for m in old]
    print(f"[compress] summarized + removed {len(old)} old messages")
    
    return {"messages": removals, "scratchpad": summary}
