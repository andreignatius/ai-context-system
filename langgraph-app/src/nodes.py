"""Graph nodes.

Each node takes the current state and returns ONLY the fields it changes.
LangGraph merges the returned partial into the full state (using each field's
reducer, e.g. `add_messages` for `messages`).
"""

from langchain_core.messages import HumanMessage, SystemMessage

from .config import get_langfuse_handler, get_llm, SYSTEM_PROMPT
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

def retrieve(state) -> dict:
    # select: fetch the top-k chunks most relevant to the current query
    docs = get_vectorstore().similarity_search(state["query"], k=3)
    context = "\n\n".join(d.page_content for d in docs)
    return {"context": context}


def generate_response(state) -> dict:
    # call the LLM on the history, injecting retrieved context when present
    context = state.get("context", "")
    messages = state["messages"]
    prompt = []

    if context:
        # give model the retrieved chunks as system instruction for this call only
        # this list is not returned into state["messages"], so it is never saved to history
        prompt = [SystemMessage(content=f"Use this retrieved context to answer:\n\n{context}")] + messages
    else:
        prompt = messages

    response = llm.invoke(
        prompt,
        config={"callbacks": [langfuse_handler]},  # <-- this is what actually enables tracing
    )

    # Accumulate working memory instead of overwriting it each turn.
    scratchpad = (state.get("scratchpad", "") + "\n" + response.content).strip()

    return {"messages": [response], "scratchpad": scratchpad}
