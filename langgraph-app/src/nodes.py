"""Graph nodes.

Each node takes the current state and returns ONLY the fields it changes.
LangGraph merges the returned partial into the full state (using each field's
reducer, e.g. `add_messages` for `messages`).
"""

from langchain_core.messages import HumanMessage, SystemMessage

from .config import get_langfuse_handler, get_llm, SYSTEM_PROMPT

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


def generate_response(state) -> dict:
    """Call the LLM on the full message history, with Langfuse tracing wired in."""
    response = llm.invoke(
        state["messages"],
        config={"callbacks": [langfuse_handler]},  # <-- this is what actually enables tracing
    )

    # Accumulate working memory instead of overwriting it each turn.
    scratchpad = (state.get("scratchpad", "") + "\n" + response.content).strip()

    return {"messages": [response], "scratchpad": scratchpad}
