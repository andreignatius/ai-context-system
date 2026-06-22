"""Graph nodes.

Each node takes the current state and returns ONLY the fields it changes.
LangGraph merges the returned partial into the full state (using each field's
reducer, e.g. `add_messages` for `messages`).
"""

from langchain_core.messages import HumanMessage

from .config import get_langfuse_handler, get_llm

llm = get_llm()
langfuse_handler = get_langfuse_handler()


def process_input(state) -> dict:
    """Turn the raw query into a HumanMessage and append it to the history."""
    return {"messages": [HumanMessage(content=state["query"])]}


def generate_response(state) -> dict:
    """Call the LLM on the full message history, with Langfuse tracing wired in."""
    response = llm.invoke(
        state["messages"],
        config={"callbacks": [langfuse_handler]},  # <-- this is what actually enables tracing
    )

    # Accumulate working memory instead of overwriting it each turn.
    scratchpad = (state.get("scratchpad", "") + "\n" + response.content).strip()

    return {"messages": [response], "scratchpad": scratchpad}
