"""Graph state definition.

The `messages` field uses LangGraph's `add_messages` reducer: nodes return only
the *new* messages they produce, and LangGraph appends them to the running list
for us. This is the idiomatic alternative to mutating the list in place.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history (reducer-managed)
    scratchpad: str                          # accumulated working memory
    query: str                               # current user question
    context: str                             # retrieved RAG chunks for this turn (replaced each turn)
