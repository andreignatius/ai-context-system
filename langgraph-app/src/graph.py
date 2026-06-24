"""Graph assembly: wires nodes and edges into a compiled, runnable app."""

from langgraph.graph import END, StateGraph

from .nodes import generate_response, retrieve, process_input, route_after_input, compress, route_compress
from .state import AgentState


def build_graph(checkpointer=None):
    """Build and compile the agent graph.

    Flow: process_input -> generate_response -> END
    """
    builder = StateGraph(AgentState)

    builder.add_node("process_input", process_input)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate_response", generate_response)
    builder.add_node("compress", compress)

    builder.set_entry_point("process_input")
    # builder.add_edge("process_input", "retrieve")
    builder.add_conditional_edges(
        "process_input",
        route_after_input,
        {"retrieve": "retrieve", "skip": "generate_response"},
    )
    builder.add_edge("retrieve", "generate_response")
    builder.add_conditional_edges(
        "generate_response",
        route_compress,
        {"compress": "compress", "end": END},
    )
    builder.add_edge("compress", END)

    return builder.compile(checkpointer=checkpointer)
