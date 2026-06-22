"""Graph assembly: wires nodes and edges into a compiled, runnable app."""

from langgraph.graph import END, StateGraph

from .nodes import generate_response, process_input
from .state import AgentState


def build_graph():
    """Build and compile the agent graph.

    Flow: process_input -> generate_response -> END
    """
    builder = StateGraph(AgentState)

    builder.add_node("process_input", process_input)
    builder.add_node("generate_response", generate_response)

    builder.set_entry_point("process_input")
    builder.add_edge("process_input", "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()
