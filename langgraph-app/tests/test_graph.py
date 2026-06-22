"""Unit tests for graph structure.

These do NOT hit the LLM — ChatOllama connects lazily, so building/compiling the
graph is offline and fast. An end-to-end test that calls Ollama would be an
integration test (mark it and run it only when Ollama is up).
"""

from langchain_core.messages import HumanMessage

from src.graph import build_graph
from src.nodes import process_input
from src.state import AgentState


def test_graph_compiles():
    app = build_graph()
    assert app is not None


def test_state_has_expected_keys():
    assert set(AgentState.__annotations__) == {"messages", "scratchpad", "query"}


def test_process_input_wraps_query_as_human_message():
    out = process_input({"query": "hello", "messages": [], "scratchpad": ""})
    assert len(out["messages"]) == 1
    msg = out["messages"][0]
    assert isinstance(msg, HumanMessage)
    assert msg.content == "hello"
