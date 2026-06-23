"""Unit tests for graph structure.

These do NOT hit the LLM — ChatOllama connects lazily, so building/compiling the
graph is offline and fast. An end-to-end test that calls Ollama would be an
integration test (mark it and run it only when Ollama is up).
"""

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph import build_graph
from src.nodes import process_input
from src.state import AgentState


def test_graph_compiles():
    app = build_graph()
    assert app is not None


def test_state_has_expected_keys():
    assert set(AgentState.__annotations__) == {"messages", "scratchpad", "query"}


def test_process_input_prepends_system_then_human_on_first_turn():
    out = process_input({"query": "hello", "messages": [], "scratchpad": ""})
    # First turn: system prompt first, then the human message.
    assert len(out["messages"]) == 2
    assert isinstance(out["messages"][0], SystemMessage)
    assert isinstance(out["messages"][1], HumanMessage)
    assert out["messages"][1].content == "hello"


def test_process_input_does_not_add_second_system_message():
    # If a SystemMessage already exists, we must NOT add another one.
    existing = [SystemMessage(content="already here"), HumanMessage(content="hi")]
    out = process_input({"query": "next question", "messages": existing, "scratchpad": ""})
    # Only the new human message is returned - no new system message.
    assert len(out["messages"]) == 1
    assert isinstance(out["messages"][0], HumanMessage)
