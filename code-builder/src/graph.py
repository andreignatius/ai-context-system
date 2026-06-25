"""Graph assembly for the code-builder: the sequential v1 flow.

Flow: write_spec -> write_code -> write_tests -> run_sandbox -> END
The three LLM agents live in agents.py; the sandbox node wraps the pytest runner.
"""

from langgraph.graph import END, StateGraph

from .state import BuilderState, BuildEvent
from .agents import write_spec, write_code, write_tests
from .sandbox import run_tests
from .config import MAX_ATTEMPTS

def run_sandbox(state) -> dict:
    """Non-LLM node: run the QA tests against the code, record the verdict."""
    result = run_tests(state["code"], state["tests"])
    status = "ok" if result["passed"] else "failed"
    print(f"[sandbox] passed: {result['passed']}")
    return {"test_result": result, "status": status,
            "ledger": [BuildEvent("sandbox", "result", f"passed={result['passed']}\n{result['failures']}")]}

def route_after_tests(state) -> str:
    # after the sandbox: 'done' if passed or out of attempts else 'retry'
    if state["test_result"]["passed"]:
        return "done"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        print(f"[router] giving up after {state['attempts']} attempts")
        return "done"
    print(f"[router] tests failed - retrying (attempt {state.get('attempts', 0)}/{MAX_ATTEMPTS})")
    return "retry"

def build_graph():
    builder = StateGraph(BuilderState)

    builder.add_node("write_spec", write_spec)
    builder.add_node("write_code", write_code)
    builder.add_node("write_tests", write_tests)
    builder.add_node("run_sandbox", run_sandbox)

    builder.set_entry_point("write_spec")
    builder.add_edge("write_spec", "write_tests") # tests first (TDD, fixed target)
    builder.add_edge("write_tests", "write_code")
    builder.add_edge("write_code", "run_sandbox")
    builder.add_conditional_edges(
        "run_sandbox",
        route_after_tests,
        {"retry": "write_code", "done": END},
    )

    return builder.compile()
