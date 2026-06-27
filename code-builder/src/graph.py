"""Graph assembly for the code-builder: the sequential v1 flow.

Flow: write_spec -> write_code -> write_tests -> run_sandbox -> END
The three LLM agents live in agents.py; the sandbox node wraps the pytest runner.
"""

from langgraph.graph import START, END, StateGraph

from .state import BuilderState, BuildEvent
from .agents import write_spec, write_code, write_tests, judge
from .sandbox import run_tests
from .config import MAX_ATTEMPTS

def run_sandbox(state) -> dict:
    """Non-LLM node: run the QA tests against the code, record the verdict."""
    result = run_tests(state["code"], state["tests"])
    status = "ok" if result["passed"] else "failed"
    print(f"[sandbox] passed: {result['passed']}")
    return {"test_result": result, "status": status,
            "ledger": [BuildEvent("sandbox", "result", f"passed={result['passed']}\n{result['failures']}")]}

# def route_after_tests(state) -> str:
#     """After the sandbox. NOTE: within a build the ONLY automatic recovery is the CODER
#     retrying - tests are held fixed here. Whether the CODE or the TESTS is at fault is the
#     HUMAN's call, between builds (see dispatch / main.py's 'fix which?')."""
#     res = state["test_result"]
#     if res["passed"]:
#         print("[router] all tests pass -> DONE")
#         return "done"

#     attempts = state.get("attempts", 0)
#     failed = [ln for ln in res["failures"].splitlines() if ln.startswith(("FAILED", "ERROR"))]
#     if not failed:
#         failed = ["(no FAILED/ERROR lines parsed - see the stuck-report for the full log)"]

#     if attempts >= MAX_ATTEMPTS:
#         print(f"[router] still red after {attempts} coder attempts -> GIVE UP, hand to the human")
#         for ln in failed:
#             print(f"         {ln}")
#         return "done"

#     print(f"[router] tests failed -> auto-retry the CODER (attempt {attempts}/{MAX_ATTEMPTS}); "
#           "tests held fixed, only the code is rewritten here")
#     for ln in failed:
#         print(f"         {ln}")
#     return "retry"

def route_after_sandbox(state) -> str:
    if state["test_result"]["passed"]:
        print("[router] all tests pass -> DONE")
        return "done"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        print(f"[router] still red after {state.get('attempts', 0)} rounds -> hand to the human")
        return "done"
    return "judge"


def dispatch(state) -> str:
    """Entry router: WHERE the invoke starts = which agent acts first. The human's fix_target
    picks the door; the graph's normal edges run from there."""
    t = state.get("fix_target", "")
    if t == "tests":
        print("[plan] target=TESTS -> QA edits test_solution.py, then re-run pytest "
              "(spec pinned; coder fixes solution.py only if the corrected tests expose a code bug)")
        return "write_tests"
    if t == "code":
        print("[plan] target=CODE -> coder edits solution.py (spec + tests pinned)")
        return "write_code"
    if t == "spec":
        print("[plan] target=SPEC -> orchestrator rewrites the spec; tests + code then REGENERATE from it")
        return "write_spec"
    print("[plan] fresh build -> orchestrator(spec) -> QA(test_solution.py) -> coder(solution.py)")
    return "write_spec"


def after_write_tests(state) -> str:
    """A tests-only fix REUSES the existing code -> straight to sandbox; otherwise the
    normal pipeline continues to the coder."""
    return "run_sandbox" if state.get("fix_target") == "tests" else "write_code"

def build_graph():
    builder = StateGraph(BuilderState)

    builder.add_node("write_spec", write_spec)
    builder.add_node("write_code", write_code)
    builder.add_node("write_tests", write_tests)
    builder.add_node("run_sandbox", run_sandbox)
    builder.add_node("judge", judge)

    builder.add_conditional_edges(START, dispatch, {       # <- replaces set_entry_point
        "write_spec": "write_spec",
        "write_tests": "write_tests",
        "write_code": "write_code",
    })

    # builder.set_entry_point("write_spec")
    # builder.add_edge("write_spec", "write_tests") # tests first (TDD, fixed target)
    # builder.add_conditional_edges("write_tests", after_write_tests, {   # <- replaces the static edge
    #     "run_sandbox": "run_sandbox",
    #     "write_code": "write_code",
    # })
    # builder.add_edge("write_tests", "write_code")
    # builder.add_edge("write_code", "run_sandbox")
    
    # FRESH build: spec fans OUT to tests + code (run concurrently); both fan IN to the sandbox barrier.
    builder.add_edge("write_spec", "write_tests")
    builder.add_edge("write_spec", "write_code")
    builder.add_edge("write_tests", "run_sandbox")
    builder.add_edge("write_code",  "run_sandbox")

    builder.add_conditional_edges("run_sandbox", route_after_sandbox, {   # was route_after_tests
        "done": END,
        "judge": "judge",
    })
    builder.add_conditional_edges("judge", dispatch, {                    # NEW: route to the culprit
        "write_spec": "write_spec",
        "write_tests": "write_tests",
        "write_code": "write_code",
    })

    return builder.compile()
