from langgraph.graph import StateGraph, START, END
from .state import BacktestState, BuildEvent
from .agents import write_spec, write_code, judge
from .runner import run_strategy
from .data import load_prices
from .config import MAX_ATTEMPTS

_cache = {}
def _prices():
    if "p" not in _cache:            # load once, on first run (not at import)
        _cache["p"] = load_prices("SPY", "2y")
    return _cache["p"]

def run_node(state):
    result = run_strategy(state["strategy_code"], _prices())
    print(f"[sandbox] passed: {result['passed']}")
    return {"run_result": result,
            "status": "ok" if result["passed"] else "failed",
            "ledger": [BuildEvent("sandbox", "result",
                                  result["failures"] or str(result["metrics"]))]}

def route_after_run(state):
    if state["run_result"]["passed"]:
        return "done"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "done"                # escalate / give up
    return "judge"

def dispatch(state):
    return state["fix_target"]       # "code" | "spec"

def build_graph():
    g = StateGraph(BacktestState)
    g.add_node("write_spec", write_spec)
    g.add_node("write_code", write_code)
    g.add_node("run", run_node)
    g.add_node("judge", judge)

    g.add_edge(START, "write_spec")
    g.add_edge("write_spec", "write_code")
    g.add_edge("write_code", "run")
    g.add_conditional_edges("run", route_after_run, {"done": END, "judge": "judge"})
    g.add_conditional_edges("judge", dispatch, {"code": "write_code", "spec": "write_spec"})
    return g.compile()
