from langgraph.graph import StateGraph, START, END
from .state import BacktestState, BuildEvent
from .agents import write_spec, write_code, judge, classify
from .runner import run_strategy, _load_strategy
from .contributions import run_contributions, monthly_dates, signal_dates
from .data import load_prices
from .config import MAX_ATTEMPTS

_cache = {}
def _prices(ticker="SPY", period="2y"):
    key = (ticker, period)           # load once per (ticker, period), on first run (not at import)
    if key not in _cache:
        _cache[key] = load_prices(ticker, period)
    return _cache[key]

def run_node(state):
    prices = state.get("prices")     # UI supplies the chosen ticker/period; CLI + evals use the default
    if prices is None:
        prices = _prices()
    result = run_strategy(state["strategy_code"], prices)
    print(f"[sandbox] passed: {result['passed']}")
    return {"run_result": result,
            "status": "ok" if result["passed"] else "failed",
            "ledger": [BuildEvent("sandbox", "result",
                                  result["failures"] or str(result["metrics"]))]}

def contribution_run(state):
    """CONTRIBUTION engine (the cash-flow tool): the coder's strategy(history) is the deposit SIGNAL.
    Deposit $amount on each signal bar, and on the first of each month (DCA), and compare in dollars."""
    prices = state.get("prices")
    if prices is None:
        prices = _prices()
    amount = state.get("amount") or 1000.0
    try:
        strategy = _load_strategy(state["strategy_code"])
        sig_dates = signal_dates(prices, strategy)
        if len(sig_dates) == 0:          # inert guard (Lesson 025/029): the signal must actually fire
            return {"status": "failed",
                    "run_result": {"passed": False, "metrics": {},
                                   "failures": "the deposit signal NEVER fires (0 deposits) - inert "
                                               "strategy; check the signal logic against the user's definition"}}
        mon_dates = monthly_dates(prices)
        sig_curve, sig_inv, sig_final = run_contributions(prices, sig_dates, amount)
        dca_curve, dca_inv, dca_final = run_contributions(prices, mon_dates, amount)
    except Exception as e:
        print(f"[contribution] error: {e}")
        return {"status": "failed",                                  # "runtime error" -> judge routes to code
                "run_result": {"passed": False, "failures": f"runtime error (contribution): {e}", "metrics": {}}}
    result = {"amount": amount,
              "signal": {"n": len(sig_dates), "invested": sig_inv, "final": sig_final},
              "dca": {"n": len(mon_dates), "invested": dca_inv, "final": dca_final},
              "signal_curve": sig_curve, "dca_curve": dca_curve}   # for the value-over-time chart
    print(f"[contribution] signal ${sig_final:,.0f}/${sig_inv:,.0f}  vs  DCA ${dca_final:,.0f}/${dca_inv:,.0f}")
    return {"status": "ok", "contribution_result": result}

def route_after_code(state):
    return "contribution" if state.get("mode") == "contribution" else "position"

def route_after_run(state):
    if state["run_result"]["passed"]:
        return "done"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "done"                # escalate / give up
    return "judge"

def route_after_contribution(state):     # same self-healing brake as the position lane
    if state.get("status") == "ok":
        return "done"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "done"                # escalate / give up
    return "judge"

def dispatch(state):
    return state["fix_target"]       # "code" | "spec"

def build_graph():
    g = StateGraph(BacktestState)
    g.add_node("classify", classify)
    g.add_node("write_spec", write_spec)
    g.add_node("write_code", write_code)
    g.add_node("run", run_node)
    g.add_node("judge", judge)
    g.add_node("contribution_run", contribution_run)

    g.add_edge(START, "classify")        # M8: classify the request, then route to the right engine
    g.add_edge("classify", "write_spec")
    g.add_edge("write_spec", "write_code")
    # branch on mode: position -> the judge-looped sandbox; contribution -> the cash-flow engine
    g.add_conditional_edges("write_code", route_after_code,
                            {"position": "run", "contribution": "contribution_run"})
    g.add_conditional_edges("run", route_after_run, {"done": END, "judge": "judge"})
    g.add_conditional_edges("judge", dispatch, {"code": "write_code", "spec": "write_spec"})
    # contribution lane now self-heals too: failure -> judge -> write_code -> (route) -> contribution_run
    g.add_conditional_edges("contribution_run", route_after_contribution, {"done": END, "judge": "judge"})
    return g.compile()


def build_run_graph():
    """The RUN half for the UI's confirm flow: starts from an ALREADY-CONFIRMED spec (skips classify +
    write_spec; the UI does those as the 'draft'). write_spec is kept only so the judge's 'spec' verdict
    during self-heal still has somewhere to go."""
    g = StateGraph(BacktestState)
    g.add_node("write_spec", write_spec)
    g.add_node("write_code", write_code)
    g.add_node("run", run_node)
    g.add_node("judge", judge)
    g.add_node("contribution_run", contribution_run)

    g.add_edge(START, "write_code")          # the spec is already in state (confirmed by the user)
    g.add_edge("write_spec", "write_code")
    g.add_conditional_edges("write_code", route_after_code,
                            {"position": "run", "contribution": "contribution_run"})
    g.add_conditional_edges("run", route_after_run, {"done": END, "judge": "judge"})
    g.add_conditional_edges("judge", dispatch, {"code": "write_code", "spec": "write_spec"})
    g.add_conditional_edges("contribution_run", route_after_contribution, {"done": END, "judge": "judge"})
    return g.compile()
