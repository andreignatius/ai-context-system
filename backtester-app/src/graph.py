from langgraph.graph import StateGraph, START, END
from .state import BacktestState, BuildEvent
from .agents import write_spec, write_code, judge, classify
from .runner import run_strategy, _load_strategy
from .contributions import run_contributions, schedule_dates
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
    """CONTRIBUTION engine (the cash-flow tool): compare TWO deposit legs in dollars. Each leg has a
    cadence (signal | weekly | monthly) and its OWN amount, so "weekly $250 vs monthly $1000" works.
    Legacy default (no legs supplied, e.g. the CLI/eval path): the coder's strategy(history) as the
    SIGNAL leg vs monthly DCA, both at `amount` - identical to before (keeps the ground-truth eval intact)."""
    prices = state.get("prices")
    if prices is None:
        prices = _prices()
    amount = state.get("amount") or 1000.0
    legs = state.get("legs") or [
        {"cadence": "signal", "amount": amount, "label": "Buy-the-signal"},
        {"cadence": "monthly", "amount": amount, "label": "Monthly DCA"},
    ]
    needs_signal = any(leg.get("cadence") == "signal" for leg in legs)   # only then must the code be valid
    try:
        strategy = _load_strategy(state["strategy_code"]) if needs_signal else None
        computed = []
        for leg in legs:
            dates = schedule_dates(prices, leg["cadence"], strategy)
            if len(dates) == 0:          # inert guard (Lesson 025/029): a leg must actually fire
                return {"status": "failed",
                        "run_result": {"passed": False, "metrics": {},
                                       "failures": f"the '{leg.get('label', leg['cadence'])}' schedule NEVER "
                                                   "fires (0 deposits) - inert; check the signal logic against "
                                                   "the user's definition"}}
            curve, inv, final = run_contributions(prices, dates, leg["amount"])
            computed.append({**leg, "n": len(dates), "invested": inv, "final": final, "curve": curve})
    except Exception as e:
        print(f"[contribution] error: {e}")
        return {"status": "failed",                                  # "runtime error" -> judge routes to code
                "run_result": {"passed": False, "failures": f"runtime error (contribution): {e}", "metrics": {}}}
    a, b = computed[0], computed[1]
    result = {"amount": amount, "legs": computed,
              # back-compat keys (the eval + render + export read these): leg A -> "signal", leg B -> "dca"
              "signal": {"n": a["n"], "invested": a["invested"], "final": a["final"]},
              "dca": {"n": b["n"], "invested": b["invested"], "final": b["final"]},
              "signal_curve": a["curve"], "dca_curve": b["curve"]}   # for the value-over-time chart
    print(f"[contribution] {a['label']} ${a['final']:,.0f}/${a['invested']:,.0f}  vs  "
          f"{b['label']} ${b['final']:,.0f}/${b['invested']:,.0f}")
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
