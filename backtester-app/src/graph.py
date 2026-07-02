from langgraph.graph import StateGraph, START, END
from .state import BacktestState, BuildEvent
from .agents import write_spec, write_code, judge, classify
from .runner import run_strategy, _load_strategy, run_pair_strategy
from .core.contributions import run_contributions, schedule_dates
from .core.data import load_prices
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

def _scope_fail(msg):                # out-of-scope refusal (terminal; the judge can't fix the REQUEST)
    return {"status": "failed", "scope_error": True,
            "run_result": {"passed": False, "metrics": {}, "failures": msg}}

def _fetch_leg_prices(tickers, start):
    """Fetch each distinct ticker (validate non-empty -> a bad/typo ticker raises). The fetch-time check
    IS the ticker validation (you can't whitelist arbitrary symbols)."""
    out = {}
    for tk in tickers:
        try:
            px = load_prices(tk, start=start) if start else load_prices(tk, period="5y")
        except Exception as e:
            raise ValueError(f"couldn't fetch data for '{tk}' ({type(e).__name__}) - check the ticker symbol")
        if px is None or len(px) == 0:
            raise ValueError(f"couldn't fetch data for '{tk}' - check the ticker symbol")
        out[tk] = px
    return out

def contribution_run(state):
    """CONTRIBUTION engine (the cash-flow tool): compare TWO deposit legs in dollars. Each leg has a
    cadence (signal | weekly | monthly), its own amount, and its own TICKER - so "GOOG monthly vs SPY
    monthly" (cross-asset) AND "weekly $250 vs monthly $1000" both work. Legacy default (no legs, e.g. the
    CLI/eval path): the coder's strategy(history) as the SIGNAL leg vs monthly DCA, both at `amount`, on the
    passed prices - identical to before (keeps the ground-truth eval intact)."""
    amount = state.get("amount") or 1000.0
    base_ticker = state.get("ticker") or "SPY"
    start = state.get("start_date")
    legs = state.get("legs") or [
        {"cadence": "signal", "amount": amount, "label": "Buy-the-signal"},
        {"cadence": "monthly", "amount": amount, "label": "Monthly DCA"},
    ]
    leg_tickers = [leg.get("ticker") or base_ticker for leg in legs]
    cross_asset = len(set(leg_tickers)) > 1

    # degenerate-comparison guard (Lesson 028 - detect-and-refuse): two IDENTICAL legs compare an asset to
    # ITSELF. Identity now includes the TICKER, so "GOOG monthly $1k vs SPY monthly $1k" is NOT identical.
    ident = [(leg_tickers[i], legs[i]["cadence"], legs[i]["amount"]) for i in range(len(legs))]
    if len(legs) >= 2 and all(x == ident[0] for x in ident):
        return _scope_fail("the two legs are IDENTICAL (same ticker + cadence + amount) - this compares an "
                           "asset to ITSELF. Vary the ticker, cadence, or amount.")
    # v1 scope gate: cross-asset supports CALENDAR cadences only (a signal leg needs the coder's single
    # strategy; "same signal on two assets" is a deliberate v2 increment).
    if cross_asset and any(leg.get("cadence") == "signal" for leg in legs):
        return _scope_fail("comparing two different ASSETS with a 'signal' cadence isn't supported yet - "
                           "use weekly/monthly cadences, or a single asset.")

    # resolve each leg's prices: cross-asset -> fetch per ticker + align to the COMMON window (fairness);
    # single ticker -> use the UI/eval-supplied prices, the CLI default, or fetch a single non-default ticker.
    warning = None
    if cross_asset:
        try:
            raw = _fetch_leg_prices(set(leg_tickers), start)
        except ValueError as e:
            return _scope_fail(str(e))
        eff_start = max(raw[t].index[0] for t in raw)        # latest first-date = common overlap window
        aligned = {t: px[px.index >= eff_start] for t, px in raw.items()}
        leg_prices = [aligned[t] for t in leg_tickers]
        earliest = min(raw[t].index[0] for t in raw)
        end = max(raw[t].index[-1] for t in raw)
        coverage = (end - eff_start).days / ((end - earliest).days or 1)
        if coverage < 0.80:                                  # overlap guard: warn (don't refuse) + show window
            warning = (f"the assets overlap for only {coverage:.0%} of the period - comparing from the common "
                       f"window starting {eff_start.date()}.")
    else:
        single = leg_tickers[0]
        prices = state.get("prices")
        if prices is not None and single == base_ticker:
            pass                                             # UI/eval supplied the right prices
        elif single == base_ticker:
            prices = _prices()                               # CLI default (cached)
        else:
            try:
                prices = _fetch_leg_prices({single}, start)[single]   # a single NON-default ticker
            except ValueError as e:
                return _scope_fail(str(e))
        leg_prices = [prices for _ in legs]

    needs_signal = any(leg.get("cadence") == "signal" for leg in legs)   # only then must the code be valid
    try:
        strategy = _load_strategy(state["strategy_code"]) if needs_signal else None
        computed = []
        for leg, lp in zip(legs, leg_prices):
            dates = schedule_dates(lp, leg["cadence"], strategy)
            if len(dates) == 0:          # inert guard (Lesson 025/029): a leg must actually fire
                return {"status": "failed",
                        "run_result": {"passed": False, "metrics": {},
                                       "failures": f"the '{leg.get('label', leg['cadence'])}' schedule NEVER "
                                                   "fires (0 deposits) - inert; check the signal logic against "
                                                   "the user's definition"}}
            curve, inv, final = run_contributions(lp, dates, leg["amount"])
            computed.append({**leg, "n": len(dates), "invested": inv, "final": final, "curve": curve})
    except Exception as e:
        print(f"[contribution] error: {e}")
        return {"status": "failed",                                  # "runtime error" -> judge routes to code
                "run_result": {"passed": False, "failures": f"runtime error (contribution): {e}", "metrics": {}}}

    # chart overlay (cosmetic): reindex both curves onto the UNION of dates + ffill so cross-asset legs with
    # different calendars (BTC weekends vs equity weekdays) plot as continuous lines, not NaN gaps. The
    # comparison NUMBERS come from each leg's own calendar above; only the plotted curve is reindexed.
    union = computed[0]["curve"].index.union(computed[1]["curve"].index)
    for c in computed:
        c["curve"] = c["curve"].reindex(union).ffill()

    a, b = computed[0], computed[1]
    result = {"amount": amount, "legs": computed, "warning": warning,
              # back-compat keys (the eval + render + export read these): leg A -> "signal", leg B -> "dca"
              "signal": {"n": a["n"], "invested": a["invested"], "final": a["final"]},
              "dca": {"n": b["n"], "invested": b["invested"], "final": b["final"]},
              "signal_curve": a["curve"], "dca_curve": b["curve"]}   # for the value-over-time chart
    print(f"[contribution] {a['label']} ${a['final']:,.0f}/${a['invested']:,.0f}  vs  "
          f"{b['label']} ${b['final']:,.0f}/${b['invested']:,.0f}")
    return {"status": "ok", "contribution_result": result}

def pairs_run(state):
    """PAIRS engine: fetch A + B, run the coder's strategy_pair through the spread engine."""
    ticker_a = state.get("ticker") or "SPY"
    ticker_b = state.get("ticker_b") or "QQQ"
    start = state.get("start_date")
    try:
        pa = load_prices(ticker_a, period="5y", start=start)
        pb = load_prices(ticker_b, period="5y", start=start)
        if len(pa) == 0 or len(pb) == 0:
            raise ValueError("empty data")
    except Exception as e:
        return {"status": "failed", "scope_error": True,
                "run_result": {"passed": False, "metrics": {},
                               "failures": f"couldn't fetch {ticker_a}/{ticker_b} - check the tickers ({e})"}}
    result = run_pair_strategy(state["strategy_code"], pa, pb)
    print(f"[pairs] {ticker_a}-{ticker_b} passed: {result['passed']}")
    pr = {"ticker_a": ticker_a, "ticker_b": ticker_b,
          "metrics": result["metrics"], "equity_curve": result.get("equity_curve")}
    return {"run_result": result, "pairs_result": pr,
            "status": "ok" if result["passed"] else "failed",
            "ledger": [BuildEvent("pairs", "result", result["failures"] or str(result["metrics"]))]}


def route_after_code(state):
    m = state.get("mode")
    if m == "contribution":
        return "contribution"
    if m == "pairs":
        return "pairs"
    return "position"

def route_after_run(state):
    if state["run_result"]["passed"]:
        return "done"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "done"                # escalate / give up
    return "judge"

def route_after_contribution(state):     # same self-healing brake as the position lane
    if state.get("status") == "ok":
        return "done"
    if state.get("scope_error"):         # out of scope -> the judge cannot fix the REQUEST; do not loop
        return "done"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "done"                # escalate / give up
    return "judge"

def route_after_pairs(state):            # same self-healing brake as the position lane
    if state["run_result"]["passed"]:
        return "done"
    if state.get("scope_error"):
        return "done"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "done"
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
    g.add_node("pairs_run", pairs_run)

    g.add_edge(START, "classify")        # M8: classify the request, then route to the right engine
    g.add_edge("classify", "write_spec")
    g.add_edge("write_spec", "write_code")
    # branch on mode: position -> the judge-looped sandbox; contribution -> the cash-flow engine
    g.add_conditional_edges("write_code", route_after_code,
                            {"position": "run", "contribution": "contribution_run", "pairs": "pairs_run"})
    g.add_conditional_edges("run", route_after_run, {"done": END, "judge": "judge"})
    g.add_conditional_edges("judge", dispatch, {"code": "write_code", "spec": "write_spec"})
    # contribution lane now self-heals too: failure -> judge -> write_code -> (route) -> contribution_run
    g.add_conditional_edges("contribution_run", route_after_contribution, {"done": END, "judge": "judge"})
    g.add_conditional_edges("pairs_run", route_after_pairs, {"done": END, "judge": "judge"})
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
    g.add_node("pairs_run", pairs_run)

    g.add_edge(START, "write_code")          # the spec is already in state (confirmed by the user)
    g.add_edge("write_spec", "write_code")
    g.add_conditional_edges("write_code", route_after_code,
                            {"position": "run", "contribution": "contribution_run", "pairs": "pairs_run"})
    g.add_conditional_edges("run", route_after_run, {"done": END, "judge": "judge"})
    g.add_conditional_edges("judge", dispatch, {"code": "write_code", "spec": "write_spec"})
    g.add_conditional_edges("contribution_run", route_after_contribution, {"done": END, "judge": "judge"})
    g.add_conditional_edges("pairs_run", route_after_pairs, {"done": END, "judge": "judge"})
    return g.compile()
