"""Ground-truth eval: does the AI's strategy MATCH a known-correct reference?
Soundness (the sandbox) checks "does it run". This checks "is it the right strategy" -
by running the AI's code and a hand-written reference through the SAME engine and
comparing. This is what catches sound-but-wrong strategies (e.g. iloc[0] vs iloc[-21])."""
from .graph import build_graph
from .runner import _load_strategy
from .engine import run_backtest
from .data import load_prices

app = build_graph()
PRICES = load_prices("SPY", "2y")


# --- the answer key: KNOWN-CORRECT reference implementations (Andre's ground truth) ---
def ref_buy_hold(history):
    return 1.0

def ref_20day_return(history):
    if len(history) < 21:
        return 0.0
    return 1.0 if history.iloc[-1] > history.iloc[-21] else 0.0   # 20-day return > 0

def ref_sma_cross(history):
    if len(history) < 50:
        return 0.0
    return 1.0 if history.iloc[-20:].mean() > history.iloc[-50:].mean() else 0.0

GOLDEN = [
    ("always stay fully long (buy and hold)", ref_buy_hold),
    ("go long when the 20-day return is positive, else flat", ref_20day_return),
    ("go long when the 20-day average is above the 50-day average, else flat", ref_sma_cross),
]

def evaluate(request, reference, tol_ret=0.02, tol_sharpe=0.15):
    result = app.invoke({"request": request})
    if result["status"] != "ok":                      # didn't even pass soundness
        return {"verdict": "UNSOUND", "detail": result["run_result"]["failures"]}
    # the ruler grades LOGIC (does the code match the documented rule), not costs -> run both GROSS (fee=0)
    # so the comparison is exact; transaction cost is a separate honesty gate applied to the user-facing run.
    ai = run_backtest(PRICES, _load_strategy(result["strategy_code"]), fee=0.0)
    ref = run_backtest(PRICES, reference, fee=0.0)
    dt, ds = abs(ai.total_return - ref.total_return), abs(ai.sharpe - ref.sharpe)
    correct = dt < tol_ret and ds < tol_sharpe
    return {"verdict": "CORRECT" if correct else "SOUND-BUT-WRONG",
            "ai": (ai.total_return, ai.sharpe), "ref": (ref.total_return, ref.sharpe),
            "diff": (dt, ds)}

N_RUNS = 5      # runs per case; temp=0.5 makes the model non-deterministic -> a pass RATE

if __name__ == "__main__":
    print(f"Ground-truth eval: {len(GOLDEN)} cases x {N_RUNS} runs (SPY 2y)\n")
    for request, reference in GOLDEN:
        verdicts = []
        for i in range(N_RUNS):
            r = evaluate(request, reference)
            verdicts.append(r["verdict"])
            extra = f"  (diff total={r['diff'][0]:.3f})" if r["verdict"] != "UNSOUND" else ""
            print(f"  run {i+1}/{N_RUNS}: {r['verdict']:16s}{extra}")
        c = verdicts.count("CORRECT")
        w = verdicts.count("SOUND-BUT-WRONG")
        u = verdicts.count("UNSOUND")
        print(f"==> {request[:50]}\n    {c}/{N_RUNS} CORRECT | {w} sound-but-wrong | {u} unsound\n")

