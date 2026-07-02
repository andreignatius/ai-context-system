"""Ground-truth eval for the CONTRIBUTION flow: run the AGENT N times on the buy-the-dip prompt and
grade each vs the human-verified BASELINE (baselines/buy_the_dip.py). Needs Ollama + network.

The 'ruler': catches the bugs we kept eyeballing (3% / 5%-peak / over-firing) by comparing the agent's
deposit-count + per-dollar multiple to the documented baseline. N runs -> a pass RATE."""
from collections import Counter

import yfinance as yf
import pandas as pd
from baselines.buy_the_dip import is_dip, run_contributions
from src.graph import build_graph

PROMPT = ("let's try to buy the dip - if there is a drawdown 3 consecutive trading days in a row I put in "
          "$1000, since the first trading day of 2021, how much money would I have at the end, compared to "
          "DCA $1000 every first trading day of the month")
AMOUNT = 1000.0
N_RUNS = 5
TOL_DEPOSITS = 3        # deposit count within +/- 3
TOL_MULT = 0.03         # per-dollar multiple within 0.03

# GUARD REQUIRED: app.invoke routes the buy-the-dip prompt through the sandboxed signal leg, which SPAWNS a
# child that re-imports THIS module. Without the `if __name__` guard the re-import re-runs everything below
# -> a recursive spawn during bootstrap -> RuntimeError. Any module that reaches the sandbox at module scope
# must be __main__-guarded (see docs/sandbox-plan.md, "Caller invariant").
if __name__ == "__main__":
    # --- data: SPY since 2021 (identical for baseline + agent) ---
    prices = yf.download("SPY", start="2021-01-01", auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices = prices.dropna()

    # --- BASELINE (truth) on these prices ---
    base_dates = prices.index[[is_dip(prices.iloc[: t + 1]) for t in range(len(prices))]]
    _, base_inv, base_final = run_contributions(prices, base_dates, AMOUNT)
    base = {"n": len(base_dates), "mult": base_final / base_inv}
    print(f"BASELINE: {base['n']} deposits / {base['mult']:.2f}x   (SPY since 2021, $1k/deposit)\n")

    # --- run the AGENT N times ---
    app = build_graph()
    verdicts = []
    for i in range(N_RUNS):
        r = app.invoke({"request": PROMPT, "prices": prices, "amount": AMOUNT})
        if r.get("mode") != "contribution":
            v, detail = "MISROUTED", ""
        elif r.get("status") != "ok":
            v, detail = "UNSOUND", str(r.get("run_result", {}).get("failures", ""))[:50]
        else:
            sig = r["contribution_result"]["signal"]
            n = sig["n"]
            mult = sig["final"] / sig["invested"] if sig["invested"] else 0.0
            ok = abs(n - base["n"]) <= TOL_DEPOSITS and abs(mult - base["mult"]) < TOL_MULT
            v, detail = ("CORRECT" if ok else "SOUND-BUT-WRONG"), f"{n} dep / {mult:.2f}x"
        verdicts.append(v)
        print(f"  run {i + 1}/{N_RUNS}: {v:16} {detail}")

    c = Counter(verdicts)
    print(f"\n{c['CORRECT']}/{N_RUNS} CORRECT | {c['SOUND-BUT-WRONG']} sound-but-wrong | "
          f"{c['UNSOUND']} unsound | {c['MISROUTED']} misrouted")
