"""Multi-baseline RULER (the "expectation vs reality" experiment) - generalizes check_contribution_eval.py
from ONE baseline to a SUITE. For each idea we DOCUMENT the expected point-in-time rule (the baseline IS the
spec), then run the agent N times and grade its output against it. The model is a clean variable: set
LLM_PROVIDER / DEEPINFRA_MODEL in the env (7B local vs DeepSeek-V3) and re-run - same baselines, same data.

Grading (position ideas): compare the agent's target series to the baseline's, bar by bar, over the
post-warm-up window. CORRECT = high agreement; DIVERGENT = runs but wrong rule; UNSOUND = inert/crash;
MISROUTED = wrong paradigm. This catches the off-by-one look-ahead (breakout) and indicator bugs (RSI)
that an outcome-only check would miss.

Run:  LLM_PROVIDER=deepinfra DEEPINFRA_MODEL=deepseek-ai/DeepSeek-V3 python eval_suite.py
      python eval_suite.py            # local Ollama 7B (default)
"""
import os
import math
from collections import Counter

import yfinance as yf
import pandas as pd

from src.graph import build_graph
from src.runner import _load_strategy
from src.config import LLM_PROVIDER

N_RUNS = int(os.getenv("EVAL_N", "3"))


# --- the documented EXPECTATIONS (each baseline is the ground-truth point-in-time rule) ---
def sma_baseline(h):
    """#4 SMA 50/200 regime: long while the 50-day SMA is above the 200-day SMA, else flat. Long-only."""
    if len(h) < 200:
        return 0.0
    return 1.0 if h.iloc[-50:].mean() > h.iloc[-200:].mean() else 0.0


def breakout_baseline(h):
    """#7 20-day breakout: long when today's close exceeds the highest close of the PRIOR 20 days
    (EXCLUDING today - the off-by-one test), else flat. Long-only."""
    if len(h) < 21:
        return 0.0
    prior = h.iloc[-21:-1]                 # the prior 20 bars, NOT including the current bar
    return 1.0 if h.iloc[-1] > prior.max() else 0.0


def rsi_baseline(h):
    """#5 RSI(14) Wilder: long when RSI < 30 (oversold), short when RSI > 70 (overbought), else flat."""
    n = 14
    if len(h) < n + 1:
        return 0.0
    diff = h.diff().dropna()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0 / n, adjust=False).mean().iloc[-1]
    al = loss.ewm(alpha=1.0 / n, adjust=False).mean().iloc[-1]
    rsi = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    if rsi < 30:
        return 1.0
    if rsi > 70:
        return -1.0
    return 0.0


IDEAS = [
    {"key": "SMA 50/200", "paradigm": "position", "warmup": 205, "thresh": 0.98, "baseline": sma_baseline,
     "prompt": "Backtest SPY: go long (fully invested) when the 50-day simple moving average is above the "
               "200-day simple moving average, and go flat otherwise. Long-only, using daily close."},
    {"key": "20d breakout", "paradigm": "position", "warmup": 25, "thresh": 0.97, "baseline": breakout_baseline,
     "prompt": "Backtest SPY: go long when today's close is above the highest close of the PRIOR 20 trading "
               "days (a 20-day high breakout), and go flat otherwise. Long-only, using daily close."},
    {"key": "RSI 14 30/70", "paradigm": "position", "warmup": 40, "thresh": 0.90, "baseline": rsi_baseline,
     "prompt": "Backtest SPY: compute the 14-day RSI. Go long when RSI is below 30 (oversold) and go short "
               "when RSI is above 70 (overbought); otherwise go flat. Use daily close."},
]


def grade_position(prices, code, base_targets, warmup, thresh):
    try:
        strat = _load_strategy(code)
        at = [strat(prices.iloc[: t + 1]) for t in range(len(prices))]
    except Exception as e:
        return "UNSOUND", f"{type(e).__name__}: {str(e)[:40]}"
    if any((not math.isfinite(x)) or abs(x) > 1.0 + 1e-9 for x in at):
        return "UNSOUND", "non-finite / out-of-range positions"
    rng = range(warmup, len(prices))
    denom = len(list(rng))
    match = sum(1 for t in rng if abs(at[t] - base_targets[t]) < 0.01) / denom
    a_act = sum(1 for t in rng if abs(at[t]) > 0.01) / denom
    b_act = sum(1 for t in rng if abs(base_targets[t]) > 0.01) / denom
    if a_act < 0.02 and b_act > 0.10:
        return "UNSOUND", f"inert (agent active {a_act:.0%} vs baseline {b_act:.0%})"
    if match >= thresh:
        return "CORRECT", f"{match:.0%} match"
    return "DIVERGENT", f"{match:.0%} match (agent active {a_act:.0%}, baseline {b_act:.0%})"


def main():
    model = os.getenv("DEEPINFRA_MODEL", "qwen2.5-coder:7B(local)") if LLM_PROVIDER == "deepinfra" else "ollama 7B"
    print(f"\n{'='*70}\nSUITE RULER   provider={LLM_PROVIDER}   model={model}   N={N_RUNS}\n{'='*70}")

    prices = yf.download("SPY", start="2021-01-01", auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices = prices.dropna()
    print(f"data: SPY  {prices.index[0].date()} -> {prices.index[-1].date()}  ({len(prices)} bars)\n")

    app = build_graph()
    summary = {}
    for idea in IDEAS:
        base_targets = [idea["baseline"](prices.iloc[: t + 1]) for t in range(len(prices))]
        print(f"--- {idea['key']}  (expect {idea['paradigm']}, >={idea['thresh']:.0%} target agreement) ---")
        verdicts = []
        for i in range(N_RUNS):
            try:
                r = app.invoke({"request": idea["prompt"], "prices": prices, "amount": 1000.0})
            except Exception as e:
                verdicts.append("UNSOUND")
                print(f"  run {i+1}/{N_RUNS}: UNSOUND      graph error: {str(e)[:40]}")
                continue
            if r.get("mode") != idea["paradigm"]:
                verdicts.append("MISROUTED")
                print(f"  run {i+1}/{N_RUNS}: MISROUTED    routed to {r.get('mode')}")
                continue
            v, detail = grade_position(prices, r.get("strategy_code", ""),
                                       base_targets, idea["warmup"], idea["thresh"])
            verdicts.append(v)
            print(f"  run {i+1}/{N_RUNS}: {v:11} {detail}")
        c = Counter(verdicts)
        summary[idea["key"]] = c
        print(f"  => {c['CORRECT']}/{N_RUNS} correct\n")

    print(f"{'='*70}\nSUMMARY  ({LLM_PROVIDER} / {model})")
    for key, c in summary.items():
        bits = "  ".join(f"{k}:{v}" for k, v in c.items())
        print(f"  {key:16} {c['CORRECT']}/{N_RUNS} correct   ({bits})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
