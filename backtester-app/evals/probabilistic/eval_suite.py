"""Multi-baseline RULER (the "expectation vs reality" experiment) - generalizes check_contribution_eval.py
from ONE baseline to a SUITE. For each idea we DOCUMENT the expected point-in-time rule(s) (the baseline IS
the spec), then run the agent N times and grade its output against them. The model is a clean variable: set
LLM_PROVIDER / DEEPINFRA_MODEL in the env (7B local vs 32B vs DeepSeek-V3) and re-run - same baselines, data.

Grading (position ideas): compare the agent's target series to the baseline's, bar by bar, over the
post-warm-up window. An idea may list SEVERAL acceptable baselines (e.g. Wilder vs simple-average RSI - both
are valid "RSI"); the agent is graded against the CLOSEST. Verdicts:
  CORRECT   - matches an accepted definition (>= thresh). Detail names WHICH variant it matched.
  NEAR      - close to the best baseline but not exact, with sane activity -> REVIEW (a valid variant we did
              not list, OR a subtle bug like the breakout off-by-one). The grader does NOT claim to know which.
  BROKEN    - far from every baseline, or wildly wrong activity (e.g. in a trade ~every bar).
  UNSOUND   - crashes / inert / non-finite / out-of-range positions.
  MISROUTED - wrong paradigm (position vs contribution).
This separates "correct-but-different-definition" from "broken" (2a) and stops penalizing valid RSI variants
(2b). It deliberately does NOT auto-label "variant" vs "subtle bug" - that needs a human/judge; NEAR flags it.

Run (from the backtester-app/ root, as a module):
  LLM_PROVIDER=deepinfra DEEPINFRA_MODEL=deepseek-ai/DeepSeek-V3 python -m evals.probabilistic.eval_suite
  LLM_PROVIDER=deepinfra DEEPINFRA_MODEL=Qwen/Qwen3-32B python -m evals.probabilistic.eval_suite
  python -m evals.probabilistic.eval_suite            # local Ollama 7B (default)
"""
import os
import math
from collections import Counter

import yfinance as yf
import numpy as np
import pandas as pd

from src.graph import build_graph
from src.runner import _load_strategy, _load_strategy_pair
from src.core.engine import compute_targets
from src.core.pairs import compute_pair_targets
from src.config import LLM_PROVIDER

N_RUNS = int(os.getenv("EVAL_N", "3"))
NEAR_FLOOR = 0.70          # below thresh but >= this (with sane activity) -> NEAR (review), else BROKEN
ACT_RATIO_MAX = 3.0        # agent-active / baseline-active beyond this -> BROKEN (structural mismatch)


# --- the documented EXPECTATIONS (each baseline is a ground-truth point-in-time rule) ---
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


def rsi_wilder(h):
    """RSI(14), WILDER smoothing (EMA alpha=1/14 - the canonical RSI): long < 30, short > 70, else flat."""
    n = 14
    if len(h) < n + 1:
        return 0.0
    diff = h.diff().dropna()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0 / n, adjust=False).mean().iloc[-1]
    al = loss.ewm(alpha=1.0 / n, adjust=False).mean().iloc[-1]
    rsi = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return 1.0 if rsi < 30 else (-1.0 if rsi > 70 else 0.0)


def rsi_simple(h):
    """RSI(14), SIMPLE 14-bar average (Cutler's - what most LLMs emit): long < 30, short > 70, else flat."""
    n = 14
    if len(h) < n + 1:
        return 0.0
    diff = h.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    ag = gain.iloc[-n:].mean()
    al = loss.iloc[-n:].mean()
    rsi = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return 1.0 if rsi < 30 else (-1.0 if rsi > 70 else 0.0)


def pairs_zscore(a, b, w=60, entry=2.0):
    """PAIRS baseline (baselines/pairs.py): LOG-spread z-score mean-reversion, point-in-time. +1 long the
    spread (long A / short B) when z < -2; -1 short the spread when z > +2; else flat."""
    if len(a) < w + 1:
        return 0.0
    spread = np.log(a) - np.log(b)
    sd = spread.tail(w).std()
    if not sd > 0:
        return 0.0
    z = (spread.iloc[-1] - spread.tail(w).mean()) / sd
    return 1.0 if z < -entry else (-1.0 if z > entry else 0.0)


IDEAS = [
    {"key": "SMA 50/200", "paradigm": "position", "warmup": 205, "thresh": 0.98,
     "baselines": [("sma", sma_baseline)],
     "prompt": "Backtest SPY: go long (fully invested) when the 50-day simple moving average is above the "
               "200-day simple moving average, and go flat otherwise. Long-only, using daily close."},
    {"key": "20d breakout", "paradigm": "position", "warmup": 25, "thresh": 0.97,
     "baselines": [("prior-20-excl-today", breakout_baseline)],
     "prompt": "Backtest SPY: go long when today's close is above the highest close of the PRIOR 20 trading "
               "days (a 20-day high breakout), and go flat otherwise. Long-only, using daily close."},
    # 2b: RSI is underspecified -> accept BOTH valid variants. Correct code is CORRECT either way.
    {"key": "RSI 14 (any valid)", "paradigm": "position", "warmup": 40, "thresh": 0.90,
     "baselines": [("wilder", rsi_wilder), ("simple-MA", rsi_simple)],
     "prompt": "Backtest SPY: compute the 14-day RSI. Go long when RSI is below 30 (oversold) and go short "
               "when RSI is above 70 (overbought); otherwise go flat. Use daily close."},
    # 2b diagnostic: PIN the variant in the prompt -> only Wilder is accepted. Tests spec-FOLLOWING
    # (does the model override its simple-MA default when told?).
    {"key": "RSI 14 (Wilder-pinned)", "paradigm": "position", "warmup": 40, "thresh": 0.90,
     "baselines": [("wilder", rsi_wilder)],
     "prompt": "Backtest SPY: compute the 14-day RSI using WILDER'S smoothing (an exponential moving average "
               "with alpha = 1/14, the standard RSI - NOT a simple average). Go long when RSI is below 30 and "
               "short when RSI is above 70; otherwise go flat. Use daily close."},
    # PAIRS engine - finally graded (was the untested third engine). paradigm="pairs" -> grade_pairs on TWO
    # series; classify now ROUTES here (was UI-only, so the CLI/eval could never reach it).
    {"key": "pairs XLF/XLI (log-z)", "paradigm": "pairs", "warmup": 65, "thresh": 0.85,
     "ticker_a": "XLF", "ticker_b": "XLI", "pairs_baseline": pairs_zscore,
     "prompt": "Pairs trade XLF vs XLI on the LOG-spread. Compute z = (log(XLF) - log(XLI), minus its 60-day "
               "rolling mean) / its 60-day rolling standard deviation. Go LONG the spread (long XLF, short XLI) "
               "when z < -2, SHORT the spread when z > +2, otherwise flat. Daily close."},
]


def grade_position(prices, code, idea):
    warmup, thresh = idea["warmup"], idea["thresh"]
    try:
        strat = _load_strategy(code)
        at = compute_targets(prices, strat).tolist()      # positional list, indexed at[t] below
    except Exception as e:
        return "UNSOUND", f"{type(e).__name__}: {str(e)[:40]}"
    if any((not math.isfinite(x)) or abs(x) > 1.0 + 1e-9 for x in at):
        return "UNSOUND", "non-finite / out-of-range positions"
    rng = list(range(warmup, len(prices)))
    n = len(rng)
    a_act = sum(1 for t in rng if abs(at[t]) > 0.01) / n

    best = (-1.0, None, 0.0)                              # (match, baseline name, that baseline's activity)
    for name, _fn in idea["baselines"]:
        bt = idea["_btargets"][name]
        m = sum(1 for t in rng if abs(at[t] - bt[t]) < 0.01) / n
        if m > best[0]:
            best = (m, name, sum(1 for t in rng if abs(bt[t]) > 0.01) / n)
    match, b_name, b_act = best
    ratio = (a_act / b_act) if b_act > 0 else (float("inf") if a_act > 0.02 else 1.0)

    if a_act < 0.02 and b_act > 0.10:
        return "UNSOUND", f"inert (agent active {a_act:.0%} vs baseline {b_act:.0%})"
    if match >= thresh:
        which = f"matched {b_name}" if len(idea["baselines"]) > 1 else f"{match:.0%} match"
        return "CORRECT", which
    detail = f"{match:.0%} vs {b_name}; act {a_act:.0%} vs {b_act:.0%} ({ratio:.1f}x)"
    if match >= NEAR_FLOOR and ratio <= ACT_RATIO_MAX:
        return "NEAR", detail + " - REVIEW (valid variant or subtle bug)"
    return "BROKEN", detail


def grade_pairs(a, b, code, idea):
    """PAIRS analog of grade_position: load strategy_pair, compute its point-in-time spread positions over the
    two series, compare bar-by-bar to the single log-z baseline's positions."""
    warmup, thresh = idea["warmup"], idea["thresh"]
    try:
        strat = _load_strategy_pair(code)
        at = compute_pair_targets(a, b, strat).tolist()
    except Exception as e:
        return "UNSOUND", f"{type(e).__name__}: {str(e)[:40]}"
    if any((not math.isfinite(x)) or abs(x) > 1.0 + 1e-9 for x in at):
        return "UNSOUND", "non-finite / out-of-range positions"
    rng = list(range(warmup, len(a)))
    n = len(rng)
    a_act = sum(1 for t in rng if abs(at[t]) > 0.01) / n
    bt = idea["_btargets"]                                # a single baseline target list (not a name->list dict)
    match = sum(1 for t in rng if abs(at[t] - bt[t]) < 0.01) / n
    b_act = sum(1 for t in rng if abs(bt[t]) > 0.01) / n
    ratio = (a_act / b_act) if b_act > 0 else (float("inf") if a_act > 0.02 else 1.0)
    # a pairs baseline sits FLAT most bars, so overall bar-agreement is inflated by matching the flats. Also
    # require agreement on the baseline's ACTIVE bars (its trades) - an "always flat" agent scores 0 there.
    active = [t for t in rng if abs(bt[t]) > 0.01]
    active_match = (sum(1 for t in active if abs(at[t] - bt[t]) < 0.01) / len(active)) if active else 1.0
    if a_act < 0.02 and b_act > 0.02:                    # inert (lower floor: the pairs baseline is low-activity)
        return "UNSOUND", f"inert (agent active {a_act:.0%} vs baseline {b_act:.0%})"
    if match >= thresh and active_match >= 0.60:
        return "CORRECT", f"{match:.0%} match ({active_match:.0%} on trades)"
    detail = f"{match:.0%} overall; {active_match:.0%} on trades; act {a_act:.0%} vs {b_act:.0%} ({ratio:.1f}x)"
    if match >= NEAR_FLOOR and active_match >= 0.40 and ratio <= ACT_RATIO_MAX:
        return "NEAR", detail + " - REVIEW (valid variant or subtle bug)"
    return "BROKEN", detail


def _close(ticker, start="2021-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
    return df.dropna()


def main():
    model = (os.getenv("DEEPINFRA_MODEL", "?") if LLM_PROVIDER == "deepinfra"
             else os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7B (local)"))
    print(f"\n{'='*72}\nSUITE RULER   provider={LLM_PROVIDER}   model={model}   N={N_RUNS}\n{'='*72}")

    prices = _close("SPY")
    print(f"data: SPY  {prices.index[0].date()} -> {prices.index[-1].date()}  ({len(prices)} bars)")

    pair_data = {}                                       # PAIRS ideas fetch their OWN two aligned series
    for idea in IDEAS:
        if idea["paradigm"] == "pairs":
            a, b = _close(idea["ticker_a"]), _close(idea["ticker_b"])
            common = a.index.intersection(b.index)
            pair_data[idea["key"]] = (a.reindex(common), b.reindex(common))
            print(f"data: {idea['ticker_a']}-{idea['ticker_b']}  {common[0].date()} -> {common[-1].date()}"
                  f"  ({len(common)} bars)")
    print()

    for idea in IDEAS:                                   # precompute the baseline target series ONCE
        if idea["paradigm"] == "pairs":
            a, b = pair_data[idea["key"]]
            idea["_btargets"] = compute_pair_targets(a, b, idea["pairs_baseline"]).tolist()
        else:
            idea["_btargets"] = {name: compute_targets(prices, fn).tolist() for name, fn in idea["baselines"]}

    app = build_graph()
    summary = {}
    for idea in IDEAS:
        accepted = "log-z" if idea["paradigm"] == "pairs" else " | ".join(n for n, _ in idea["baselines"])
        print(f"--- {idea['key']}  (accept: {accepted}; >={idea['thresh']:.0%}) ---")
        verdicts = []
        for i in range(N_RUNS):
            try:
                r = app.invoke({"request": idea["prompt"], "prices": prices, "amount": 1000.0})
            except Exception as e:
                verdicts.append("UNSOUND")
                print(f"  run {i+1}/{N_RUNS}: UNSOUND    graph error: {str(e)[:40]}")
                continue
            if r.get("mode") != idea["paradigm"]:
                verdicts.append("MISROUTED")
                print(f"  run {i+1}/{N_RUNS}: MISROUTED  routed to {r.get('mode')}")
                continue
            code = r.get("strategy_code", "")
            if idea["paradigm"] == "pairs":
                a, b = pair_data[idea["key"]]
                v, detail = grade_pairs(a, b, code, idea)
            else:
                v, detail = grade_position(prices, code, idea)
            verdicts.append(v)
            print(f"  run {i+1}/{N_RUNS}: {v:9} {detail}")
        c = Counter(verdicts)
        summary[idea["key"]] = c
        print(f"  => {c['CORRECT']}/{N_RUNS} correct\n")

    print(f"{'='*72}\nSUMMARY  ({LLM_PROVIDER} / {model})")
    for key, c in summary.items():
        bits = "  ".join(f"{k}:{v}" for k, v in c.items())
        print(f"  {key:22} {c['CORRECT']}/{N_RUNS} correct   ({bits})")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
