"""Check the request classifier + param extraction (needs the LLM, e.g. local Ollama)."""
from src.agents import classify

# (request, expected mode, expected start present?, expected amount)
cases = [
    ("go long when the 20-day return is positive, else flat", "position", None, None),
    ("backtest an SMA 20/50 crossover strategy", "position", None, None),
    ("if there is a 3-day drawdown I put in $1000 since 2021, how much vs monthly DCA",
     "contribution", "2021-01-01", 1000.0),
    ("how much money would I have if I DCA'd $500 a month into QQQ", "contribution", None, 500.0),
    ("invest $2k every time RSI drops below 30 since 2020, total value?", "contribution", "2020-01-01", 2000.0),
    # IN-SCOPE, must NOT regress to out_of_scope (fancy-sounding but price-only):
    ("pairs trade KO vs PEP with a 2 sigma z-score band", "position", None, None),
    ("mean reversion: long AAPL when RSI(14) < 30, exit at 70", "position", None, None),
    # HELP (meta) must stay help, not out_of_scope:
    ("hi what can you do?", "help", None, None),
    # OUT_OF_SCOPE (the feasibility gate, gate 0):
    ("design a strategy on trade-finance spreads via CDS proxies on commodity banks", "out_of_scope", None, None),
    ("backtest a strategy using P/E ratios and short interest", "out_of_scope", None, None),
    ("long SPY only when VIX is below 30", "out_of_scope", None, None),
    ("buy when volume spikes to 2x the 20-day average", "out_of_scope", None, None),
]

correct = 0
for request, exp_mode, exp_start, exp_amt in cases:
    out = classify({"request": request})
    mode_ok = out["mode"] == exp_mode
    start_ok = (out.get("start_date") is not None) == (exp_start is not None)
    amt_ok = out.get("amount") == exp_amt
    ok = mode_ok and start_ok and amt_ok
    correct += ok
    print(f"[{'ok  ' if ok else 'MISS'}] mode={out['mode']:12} start={str(out.get('start_date')):11} "
          f"amount={out.get('amount')!s:7} | {request[:46]}")

print(f"\n{correct}/{len(cases)} fully correct (mode + start presence + amount)")
