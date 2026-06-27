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
