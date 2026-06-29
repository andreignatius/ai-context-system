from src.runner import run_strategy
from src.data import load_prices

prices = load_prices("SPY", "2y")

good = """
def strategy(history, fast=20, slow=50):
    if len(history) < slow:
        return 0.0
    return 1.0 if history.iloc[-fast:].mean() > history.iloc[-slow:].mean() else 0.0
"""
bad_range = "def strategy(history):\n    return 5.0    # out of [-1, 1]\n"
bad_load  = "def not_strategy(history):\n    return 1.0   # wrong name\n"

for name, code in [("good SMA", good), ("bad range", bad_range), ("bad load", bad_load)]:
    r = run_strategy(code, prices)
    print(f"{name:10s} passed={r['passed']}  failures='{r['failures']}'  "
          f"metrics={ {k: round(v,3) for k,v in r['metrics'].items()} }")
