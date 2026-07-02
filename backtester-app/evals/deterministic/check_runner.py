from src.runner import run_strategy
from src.core.data import load_prices

good = """
def strategy(history, fast=20, slow=50):
    if len(history) < slow:
        return 0.0
    return 1.0 if history.iloc[-fast:].mean() > history.iloc[-slow:].mean() else 0.0
"""
bad_range = "def strategy(history):\n    return 5.0    # out of [-1, 1]\n"
bad_load  = "def not_strategy(history):\n    return 1.0   # wrong name\n"

# GUARD REQUIRED: run_strategy SPAWNS a sandbox child that re-imports THIS module. Without the `if __name__`
# guard, the re-import re-runs the loop below -> a recursive spawn during bootstrap -> RuntimeError -> the
# child dies -> every case falsely "times out". Any module that reaches the sandbox at module scope must be
# __main__-guarded (see docs/sandbox-plan.md, "Caller invariant"). load_prices (network) is also gated here.
if __name__ == "__main__":
    prices = load_prices("SPY", "2y")
    results = {}
    for name, code in [("good SMA", good), ("bad range", bad_range), ("bad load", bad_load)]:
        r = run_strategy(code, prices)
        results[name] = r
        print(f"{name:10s} passed={r['passed']}  failures='{r['failures']}'  "
              f"metrics={ {k: round(v,3) for k,v in r['metrics'].items()} }")

    # ASSERTS so this is a REAL check, not a print-only eyeball (a no-assert check silently false-greens -
    # that is exactly how the spawn regression hid here).
    assert results["good SMA"]["passed"] is True, f"good SMA should pass: {results['good SMA']['failures']}"
    assert results["bad range"]["passed"] is False and "out of range" in results["bad range"]["failures"], \
        "bad range should fail with an out-of-range message"
    assert results["bad load"]["passed"] is False and "load error" in results["bad load"]["failures"], \
        "bad load should fail with a load error"
    print("\nPASS: runner soundness verdicts correct (good passes; bad range / bad load fail as expected).")
