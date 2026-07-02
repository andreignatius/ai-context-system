"""Check the contribution_run node in isolation (no LLM; feeds a strategy_code + toy prices)."""
import numpy as np
import pandas as pd
from src.graph import contribution_run

dip_code = '''def strategy(history):
    if len(history) < 3:
        return 0.0
    return 1.0 if (history.diff().dropna().iloc[-2:] < 0).all() else 0.0'''

# a wavy 120-day series (Jan-Apr) so dips fire and 4 months span
idx = pd.date_range("2021-01-01", periods=120, freq="D")
prices = pd.Series(100 + np.cumsum(np.sin(np.arange(120) / 3.0)), index=idx)

# GUARD REQUIRED: contribution_run's signal leg SPAWNS a sandbox child that re-imports THIS module. Without
# the `if __name__` guard, that re-import re-runs the code below -> a recursive spawn during bootstrap ->
# RuntimeError -> the child dies -> the parent times out. Any module that reaches the sandbox at module
# scope must be __main__-guarded (see docs/sandbox-plan.md, "Caller invariant").
if __name__ == "__main__":
    out = contribution_run({"strategy_code": dip_code, "prices": prices, "amount": 1000.0, "mode": "contribution"})
    print("status:", out["status"])
    cr = out["contribution_result"]
    print("signal:", cr["signal"])
    print("dca   :", cr["dca"])

    assert out["status"] == "ok"
    assert cr["signal"]["invested"] == 1000.0 * cr["signal"]["n"]      # $1k per deposit
    assert cr["dca"]["n"] == 4                                          # Jan/Feb/Mar/Apr firsts
    assert cr["dca"]["invested"] == 4000.0
    print("\nPASS: contribution_run produces the dollar comparison (signal deposits vs monthly DCA).")
