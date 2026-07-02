"""Ground-truth the contribution engine by hand on toy series (no network, no LLM)."""
import pandas as pd
from src.core.contributions import run_contributions, monthly_dates, signal_dates


def dip(history):
    """3-day drawdown = the last 2 day-over-day moves are both down."""
    if len(history) < 3:
        return 0.0
    return 1.0 if (history.diff().dropna().iloc[-2:] < 0).all() else 0.0


# 1. flat prices, 2 deposits -> invested == final (no growth)
idx = pd.date_range("2021-01-04", periods=4, freq="D")
p = pd.Series([100, 100, 100, 100], index=idx, dtype=float)
vc, inv, fin = run_contributions(p, [idx[0], idx[2]], 1000.0)
print(f"flat       : invested={inv:.0f} final={fin:.0f}  (expect 2000 / 2000)")
assert inv == 2000 and fin == 2000

# 2. price doubles between the two deposits (10 units -> $2000, 5 units -> $1000)
p2 = pd.Series([100, 100, 200, 200], index=idx, dtype=float)
vc2, inv2, fin2 = run_contributions(p2, [idx[0], idx[2]], 1000.0)
print(f"doubling   : invested={inv2:.0f} final={fin2:.0f}  (expect 2000 / 3000)")
assert inv2 == 2000 and fin2 == 3000

# 3. monthly first-trading-day schedule over 3 calendar months
idx3 = pd.date_range("2021-01-01", periods=90, freq="D")
p3 = pd.Series(range(100, 190), index=idx3, dtype=float)
md = monthly_dates(p3)
print(f"monthly    : {len(md)} deposit dates  (expect 3 = Jan/Feb/Mar firsts)")
assert len(md) == 3

# 4. dip signal: never fires on a rising series; fires twice on a 3-day decline
p4 = pd.Series([100, 99, 98, 97, 100], index=pd.date_range("2021-01-01", periods=5, freq="D"), dtype=float)
sd_rising = signal_dates(p3, dip)
sd_dip = signal_dates(p4, dip)
print(f"signal     : rising->{len(sd_rising)} dips, declining->{len(sd_dip)} dips  (expect 0 / 2)")
assert len(sd_rising) == 0 and len(sd_dip) == 2

print("\nPASS: contribution engine reconciles (deposits, monthly schedule, dip signal).")
