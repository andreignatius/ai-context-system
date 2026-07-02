"""Display-only helpers: benchmarks + extra performance metrics, in Andre's coursework conventions.
Kept OUT of the engine on purpose - the engine (src/engine.py) stays the minimal trusted verifier;
these are presentation-layer extras for the UI (rule of three: graft viz/metrics, not the whole lib)."""
import pandas as pd


def buy_and_hold(prices) -> pd.Series:
    """Growth of $1, lump sum (equity starts at 1.0) - the clean same-basis benchmark."""
    return prices / prices.iloc[0]


def longest_drawdown_days(equity) -> int:
    """Longest underwater stretch, in bars. Uses the (dd==0).cumsum() period idiom: each new high
    opens a new group, so (group length - 1) is that episode's days underwater (your convention)."""
    dd = equity / equity.cummax() - 1
    period = (dd == 0).cumsum()
    return int(dd.groupby(period).count().max()) - 1


def annual_returns(equity) -> pd.Series:
    """Year-over-year returns from an equity curve (indexed by year-end)."""
    yearly = equity.resample("YE").last()
    return (yearly / yearly.shift(1) - 1).dropna()


def dca(prices, monthly: float = 100.0):
    """Dollar-cost-average `monthly`, buying on the FIRST trading day of each calendar month
    (indexed at real bars, so no synthetic month-end label issues). Returns (value_curve,
    final_value, total_invested). NOTE: a DIFFERENT cash-flow basis from a lump sum - report as an
    end metric, not a same-axis line."""
    first_of_month = ~prices.index.to_period("M").duplicated()      # first bar of each month
    units_bought = (monthly / prices.where(first_of_month)).fillna(0.0)
    units = units_bought.cumsum()                                   # cumulative units owned
    value = units * prices                                          # portfolio value over time
    total_invested = monthly * int(first_of_month.sum())
    return value, float(value.iloc[-1]), float(total_invested)
