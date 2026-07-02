"""Contribution / cash-flow engine - a DIFFERENT paradigm from the position engine.

The position engine (src/engine.py) answers "what FRACTION to hold" (growth of $1, time-weighted).
This answers "how much MONEY would I have": deposit $X on a schedule, track units bought -> total
invested -> final dollar value. Lets you compare buy-the-dip ($X on each signal) vs DCA ($X each
month) in actual dollars. See Lesson 028 (the position-vs-contribution boundary)."""
import pandas as pd

from .engine import compute_targets


def run_contributions(prices, deposit_dates, amount: float = 1000.0):
    """Deposit `amount` on each date in `deposit_dates`, buying amount/price units that day.
    Returns (value_curve, total_invested, final_value)."""
    deposit_set = set(pd.DatetimeIndex(deposit_dates))
    units, values, n = 0.0, [], 0
    for date, px in prices.items():
        if date in deposit_set:
            units += amount / px            # deploy $amount at that day's price
            n += 1
        values.append(units * px)           # mark-to-market value of all units held
    value_curve = pd.Series(values, index=prices.index)
    return value_curve, amount * n, float(value_curve.iloc[-1])


def monthly_dates(prices):
    """First trading day of each calendar month - the schedule for $X/month DCA."""
    first_of_month = ~prices.index.to_period("M").duplicated()
    return prices.index[first_of_month]


def weekly_dates(prices):
    """First trading day of each calendar week - the schedule for $X/week DCA."""
    first_of_week = ~prices.index.to_period("W").duplicated()
    return prices.index[first_of_week]


def signal_dates(prices, strategy):
    """Bars where the strategy fires (>0) - deposit on each such bar (e.g. each dip). Point-in-time:
    the signal at bar t sees only prices up to t, and we deposit at t's close (no look-ahead)."""
    fires = compute_targets(prices, strategy) > 0    # point-in-time: the signal at t sees only prices up to t
    return prices.index[fires.values]


def schedule_dates(prices, cadence, strategy=None):
    """Resolve a cadence label to deposit dates. 'weekly'/'monthly' are calendar schedules; 'signal'
    uses the coder's strategy(history) (deposit on each bar it fires). One dispatcher so a contribution
    leg can be ANY of the three - lets us compare e.g. weekly $250 vs monthly $1000 in one run."""
    if cadence == "weekly":
        return weekly_dates(prices)
    if cadence == "monthly":
        return monthly_dates(prices)
    if cadence == "signal":
        if strategy is None:
            raise ValueError("cadence 'signal' needs a strategy(history) to fire on")
        return signal_dates(prices, strategy)
    raise ValueError(f"unknown cadence: {cadence!r} (expected signal | weekly | monthly)")
