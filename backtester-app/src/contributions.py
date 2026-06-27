"""Contribution / cash-flow engine - a DIFFERENT paradigm from the position engine.

The position engine (src/engine.py) answers "what FRACTION to hold" (growth of $1, time-weighted).
This answers "how much MONEY would I have": deposit $X on a schedule, track units bought -> total
invested -> final dollar value. Lets you compare buy-the-dip ($X on each signal) vs DCA ($X each
month) in actual dollars. See Lesson 028 (the position-vs-contribution boundary)."""
import numpy as np
import pandas as pd


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


def signal_dates(prices, strategy):
    """Bars where the strategy fires (>0) - deposit on each such bar (e.g. each dip). Point-in-time:
    the signal at bar t sees only prices up to t, and we deposit at t's close (no look-ahead)."""
    fires = np.array([strategy(prices.iloc[: t + 1]) > 0 for t in range(len(prices))])
    return prices.index[fires]
