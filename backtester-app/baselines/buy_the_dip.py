"""BASELINE (human-verified ground truth) for the prompt:
  "buy the dip - if there's a drawdown 3 consecutive trading days I put in $1k, since the first
   trading day of 2021, how much at the end - vs DCA $1k every first trading day of the month."

Definitional choices (documented, because the user's phrasing is what matters):
  - DIP = the last 3 day-over-day changes are ALL negative (3 consecutive down days), POINT-IN-TIME
    (decided from data up to the current bar - NOT by scanning all of history).
  - DEPOSIT on EACH bar the dip condition holds (literal "each time there's a 3-day drawdown").
  - Compare buy-the-dip vs monthly-DCA in DOLLARS; the fair number is the per-dollar MULTIPLE.
Run: python -m baselines.buy_the_dip   (or python baselines/buy_the_dip.py from backtester-app/)"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

AMOUNT = 1000.0
START = "2021-01-01"


def is_dip(history) -> bool:
    """Point-in-time: the last 3 daily changes are all negative."""
    if len(history) < 4:
        return False
    return bool((history.diff().iloc[-3:] < 0).all())


def run_contributions(prices, deposit_dates, amount):
    deposit_set = set(pd.DatetimeIndex(deposit_dates))
    units, values, n = 0.0, [], 0
    for date, px in prices.items():
        if date in deposit_set:
            units += amount / px
            n += 1
        values.append(units * px)
    return pd.Series(values, index=prices.index), amount * n, float(values[-1])


def baseline(ticker="SPY", start=START, amount=AMOUNT):
    prices = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices = prices.dropna()

    dip_dates = prices.index[[is_dip(prices.iloc[: t + 1]) for t in range(len(prices))]]
    monthly_dates = prices.index[~prices.index.to_period("M").duplicated()]

    dip_curve, dip_inv, dip_final = run_contributions(prices, dip_dates, amount)
    dca_curve, dca_inv, dca_final = run_contributions(prices, monthly_dates, amount)
    return {
        "range": (prices.index[0].date(), prices.index[-1].date()),
        "dip": {"n": len(dip_dates), "invested": dip_inv, "final": dip_final, "mult": dip_final / dip_inv},
        "dca": {"n": len(monthly_dates), "invested": dca_inv, "final": dca_final, "mult": dca_final / dca_inv},
        "curves": (dip_curve, dca_curve),
    }


if __name__ == "__main__":
    r = baseline()
    print(f"SPY {r['range'][0]} -> {r['range'][1]}  (${AMOUNT:,.0f} per deposit)\n")
    for name, k in [("Buy-the-dip", "dip"), ("Monthly DCA", "dca")]:
        m = r[k]
        print(f"{name:13}: {m['n']:3d} deposits  ${m['invested']:>9,.0f} -> ${m['final']:>11,.0f}  ({m['mult']:.2f}x)")
    dip_c, dca_c = r["curves"]
    plt.figure(figsize=(14, 7))
    plt.plot(dip_c.index, dip_c, label="Buy-the-dip")
    plt.plot(dca_c.index, dca_c, label="Monthly DCA")
    plt.title("SPY since 2021 - portfolio value, $1,000 per deposit (BASELINE)")
    plt.ylabel("Portfolio value ($)"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.show()
