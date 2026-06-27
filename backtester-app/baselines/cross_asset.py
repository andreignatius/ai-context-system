"""Ground-truth baseline for the CROSS-ASSET contribution comparison (per-leg-ticker-plan, Phase A).
GOOG monthly $1k vs SPY monthly $1k since 2021, aligned to the COMMON window (fairness). These are the
golden numbers the per-leg-ticker feature must reproduce. Needs network (yfinance).

Run:  python -m baselines.cross_asset   (or python baselines/cross_asset.py)
"""
import yfinance as yf
import pandas as pd

START = "2021-01-01"
AMOUNT = 1000.0
TICKERS = ["GOOG", "SPY"]


def _close(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
    return df.dropna()


def monthly_dates(prices):
    return prices.index[~prices.index.to_period("M").duplicated()]


def run_contributions(prices, deposit_dates, amount):
    deposit_set = set(pd.DatetimeIndex(deposit_dates))
    units, values, n = 0.0, [], 0
    for date, px in prices.items():
        if date in deposit_set:
            units += amount / px
            n += 1
        values.append(units * px)
    return pd.Series(values, index=prices.index), amount * n, float(values[-1])


def main():
    raw = {t: _close(t, START) for t in TICKERS}
    eff_start = max(px.index[0] for px in raw.values())     # common window = latest first-date (fairness)
    aligned = {t: px[px.index >= eff_start] for t, px in raw.items()}
    end = min(px.index[-1] for px in aligned.values())
    print(f"BASELINE: {TICKERS[0]} vs {TICKERS[1]} monthly ${AMOUNT:,.0f}  |  common window "
          f"{eff_start.date()} -> {end.date()}\n")
    for t in TICKERS:
        px = aligned[t]
        mdates = monthly_dates(px)
        curve, inv, final = run_contributions(px, mdates, AMOUNT)
        print(f"  {t:5}: {len(mdates):3} deposits  invested ${inv:>9,.0f} -> ${final:>11,.0f}  "
              f"({final / inv:.3f}x)")


if __name__ == "__main__":
    main()
