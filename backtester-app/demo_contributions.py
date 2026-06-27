"""Answer the real question: buy-the-dip $1k vs monthly DCA $1k on SPY since the start of 2021.
Uses the verified contribution engine (src/contributions.py). Needs network (yfinance)."""
import yfinance as yf
import pandas as pd
from src.contributions import run_contributions, monthly_dates, signal_dates

AMOUNT = 1000.0
START = "2021-01-01"      # exact start date (the contribution question wants a real date, not "2y")

# --- data: SPY daily close since 2021 ---
df = yf.download("SPY", start=START, auto_adjust=True, progress=False)
prices = df["Close"]
if isinstance(prices, pd.DataFrame):
    prices = prices.iloc[:, 0]
prices = prices.dropna()
print(f"SPY  {prices.index[0].date()} -> {prices.index[-1].date()}  ({len(prices)} trading days)\n")


# --- the dip: 3-day drawdown = the last 2 day-over-day moves are both down ---
def dip(history):
    if len(history) < 3:
        return 0.0
    return 1.0 if (history.diff().dropna().iloc[-2:] < 0).all() else 0.0


# buy-the-dip: deposit $1k on EVERY qualifying bar (literal "each time there's a 3-day drawdown")
dip_dates = signal_dates(prices, dip)
_, dip_invested, dip_final = run_contributions(prices, dip_dates, AMOUNT)

# monthly DCA: deposit $1k on the first trading day of each month
mon_dates = monthly_dates(prices)
_, dca_invested, dca_final = run_contributions(prices, mon_dates, AMOUNT)

print("=== $1,000 per deposit, SPY since 2021 ===")
print(f"Buy-the-dip : {len(dip_dates):3d} deposits | invested ${dip_invested:>10,.0f} | "
      f"now ${dip_final:>11,.0f} | {dip_final / dip_invested:.2f}x")
print(f"Monthly DCA : {len(mon_dates):3d} deposits | invested ${dca_invested:>10,.0f} | "
      f"now ${dca_final:>11,.0f} | {dca_final / dca_invested:.2f}x")
print("\n(Buy-the-dip deposits on EVERY day the 3-day-drawdown holds - a literal reading; "
      "switch to once-per-dip-event if preferred.)")
