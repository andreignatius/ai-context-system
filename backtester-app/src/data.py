import yfinance as yf
import pandas as pd

# UI periods -> approx trading days. We download a 5y window once and slice by tail, so periods
# yfinance does not natively accept (e.g. "3y") still work.
_PERIOD_DAYS = {"6m": 126, "1y": 252, "2y": 504, "3y": 756, "5y": 1260}

def load_prices(ticker="SPY", period="2y", start=None) -> pd.Series:
    """Daily close prices as a pd.Series. If `start` is given (e.g. '2021-01-01') it OVERRIDES
    `period` and downloads from that exact date; otherwise use `period` (sliced from a 5y window)."""
    if start:
        df = yf.download(ticker, start=str(start), auto_adjust=True, progress=False)
    else:
        df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):   # newer yfinance returns multiindex cols
        close = close.iloc[:, 0]
    close.name = ticker
    close = close.dropna()
    return close if start else close.tail(_PERIOD_DAYS.get(period, 504))
