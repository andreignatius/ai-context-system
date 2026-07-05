import yfinance as yf
import pandas as pd

# UI periods -> approx trading days. We download a 5y window once and slice by tail, so periods
# yfinance does not natively accept (e.g. "3y") still work.
_PERIOD_DAYS = {"6m": 126, "1y": 252, "2y": 504, "3y": 756, "5y": 1260}
MIN_BARS = 2   # fewer usable bars than this = effectively no data (a bad/typo ticker) -> fail fast

def load_prices(ticker="SPY", period="2y", start=None) -> pd.Series:
    """Daily close prices as a pd.Series. If `start` is given (e.g. '2021-01-01') it OVERRIDES
    `period` and downloads from that exact date; otherwise use `period` (sliced from a 5y window)."""
    if start:
        df = yf.download(ticker, start=str(start), auto_adjust=True, progress=False)
    else:
        df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
    close = df["Close"] if "Close" in df else pd.Series(dtype=float)
    if isinstance(close, pd.DataFrame):   # newer yfinance returns multiindex cols
        close = close.iloc[:, 0]
    close.name = ticker
    close = close.dropna()
    out = close if start else close.tail(_PERIOD_DAYS.get(period, 504))
    # FAIL FAST: a bad/typo ticker returns EMPTY data (not an error) -> guard HERE so it surfaces as a
    # clear message, not a cryptic IndexError deep in the engine (equity.iloc[-1] on an empty curve).
    if len(out) < MIN_BARS:
        raise ValueError(f"No usable price data for '{ticker}'. Check the Yahoo Finance symbol - "
                         f"indices/futures use suffixes (e.g. the US Dollar Index is 'DX-Y.NYB', not 'DXY').")
    return out
