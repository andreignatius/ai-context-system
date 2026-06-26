import yfinance as yf
import pandas as pd

# UI periods -> approx trading days. We download a 5y window once and slice by tail, so periods
# yfinance does not natively accept (e.g. "3y") still work.
_PERIOD_DAYS = {"6m": 126, "1y": 252, "2y": 504, "3y": 756, "5y": 1260}

def load_prices(ticker="SPY", period="2y") -> pd.Series:
    """Daily close prices as a pd.Series (datetime index). `period` in {6m,1y,2y,3y,5y}."""
    df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):   # newer yfinance returns multiindex cols
        close = close.iloc[:, 0]
    close.name = ticker
    return close.dropna().tail(_PERIOD_DAYS.get(period, 504))
