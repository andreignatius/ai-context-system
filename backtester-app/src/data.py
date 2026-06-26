import yfinance as yf
import pandas as pd

def load_prices(ticker="SPY", period="2y") -> pd.Series:
    """Daily close prices as a pd.Series (datetime index)."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):   # newer yfinance returns multiindex cols
        close = close.iloc[:, 0]
    close.name = ticker
    return close.dropna()
