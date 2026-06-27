"""Ground-truth baseline for the PAIRS engine (next-steps, pairs Phase A).
XLF vs XLI, LOG-SPREAD z-score mean-reversion, point-in-time. The golden numbers the pairs engine must match.

v1 definition (documented = the spec):
- spread  = log(XLF) - log(XLI)
- z       = (spread - rolling-mean) / rolling-std over W=60 bars (point-in-time: rolling uses only past+present)
- signal  = +1 (long the spread: long XLF / short XLI) when z < -2
            -1 (short the spread: short XLF / long XLI) when z > +2
            0  otherwise   [STATELESS bands - v2 = stateful enter +/-2 / exit at 0, needs engine position-state]
- P&L     = pos.shift(1) * (ret_XLF - ret_XLI)   [dollar-neutral pair];  equity = cumprod(1 + pnl)
Run:  python -m baselines.pairs
"""
import yfinance as yf
import numpy as np
import pandas as pd

A, B = "XLF", "XLI"
START = "2019-01-01"
W = 60
ENTRY = 2.0


def _close(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
    return df.dropna()


def main():
    a, b = _close(A, START), _close(B, START)
    idx = a.index.intersection(b.index)                 # common window (alignment)
    a, b = a.reindex(idx), b.reindex(idx)

    spread = np.log(a) - np.log(b)
    z = (spread - spread.rolling(W).mean()) / spread.rolling(W).std()
    pos = pd.Series(0.0, index=idx)
    pos[z < -ENTRY] = 1.0                                # long the spread (long XLF / short XLI)
    pos[z > ENTRY] = -1.0                                # short the spread
    ret_a, ret_b = a.pct_change().fillna(0.0), b.pct_change().fillna(0.0)
    pnl = pos.shift(1).fillna(0.0) * (ret_a - ret_b)    # 1-bar lag; dollar-neutral spread return
    equity = (1.0 + pnl).cumprod()

    total = equity.iloc[-1] - 1.0
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0.0
    maxdd = (equity / equity.cummax() - 1.0).min()
    trades = int((pos.diff().abs() > 0).sum())
    print(f"PAIRS BASELINE  {A}-{B}  (log-spread, W={W}, entry +/-{ENTRY})  |  "
          f"{idx[0].date()} -> {idx[-1].date()}  ({len(idx)} bars)")
    print(f"  total_return={total:+.1%}   Sharpe={sharpe:.2f}   maxDD={maxdd:.1%}   "
          f"trades={trades}   time-in-market={ (pos != 0).mean():.0%}")


if __name__ == "__main__":
    main()
