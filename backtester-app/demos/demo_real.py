from src.core.engine import run_backtest
from src.core.data import load_prices

prices = load_prices("SPY", "2y")
print(f"loaded {len(prices)} bars of SPY: {prices.index[0].date()} -> {prices.index[-1].date()}\n")

def sma_crossover(history, fast=20, slow=50):
    if len(history) < slow:
        return 0.0
    return 1.0 if history.iloc[-fast:].mean() > history.iloc[-slow:].mean() else 0.0

for name, strat in [("buy-and-hold", lambda h: 1.0), ("SMA 20/50", sma_crossover)]:
    r = run_backtest(prices, strat)
    print(f"{name:14s} total={r.total_return:+.2%} ann={r.ann_return:+.2%} "
          f"sharpe={r.sharpe:.2f} maxDD={r.max_drawdown:.2%} trades={r.n_trades}")
