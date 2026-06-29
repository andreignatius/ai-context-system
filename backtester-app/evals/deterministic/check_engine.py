"""Ground-truth the engine: buy-and-hold must EXACTLY track the normalized price."""
import numpy as np
import pandas as pd

from src.engine import run_backtest

# a deterministic toy price series (no network needed)
prices = pd.Series(
    [100, 101, 102, 99, 105, 110, 108],
    index=pd.date_range("2020-01-01", periods=7, freq="D"),
    dtype=float,
)

buy_and_hold = lambda h: 1.0
res = run_backtest(prices, buy_and_hold)

expected_total = prices.iloc[-1] / prices.iloc[0] - 1
expected_equity = prices / prices.iloc[0]

print("total_return :", round(res.total_return, 6), "| expected:", round(expected_total, 6))
print("equity == normalized price:",
      np.allclose(res.equity_curve.values, expected_equity.values))
print("sharpe:", round(res.sharpe, 3), "| max_dd:", round(res.max_drawdown, 4),
      "| n_trades:", res.n_trades, "| turnover:", res.turnover_total)

assert np.isclose(res.total_return, expected_total), "buy-and-hold total return mismatch!"
assert np.allclose(res.equity_curve.values, expected_equity.values), "equity != normalized price!"
print("\nPASS: engine reconciles buy-and-hold exactly.")


# --- ground-truth 2: SMA crossover must match an independent vectorized recompute ---
def sma_crossover(history, fast=2, slow=4):
    if len(history) < slow:
        return 0.0
    return 1.0 if history.iloc[-fast:].mean() > history.iloc[-slow:].mean() else 0.0

res2 = run_backtest(prices, sma_crossover)

# independent vectorized reference (NO point-in-time loop)
fast_ma = prices.rolling(2).mean()
slow_ma = prices.rolling(4).mean()
ref_pos = (fast_ma > slow_ma).astype(float).shift(1).fillna(0.0)
ref_ret = ref_pos * prices.pct_change().fillna(0.0)
ref_equity = (1.0 + ref_ret).cumprod()

print("\n[SMA crossover]")
print("total_return:", round(res2.total_return, 6),
      "| n_trades:", res2.n_trades, "| turnover:", res2.turnover_total)
print("engine == vectorized reference:",
      np.allclose(res2.equity_curve.values, ref_equity.values))
assert np.allclose(res2.equity_curve.values, ref_equity.values), "engine != vectorized SMA reference!"
print("PASS: point-in-time engine matches the vectorized SMA recompute.")


# --- ground-truth 3: forward-bias EVAL DEVICES ---

# (a) perfect-foresight CEILING: a CHEATING strategy that peeks at the next bar (closes over the
#     full series). An honest strategy can't do this - the engine only ever passes it `history`,
#     not the future. This is the theoretical MAX; any real strategy must sit BELOW it, and one
#     that MATCHES it is a look-ahead red flag the judge can use.
def perfect_foresight(history):
    t = len(history) - 1
    if t + 1 >= len(prices):
        return 0.0
    return 1.0 if prices.iloc[t + 1] > prices.iloc[t] else 0.0

ceiling = run_backtest(prices, perfect_foresight)
bh = run_backtest(prices, lambda h: 1.0)
print("\n[forward-bias eval devices]")
print("perfect-foresight ceiling total_return:", round(ceiling.total_return, 6))
print("buy-and-hold total_return:             ", round(bh.total_return, 6))
assert ceiling.total_return > bh.total_return, "ceiling should beat buy-and-hold!"
print("PASS: ceiling > buy-and-hold (captures up-moves, skips down-moves).")

# (b) determinism: same inputs -> identical result (the engine is a pure function)
r1 = run_backtest(prices, sma_crossover)
r2 = run_backtest(prices, sma_crossover)
assert np.array_equal(r1.equity_curve.values, r2.equity_curve.values), "engine not deterministic!"
print("PASS: engine is deterministic (same inputs -> identical equity).")
