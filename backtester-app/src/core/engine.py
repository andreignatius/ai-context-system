"""The backtest engine: a pure, deterministic verifier.

Given prices + a strategy callable, step bar-by-bar (point-in-time), lag the
target position by one bar, and compute the equity curve + metrics. Look-ahead
is structurally impossible: the strategy only ever sees prices up to the current
bar, and its target is acted on the NEXT bar.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Default transaction cost: ~5 bps per unit turnover (~10 bps per round-trip) - commissions + slippage for
# liquid ETFs. Applied as fee * |delta position|, so a frictionless gross result no longer flatters churn
# (Lesson 035: a 412-trade strategy can't masquerade as a 7-trade one). Override per-call with fee=...
DEFAULT_FEE = 0.0005


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    total_return: float
    ann_return: float
    sharpe: float
    max_drawdown: float
    turnover_total: float
    n_trades: int


def compute_targets(prices, strategy) -> pd.Series:
    """Point-in-time target series: the ONE expanding-window loop the whole app shares. `strategy`
    sees ONLY prices up to AND INCLUDING bar t (no look-ahead). Callers that need the targets for
    both the engine AND a soundness/robustness check compute them once here and thread them through
    (via run_backtest's `targets=`), instead of re-running this O(N^2) loop. See docs/review.md #2."""
    return pd.Series([strategy(prices.iloc[: t + 1]) for t in range(len(prices))],
                     index=prices.index, dtype=float)


def run_backtest(prices, strategy, fee=DEFAULT_FEE, periods_per_year=252, targets=None) -> BacktestResult:
    # 1. point-in-time: strategy sees ONLY prices up to and including bar t (reuse precomputed if given)
    targets = compute_targets(prices, strategy) if targets is None else targets

    # 2. the LAG: a target decided at t is held from t+1 (no same-bar fill)
    position = targets.shift(1).fillna(0.0)

    # 3. returns, turnover, costs
    asset_ret = prices.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strat_ret = position * asset_ret - fee * turnover

    # 4. equity curve (starts at 1.0)
    equity = (1.0 + strat_ret).cumprod()

    # 5. metrics
    total_return = float(equity.iloc[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (periods_per_year / len(equity)) - 1.0)
    std = strat_ret.std()
    sharpe = float(strat_ret.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    max_drawdown = float((equity / equity.cummax() - 1.0).min())

    return BacktestResult(
        equity_curve=equity,
        total_return=total_return,
        ann_return=ann_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        turnover_total=float(turnover.sum()),
        n_trades=int((turnover > 0).sum()),
    )
