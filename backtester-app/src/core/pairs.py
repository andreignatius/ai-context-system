"""Pairs / spread engine - a THIRD paradigm (multi-asset POSITION). Trades the SPREAD of TWO assets.

Generalizes the single-asset engine: strategy_pair(history_a, history_b) -> spread position in [-1, 1]
  +1 = long the spread  (long A / short B)
  -1 = short the spread (short A / long B)
   0 = flat
Point-in-time + 1-bar lag, so look-ahead is structurally impossible - the strategy only ever sees prices up
to the CURRENT bar for BOTH legs (incl. any rolling z-score or hedge-ratio regression it computes inside).
P&L is dollar-neutral: pos.shift(1) * (ret_A - ret_B). 'correct' != 'profitable' - the engine checks soundness.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .engine import DEFAULT_FEE        # one source of truth for the transaction-cost default


@dataclass
class PairsResult:
    equity_curve: pd.Series
    total_return: float
    ann_return: float
    sharpe: float
    max_drawdown: float
    n_trades: int


def compute_pair_targets(a, b, strategy_pair) -> pd.Series:
    """Point-in-time spread-target series (the pairs twin of engine.compute_targets). `a` and `b` must
    ALREADY be aligned to a common index; the strategy sees only prices up to AND INCLUDING bar t for
    both legs. Shared so run_pairs_backtest + the soundness/robustness checks reuse one loop."""
    return pd.Series([strategy_pair(a.iloc[: t + 1], b.iloc[: t + 1]) for t in range(len(a))],
                     index=a.index, dtype=float)


def run_pairs_backtest(prices_a, prices_b, strategy_pair, fee=DEFAULT_FEE, periods_per_year=252,
                       targets=None) -> PairsResult:
    idx = prices_a.index.intersection(prices_b.index)        # common window (alignment)
    a, b = prices_a.reindex(idx), prices_b.reindex(idx)

    # 1. point-in-time: the strategy sees ONLY prices up to and including bar t, for BOTH legs
    targets = compute_pair_targets(a, b, strategy_pair) if targets is None else targets
    # 2. the LAG: a target decided at t is held from t+1 (no same-bar fill)
    position = targets.shift(1).fillna(0.0)
    # 3. dollar-neutral spread return: long the spread = long A / short B -> earns ret_A - ret_B
    ret_a = a.pct_change().fillna(0.0)
    ret_b = b.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    pnl = position * (ret_a - ret_b) - fee * turnover
    equity = (1.0 + pnl).cumprod()

    total = float(equity.iloc[-1] - 1.0)
    cagr = float((1.0 + total) ** (periods_per_year / len(equity)) - 1.0) if len(equity) else 0.0
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year)) if pnl.std() > 0 else 0.0
    max_dd = float((equity / equity.cummax() - 1.0).min())
    n_trades = int((position.diff().abs() > 0).sum())
    return PairsResult(equity, total, cagr, sharpe, max_dd, n_trades)
