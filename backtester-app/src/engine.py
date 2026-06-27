"""The backtest engine: a pure, deterministic verifier.

Given prices + a strategy callable, step bar-by-bar (point-in-time), lag the
target position by one bar, and compute the equity curve + metrics. Look-ahead
is structurally impossible: the strategy only ever sees prices up to the current
bar, and its target is acted on the NEXT bar.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    total_return: float
    ann_return: float
    sharpe: float
    max_drawdown: float
    turnover_total: float
    n_trades: int


def run_backtest(prices, strategy, fee=0.0, periods_per_year=252) -> BacktestResult:
    # 1. point-in-time: strategy sees ONLY prices up to and including bar t
    targets = [strategy(prices.iloc[: t + 1]) for t in range(len(prices))]
    targets = pd.Series(targets, index=prices.index, dtype=float)

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
