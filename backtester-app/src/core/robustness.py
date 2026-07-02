"""Out-of-sample REGIME ROBUSTNESS (robustness-plan.md).

Run the point-in-time engine ONCE on the full series, then slice the per-bar returns into rolling OUT-OF-SAMPLE
windows. NO fitting (fixed rule-based strategies have no params to fit) - this measures whether a strategy
GENERALIZES across regimes, not a train/test cycle. Warm-up is handled by STARTING the windows at the
strategy's first active bar (so an inert warm-up never counts against the headline '% positive')."""
import numpy as np
import pandas as pd

from .engine import run_backtest


def _window_metrics(eq_norm, ppy):
    """Metrics from an equity path NORMALIZED to 1.0 at the window start (growth WITHIN the window)."""
    r = eq_norm.pct_change().fillna(0.0)
    total = float(eq_norm.iloc[-1] - 1.0)
    sharpe = float(r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 0 else 0.0
    max_dd = float((eq_norm / eq_norm.cummax() - 1.0).min())
    return total, sharpe, max_dd


def _roll(eqc, position, index, bench_prices, test_months, step_months, ppy):
    """Shared core: slice the strategy equity into rolling windows; bench_prices = the buy-and-hold benchmark
    (POSITION only; None for market-neutral pairs)."""
    active = position[position.abs() > 1e-9]                    # WARM-UP: start at the first ACTIVE bar
    w_start = active.index[0] if len(active) else index[0]
    end = index[-1]
    rows = []
    while w_start + pd.DateOffset(months=test_months) <= end:
        w_end = w_start + pd.DateOffset(months=test_months)
        idx = index[(index >= w_start) & (index < w_end)]
        if len(idx) >= 2:
            total, sharpe, max_dd = _window_metrics(eqc.loc[idx] / eqc.loc[idx[0]], ppy)
            row = {"test_start": idx[0].date(), "test_end": idx[-1].date(),
                   "total_return": total, "sharpe": sharpe, "max_drawdown": max_dd,
                   "n_trades": int((position.loc[idx].diff().abs() > 0).sum())}
            if bench_prices is not None:
                bh_t, bh_s, _ = _window_metrics(bench_prices.loc[idx] / bench_prices.loc[idx[0]], ppy)
                row["bh_return"], row["bh_sharpe"] = bh_t, bh_s
            rows.append(row)
        w_start = w_start + pd.DateOffset(months=step_months)

    table = pd.DataFrame(rows)
    n = len(table)
    worst = table.loc[table["total_return"].idxmin()] if n else None
    summary = {"windows": n,
               "pct_positive": float((table["total_return"] > 0).mean()) if n else 0.0,
               "avg_sharpe": float(table["sharpe"].mean()) if n else 0.0,
               "worst_return": float(worst["total_return"]) if n else 0.0,
               "worst_window": (str(worst["test_start"]), str(worst["test_end"])) if n else None,
               "full_sharpe": 0.0}
    if bench_prices is not None and n:
        summary["beat_bh"] = int((table["total_return"] > table["bh_return"]).sum())
    return table, summary


def rolling_robustness(prices, strategy, test_months=12, step_months=6, periods_per_year=252):
    """POSITION (single-asset): regime robustness + a per-window buy-and-hold benchmark."""
    bt = run_backtest(prices, strategy)
    targets = pd.Series([strategy(prices.iloc[: t + 1]) for t in range(len(prices))],
                        index=prices.index, dtype=float)
    position = targets.shift(1).fillna(0.0)
    table, summary = _roll(bt.equity_curve, position, prices.index, prices,
                           test_months, step_months, periods_per_year)
    summary["full_sharpe"] = float(bt.sharpe)
    return table, summary


def rolling_robustness_pairs(prices_a, prices_b, strategy_pair, test_months=12, step_months=6, periods_per_year=252):
    """PAIRS (multi-asset, dollar-neutral): regime robustness. NO buy-and-hold benchmark - a pair is
    market-neutral, so B&H is not a meaningful comparison."""
    from .pairs import run_pairs_backtest
    idx = prices_a.index.intersection(prices_b.index)
    a, b = prices_a.reindex(idx), prices_b.reindex(idx)
    res = run_pairs_backtest(a, b, strategy_pair)
    targets = pd.Series([strategy_pair(a.iloc[: t + 1], b.iloc[: t + 1]) for t in range(len(idx))],
                        index=idx, dtype=float)
    position = targets.shift(1).fillna(0.0)
    table, summary = _roll(res.equity_curve, position, idx, None, test_months, step_months, periods_per_year)
    summary["full_sharpe"] = float(res.sharpe)
    return table, summary
