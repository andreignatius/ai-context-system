"""The strategy runner = the backtester's SANDBOX (the verifier).

Takes LLM-written strategy code (a string defining `strategy(history) -> float`),
execs it, runs it through the engine, and checks SOUNDNESS: did it run, are
positions valid, is it deterministic. This is the role pytest played in the
code-builder - here the engine IS the test.
"""
import numpy as np
import pandas as pd
import traceback
import ast

from .engine import run_backtest
from .pairs import run_pairs_backtest

BANNED_CALLS = {"eval", "exec", "open", "__import__", "compile", "getattr", "globals"}
ALLOWED_IMPORTS = {"pandas", "numpy", "math"}


def _validate(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = (getattr(node, "module", None) or node.names[0].name).split(".")[0]
            if mod not in ALLOWED_IMPORTS:
                raise ValueError(f"import '{mod}' not allowed")
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in BANNED_CALLS:
            raise ValueError(f"call '{node.func.id}' not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder access not allowed")   # blocks __globals__, __subclasses__ escapes


def _load_strategy(code: str):
    _validate(code) # SECURITY: reject dangerous code BEFORE exec
    namespace = {"pd": pd, "np": np}
    exec(code, namespace)
    if "strategy" not in namespace or not callable(namespace["strategy"]):
        raise ValueError("code must define a callable `strategy(history)`")
    return namespace["strategy"]


def run_strategy(strategy_code: str, prices) -> dict:
    # 1. load
    try:
        strategy = _load_strategy(strategy_code)
    except Exception as e:
        return {"passed": False, "failures": f"load error: {e}", "metrics": {}}

    # 2. run through the engine
    try:
        result = run_backtest(prices, strategy)
    except Exception as e:
        # return {"passed": False, "failures": f"runtime error: {e}", "metrics": {}}
        return {"passed": False,
                "failures": f"runtime error: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                "metrics": {}}
    # 3. soundness checks (NOT profitability)
    failures = []
    targets = pd.Series([strategy(prices.iloc[: t + 1]) for t in range(len(prices))], dtype=float)
    if not np.isfinite(targets).all():
        failures.append("strategy produced non-finite positions")
    if (targets.abs() > 1.0 + 1e-9).any():
        failures.append("positions out of range [-1, 1]")
    result2 = run_backtest(prices, strategy)
    if not np.array_equal(result.equity_curve.values, result2.equity_curve.values):
        failures.append("strategy is not deterministic")

    metrics = {"total_return": result.total_return, "ann_return": result.ann_return,
               "sharpe": result.sharpe, "max_drawdown": result.max_drawdown,
               "n_trades": result.n_trades}
    if result.n_trades == 0:
        failures.append("strategy never takes a position (inert) - warm-up may exceed the data length")

    return {"passed": len(failures) == 0, "failures": "; ".join(failures), "metrics": metrics}

def _load_strategy_pair(code: str):
    _validate(code)  # SECURITY: same AST sandbox as single-asset
    namespace = {"pd": pd, "np": np}
    exec(code, namespace)
    if "strategy_pair" not in namespace or not callable(namespace["strategy_pair"]):
        raise ValueError("code must define a callable `strategy_pair(history_a, history_b)`")
    return namespace["strategy_pair"]


def run_pair_strategy(code: str, prices_a, prices_b) -> dict:
    try:
        strat = _load_strategy_pair(code)
    except Exception as e:
        return {"passed": False, "failures": f"load error: {e}", "metrics": {}}
    try:
        result = run_pairs_backtest(prices_a, prices_b, strat)
    except Exception as e:
        return {"passed": False, "failures": f"runtime error: {type(e).__name__}: {e}", "metrics": {}}
    idx = prices_a.index.intersection(prices_b.index)
    a, b = prices_a.reindex(idx), prices_b.reindex(idx)
    targets = pd.Series([strat(a.iloc[: t + 1], b.iloc[: t + 1]) for t in range(len(idx))], dtype=float)
    failures = []
    if not np.isfinite(targets).all():
        failures.append("strategy produced non-finite positions")
    if (targets.abs() > 1.0 + 1e-9).any():
        failures.append("positions out of range [-1, 1]")
    if result.n_trades == 0:
        failures.append("strategy never takes a position (inert)")
    metrics = {"total_return": result.total_return, "ann_return": result.ann_return,
               "sharpe": result.sharpe, "max_drawdown": result.max_drawdown, "n_trades": result.n_trades}
    return {"passed": len(failures) == 0, "failures": "; ".join(failures),
            "metrics": metrics, "equity_curve": result.equity_curve}
