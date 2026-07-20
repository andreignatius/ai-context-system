"""The strategy runner = the backtester's SANDBOX (the verifier).

LLM-written strategy code (a string defining `strategy(history) -> float`) is exec'd + run inside a SPAWNED
child process with a wall-clock timeout and a memory cap, so an infinite loop or a memory bomb produces a CLEAN
soundness failure - never a hung / OOM-killed app. Two layers of defense:
  1. AST `_validate` - a cheap GUARDRAIL (blocks os/signal imports, dunders, banned calls) BEFORE we spawn. It
     is NOT a security boundary (blocklists leak); its load-bearing job is that the child can't import
     `signal`, so it can't install a SIGTERM handler to survive our terminate().
  2. The child process - the actual CONTAINMENT: time it out, memory-cap it; worst case = a dead child + a
     clean failure string.
This module imports ONLY stdlib + `core/` (no agents/graph), so the `spawn` re-import in the child stays light
and has no import-time LLM side effect. Design corrections in sandbox-plan.md (cap AFTER warm imports;
get-before-join; reap-and-synthesize). This is the role pytest played in the code-builder - here the engine
IS the test.
"""
import ast
import queue                       # stdlib: multiprocessing.Queue.get(timeout) raises queue.Empty
import multiprocessing
import traceback

import numpy as np                 # warm the heavy imports at MODULE load - OUTSIDE the child's memory cap
import pandas as pd

from .core.engine import run_backtest, compute_targets
from .core.pairs import run_pairs_backtest, compute_pair_targets
from .core.contributions import signal_dates

TIMEOUT_S = 25                     # a 5y daily backtest is <2s on a fast box, but the deployed Streamlit
                                   # Cloud CPU is ~5-10x slower (shared free tier) + each run re-imports
                                   # pandas/numpy in the spawned child -> 10s false-killed legit strategies
                                   # (e.g. RSI on ~1600 bars). 25s keeps the infinite-loop guard while giving
                                   # the slow cloud margin.
MEM_BYTES = 2 * 1024 ** 3          # 2 GB RLIMIT_AS (VSZ); generous - the bomb (~TBs) is caught with huge margin,
                                   # and this sits above the warmed pandas/OpenBLAS baseline (no false kill)

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
            raise ValueError("dunder access not allowed")   # blocks __globals__ / __subclasses__ escapes


def _load_strategy(code: str):
    _validate(code)                # guardrail (AST): reject obvious junk BEFORE exec; the process is the real bound
    namespace = {"pd": pd, "np": np}
    exec(code, namespace)
    if "strategy" not in namespace or not callable(namespace["strategy"]):
        raise ValueError("code must define a callable `strategy(history)`")
    return namespace["strategy"]


def _load_strategy_pair(code: str):
    _validate(code)                # guardrail (AST): same as single-asset
    namespace = {"pd": pd, "np": np}
    exec(code, namespace)
    if "strategy_pair" not in namespace or not callable(namespace["strategy_pair"]):
        raise ValueError("code must define a callable `strategy_pair(history_a, history_b)`")
    return namespace["strategy_pair"]


def _overlap_warning(code, prices):
    """Fix 4: deterministic spec-consistency lint. If the strategy ends in
    `return A if <long> else B if <short> else C`, probe whether <long> and <short> can BOTH be true on
    some bar - a contradiction (the SHORT branch is then dead code, long is checked first). Returns a
    warning string or None. Fires ONLY on the recognised pattern (no false positives on other shapes);
    NEVER raises - a lint must not break a run."""
    try:
        tree = ast.parse(code)
        fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "strategy"), None)
        if fn is None:
            return None
        ret = None                        # the last top-level `A if long else B if short else C`
        for n in fn.body:
            if (isinstance(n, ast.Return) and isinstance(n.value, ast.IfExp)
                    and isinstance(n.value.orelse, ast.IfExp)):
                ret = n
        if ret is None:
            return None
        long_test, short_test = ret.value.test, ret.value.orelse.test
        ret.value = ast.Tuple(elts=[long_test, short_test], ctx=ast.Load())   # probe: return (long, short)
        ast.fix_missing_locations(tree)
        ns = {"pd": pd, "np": np}
        exec(compile(tree, "<overlap-probe>", "exec"), ns)
        probe = ns["strategy"]
        step = max(1, len(prices) // 200)  # sample ~200 bars (bound the O(n^2) point-in-time cost)
        for i in range(20, len(prices), step):
            r = probe(prices.iloc[:i + 1])
            if isinstance(r, tuple) and len(r) == 2 and bool(r[0]) and bool(r[1]):
                return ("long and short conditions can BOTH be true on some bar - the SHORT branch is "
                        "dead code (long is evaluated first). The rule may be self-contradictory "
                        "(e.g. mean-reversion fading the band vs momentum riding it).")
    except Exception:
        return None                       # a lint must NEVER break a run
    return None


# --- BODY functions: run INSIDE the child. Each returns a plain, picklable dict. ---
def _position_body(code, prices):
    try:
        strategy = _load_strategy(code)
    except Exception as e:
        return {"passed": False, "failures": f"load error: {e}", "metrics": {}}
    try:
        targets = compute_targets(prices, strategy)                     # point-in-time targets ONCE, reuse below
        result = run_backtest(prices, strategy, targets=targets)
    except Exception as e:
        tb = traceback.format_exc()
        # Fix 3: a traceback that never enters the STRATEGY (exec'd as "<string>") is a HARNESS/DATA
        # failure the coder cannot fix (e.g. degenerate data reaching the engine) - label it so the loop
        # TERMINATES instead of burning coder attempts. A real strategy bug DOES show a "<string>" frame.
        kind = "runtime error" if ("<string>" in tb or ", in strategy" in tb) else "harness error"
        return {"passed": False, "metrics": {},
                "failures": f"{kind}: {type(e).__name__}: {e}\n{tb}"}
    failures = []
    if not np.isfinite(targets).all():
        failures.append("strategy produced non-finite positions")
    if (targets.abs() > 1.0 + 1e-9).any():
        failures.append("positions out of range [-1, 1]")
    result2 = run_backtest(prices, strategy, targets=compute_targets(prices, strategy))   # determinism recompute
    if not np.array_equal(result.equity_curve.values, result2.equity_curve.values):
        failures.append("strategy is not deterministic")
    metrics = {"total_return": result.total_return, "ann_return": result.ann_return,
               "sharpe": result.sharpe, "max_drawdown": result.max_drawdown, "n_trades": result.n_trades}
    if result.n_trades == 0:
        failures.append("strategy never takes a position (inert) - warm-up may exceed the data length")
    out = {"passed": len(failures) == 0, "failures": "; ".join(failures), "metrics": metrics}
    warn = _overlap_warning(code, prices)          # Fix 4: flag self-contradictory long/short (non-failing)
    if warn:
        out["warnings"] = warn
    return out


def _pairs_body(code, prices_a, prices_b):
    try:
        strat = _load_strategy_pair(code)
    except Exception as e:
        return {"passed": False, "failures": f"load error: {e}", "metrics": {}}
    idx = prices_a.index.intersection(prices_b.index)
    a, b = prices_a.reindex(idx), prices_b.reindex(idx)
    try:
        targets = compute_pair_targets(a, b, strat)          # compute ONCE, reuse for the engine + checks
        result = run_pairs_backtest(a, b, strat, targets=targets)
    except Exception as e:
        tb = traceback.format_exc()                          # Fix 3: harness/data error vs strategy bug
        kind = "runtime error" if ("<string>" in tb or ", in strategy" in tb) else "harness error"
        return {"passed": False, "failures": f"{kind}: {type(e).__name__}: {e}\n{tb}", "metrics": {}}
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


def _signal_body(code, prices):
    """Contribution SIGNAL leg: load + compute the deposit dates the strategy fires on. -> {passed, dates, ...}."""
    try:
        strategy = _load_strategy(code)
    except Exception as e:
        return {"passed": False, "dates": [], "failures": f"load error: {e}"}
    try:
        dates = signal_dates(prices, strategy)
    except Exception as e:
        return {"passed": False, "dates": [], "failures": f"runtime error: {type(e).__name__}: {e}"}
    return {"passed": True, "dates": list(dates), "failures": ""}


_BODIES = {"position": _position_body, "pairs": _pairs_body, "signal": _signal_body}


def _worker(kind, args, q, mem_bytes):
    """Runs in the SPAWNED child. Cap memory (AFTER the module's heavy imports), then run the body under a try
    that turns MemoryError / any exception into a clean failure dict pushed to the queue."""
    try:
        import resource
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard))    # soft cap only; keep hard (usually unlimited)
    except Exception:
        pass                       # Windows / no `resource` -> degrade to timeout-only
    try:
        q.put(_BODIES[kind](*args))
    except MemoryError:
        q.put({"passed": False, "failures": "strategy exceeded the memory limit", "metrics": {}, "dates": []})
    except Exception as e:
        q.put({"passed": False, "failures": f"runtime error: {type(e).__name__}: {e}", "metrics": {}, "dates": []})


def _sandboxed(kind, args, timeout=TIMEOUT_S, mem_bytes=MEM_BYTES):
    ctx = multiprocessing.get_context("spawn")     # spawn: fresh interp (light re-import, no fork-with-threads hazard)
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(kind, args, q, mem_bytes), daemon=True)
    p.start()
    try:
        result = q.get(timeout=timeout)            # GET FIRST: the wall-clock guard AND it drains the pipe buffer
    except queue.Empty:                            # timed out, or the child was killed before it could reply
        result = {"passed": False, "metrics": {}, "dates": [],
                  "failures": f"strategy timed out or exceeded resource limits (>{timeout}s)"}
    finally:
        p.terminate()                              # SIGTERM (no-op if it already exited)
        p.join(1)
        if p.is_alive():
            p.kill(); p.join()                     # escalate to SIGKILL if it ignored SIGTERM
        q.cancel_join_thread()                     # don't let the queue feeder thread block the parent
    return result


def run_strategy(strategy_code: str, prices, timeout=TIMEOUT_S, mem_bytes=MEM_BYTES) -> dict:
    return _sandboxed("position", (strategy_code, prices), timeout, mem_bytes)


def run_pair_strategy(code: str, prices_a, prices_b, timeout=TIMEOUT_S, mem_bytes=MEM_BYTES) -> dict:
    return _sandboxed("pairs", (code, prices_a, prices_b), timeout, mem_bytes)


def run_signal_dates(code: str, prices, timeout=TIMEOUT_S, mem_bytes=MEM_BYTES) -> dict:
    """Sandboxed deposit-date computation for a contribution SIGNAL leg -> {passed, dates, failures}."""
    return _sandboxed("signal", (code, prices), timeout, mem_bytes)
