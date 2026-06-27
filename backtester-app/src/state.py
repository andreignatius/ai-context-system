from typing import TypedDict, Annotated
from operator import add
from dataclasses import dataclass


class BacktestState(TypedDict):
    request: str
    mode: str               # "position" | "contribution" - which engine to route to (M8)
    spec: str
    strategy_code: str
    run_result: dict        # from run_strategy: {passed, failures, metrics}
    status: str             # "ok" | "failed" | ""
    attempts: int
    feedback: str
    fix_target: str         # "" fresh | "code" | "spec"
    prices: object          # price Series the build runs on (UI-supplied; falls back to default)
    ticker: str
    period: str
    start_date: str         # extracted from the request ('since 2021' -> '2021-01-01'); "" = use period
    amount: float           # contribution mode: $ per deposit (default 1000)
    legs: list              # contribution mode (UI-supplied): [{cadence, amount, label}, ...] - two
                            # deposit schedules to compare; absent -> legacy signal-vs-monthly @ amount
    ticker_b: str           # pairs mode: the SECOND ticker (ticker = A, ticker_b = B)
    pairs_result: dict      # pairs mode: {ticker_a, ticker_b, metrics, equity_curve}
    scope_error: bool       # the request is OUT OF SCOPE (e.g. identical legs / cross-asset) - refuse,
                            # do NOT self-heal (the judge can fix code, not an unsupported request)
    contribution_result: dict   # contribution mode: {amount, legs, signal:{...}, dca:{...}}
    ledger: Annotated[list, add]


@dataclass
class BuildEvent:
    author: str
    artifact: str
    content: str
