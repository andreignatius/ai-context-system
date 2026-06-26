from typing import TypedDict, Annotated
from operator import add
from dataclasses import dataclass


class BacktestState(TypedDict):
    request: str
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
    ledger: Annotated[list, add]


@dataclass
class BuildEvent:
    author: str
    artifact: str
    content: str
