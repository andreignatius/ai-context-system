"""Shared state for the code-builder graph (the orchestrator's view).

NOTE: there is deliberately NO `messages` field here. The agents do NOT share a
conversation - each builds its OWN isolated message list. The shared state holds only
the ARTIFACTS that cross between them (spec, code, tests, result). That is ISOLATE.
"""

from typing import Annotated, TypedDict
from operator import add            # list + list = append; a valid reducer (Lesson 001)
from dataclasses import dataclass, field


@dataclass
class BuildEvent:
    author: str     # "orchestrator" | "qa" | "coder" | "sandbox"
    artifact: str   # "spec" | "tests" | "code" | "result"
    content: str    # the text written, or a short result summary


class BuilderState(TypedDict):
    request: str
    spec: str
    code: str
    tests: str
    test_result: dict
    status: str
    attempts: int
    ledger: Annotated[list, add]    # append-only trail of BuildEvents (the trajectory)
