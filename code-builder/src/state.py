"""Shared state for the code-builder graph (the orchestrator's view).

NOTE: there is deliberately NO `messages` field here. The agents do NOT share a
conversation - each builds its OWN isolated message list. The shared state holds only
the ARTIFACTS that cross between them (spec, code, tests, result). That is ISOLATE.
"""

from typing import TypedDict

class BuilderState(TypedDict):
    request: str        # user ask
    spec: str           # orchestrator self-contained spec
    code: str           # coder output
    tests: str          # QA pytest file
    test_result: dict   # {"passed": bool, "failures": str}
    status: str         # "ok" | "failed"