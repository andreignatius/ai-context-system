# Capstone: Multi-Agent Code-Builder

The capstone that fulfils the context-engineering thesis (Write/Select/Compress/Isolate)
by applying it in one coherent product: a multi-agent system that writes and tests Python
code. High-level framing + milestone mapping + staging live in `journal.md` section C.
This file is the living DESIGN doc (it evolves v1 -> v2 -> v3).

A NEW app that reuses the patterns proven in `langgraph-app/` (state / nodes / edges /
conditional edges / isolated context). Same repo, same thesis, new module.

Generic code-builder first (cleanest success signal: tests pass). The skeleton + the
debugging lessons (Lessons 007-010) port to a backtester agent later with little friction.

---

## v1 design (crawl): sequential, single pass, NO loop

Goal of v1: prove the multi-agent SKELETON - isolated specialists + clean boundaries +
an objective done-check. The fix-loop is v2; parallelism + research agent are v3.

### The three agents
| Agent        | Isolated system prompt (own context)                    | IN          | OUT                          |
|--------------|---------------------------------------------------------|-------------|------------------------------|
| Orchestrator | "structure the request into a clear spec; sequence work"| user request| a self-contained `spec`      |
| Coder (B)    | "implement exactly this spec; return only Python code"  | spec        | `code` (a .py string)        |
| QA (C)       | "from the SPEC ALONE (TDD), write a pytest suite"        | spec ONLY   | `{tests, passed, failures}`  |

### The flow (no conditional edges yet)
```
write_spec  ->  write_code  ->  write_and_run_tests  ->  report  ->  END
(orchestrator)   (coder B)         (QA C)
```

### The boundaries (the part that matters - Lesson 010)
- IN -> Coder: the `spec` only (self-contained; the orchestrator does the rewriting).
- IN -> QA:    the `spec` ONLY (decision 24-Jun). TDD - tests derive from the CONTRACT, not
  the code, so they verify CORRECTNESS independently and can catch coder bugs. (Feeding code
  too would bias the tests to confirm what the code does = false green.)
- OUT <- Coder: just `code`.   OUT <- QA: a STRUCTURED `{passed, failures}`.
- ISOLATION proof: each agent builds its OWN message list (own prompt + scoped input);
  the shared state holds only the ARTIFACTS (spec/code/result), never the agents'
  internal messages. Verify after a run: the state has no agent's working chatter.

### Sandbox (executing LLM-written code - the real safety boundary)
- Write `code` + `tests` to a TEMP directory.
- Run `pytest` in a SUBPROCESS with a TIMEOUT (~30s), restricted cwd, no network.
- Capture stdout / stderr / exit code.
- v1 = temp-dir + subprocess + timeout. Full container sandbox = later hardening.
- NEVER exec the generated code in-process.

### The objective "done" check (the clean success signal)
- QA runs the tests -> exit 0 / all pass -> DONE: report code + green tests.
- Any fail -> v1 just REPORTS the failures (the fix-loop is v2).

### Capstone state (its own AgentState)
```python
class BuilderState(TypedDict):
    request: str        # the user's ask
    spec: str           # orchestrator's self-contained spec
    code: str           # coder's output
    tests: str          # QA's pytest file
    test_result: dict   # {passed: bool, failures: str}
    status: str         # "ok" | "failed"
```

### Module layout (proposed)
```
code-builder/
├── main.py            # entry: take a request, run the graph, print result
└── src/
    ├── state.py       # BuilderState
    ├── config.py      # LLM + constants (reuse langgraph-app patterns)
    ├── agents.py      # orchestrator / coder / QA (isolated-context functions)
    ├── sandbox.py     # write to temp dir + run pytest in subprocess (timeout)
    └── graph.py       # wires the sequential flow
```

---

## Open decisions (confirm / adjust before Phase A)
1. Orchestrator scope in v1 -> PROPOSED: thin orchestrator writes a LIGHT spec (enough to
   scope coder/QA), not a research doc. Real spec-writing = a research agent in v3.
2. Module name -> PROPOSED: `code-builder/` (matches `langgraph-app/` style).
3. Agent form -> PROPOSED: plain FUNCTIONS that build their own message list + call the LLM
   (Lesson 010 lightweight path), not subgraphs.
4. First target task -> PROPOSED: a tiny, unambiguous function e.g. `is_palindrome(s)` -
   to debug the pipeline before anything fancy.

---

## Build phases
- Phase A: scaffold the module + state + config; build the 3 agents as isolated functions;
  verify each in isolation (a demo: spec -> code -> tests).
- Phase B: the sandbox (write temp + run pytest) + the done-check; wire the sequential
  graph; run end-to-end on the first target; verify isolation (state holds artifacts only).

## Later (parked)
- v2: fix-loop (QA fail -> orchestrator -> coder iterate) + max-iteration brake +
  human intervention when stuck.
- v3: parallelise coder/QA from the spec (reducer fan-in, Lesson 001); add a research/spec
  agent (internet/prior-art); structured results with sources/confidence.
- Domain port: backtester agent (yfinance data, Sharpe/drawdown/turnover, forward-bias
  prevention by engine structure). "correct" != "profitable".
