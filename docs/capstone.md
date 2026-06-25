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

## v1 design (crawl): sequential, single pass, NO loop   [DONE 24-Jun]

Goal of v1: prove the multi-agent SKELETON - isolated specialists + clean boundaries +
an objective done-check. The fix-loop is v2; parallelism + research agent are v3.
STATUS: built + verified (graph.py 4-node flow + main.py user input; is_palindrome green).

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

## v2 (walk): the fix-loop   [IN PROGRESS 25-Jun]
Flow reordered to write_spec -> write_tests -> write_code -> run_sandbox, then a conditional
edge loops run_sandbox -> write_code on failure (coder gets spec + prev code + failures),
until green or MAX_ATTEMPTS (the brake). Tests generated ONCE so the target is fixed (TDD).
- DONE: the loop + brake (state `attempts`, MAX_ATTEMPTS, retry-aware write_code,
  route_after_tests, conditional edge). See Change 011.
- DONE: tested + validated (step 1). Live runs proved the loop's PLUMBING; convergence (a
  coder fixing a REAL bug red -> green) was proven separately by a deterministic planted-bug
  test (check_fixloop.py). The 3-run roman-numeral saga + 3 upstream prompt fixes: Change
  012, Lesson 013.
- DONE: step 2 = build ledger + user-intervention on stuck (Change 013, Lesson 014). The
  ledger (`Annotated[list, add]` of BuildEvents) keeps the per-attempt trajectory; main.py's
  outer while-loop prints a truthful stuck-report and augments the request with accumulated
  human feedback. Validated: a 3-round whack-a-mole converged to a genuine green.
  Finding (the coarse-lever problem): feedback enters at request -> orchestrator, but QA bugs
  live downstream, so steering is whack-a-mole - motivates a SURGICAL feedback lever (edit
  tests directly / QA-visible feedback) later. Known bugs parked: multi-line input() mangling
  (single-line workaround); accumulated feedback may bloat the request.
- DONE: step 3 = while-True multi-build (run_one_build + "build another?" loop).
- DONE (attempt): step 2.5 = surgical feedback-to-QA (thread feedback into the QA's context,
  not just the orchestrator). It WORKED (feedback reaches the QA) but still whack-a-moled,
  revealing the deeper flaw (Lesson 015): no MEMORY + no TARGETING. -> superseded by v3.

### Architectural finding (Lesson 013): the loop routes ONLY to the coder
The conditional edge is {retry: write_code, done: END} - it loops back to the CODER only.
write_tests (QA) runs ONCE and is never re-invoked, so a QA bug (e.g. a test file that will
not import) gets MIS-ROUTED to the coder, who can only rewrite solution.py. A red verdict
says "code and tests DISAGREE", not "the code is wrong" - and there are THREE authors
(orchestrator/QA/coder) but only TWO scripts; the spec author writes no file yet is upstream
of both.

The corrected dividing line is DECIDABILITY, not v2-vs-v3 (Andre's refinement):
- DECIDABLE failure - the test file won't LOAD (pytest exit 2/5). The exit code NAMES the
  culprit (the QA). The SYSTEM can auto-route this back to the QA, NO judge - fine in v2.
  It does NOT ping-pong: each failure has exactly one home (exit 2/5 -> QA; exit 1 -> coder).
- UNDECIDABLE failure - an assertion fails (exit 1). Code and tests disagree; the culprit
  could be code, test, OR spec. This is the ONLY case needing a judge, and the ONLY place
  ping-pong (an endless code <-> test tug-of-war) can arise.
So:
  decidable   -> auto-route to the author (v2, step 2.5).
  undecidable -> v2 escalates to a HUMAN (the human is the judge); v3 builds a JUDGE agent.

### v2 step 2 design: the build ledger + user-intervention on stuck
PRINCIPLE (Andre): the system ASSISTS, it does not GUARANTEE outcomes. On giving up it must
be TRUTHFUL about what it tried and show its results - complete or not - then hand control to
the user. The human is the judge; NO auto-routing / auto-judge in v2.

- BUILD LEDGER (a new state field): a running record of who-wrote-what-and-what-resulted, so
  the stuck-report is truthful and complete. NOT a checkpointer - this is a one-shot task
  (Lesson 012), so an explicit in-state ledger is the right tool; the checkpointer solves a
  different problem (cross-run persistence = the optional step 3 / resume).
    @dataclass
    class BuildEvent:
        step: int          # attempt / order
        author: str        # "orchestrator" | "qa" | "coder"
        artifact: str      # "spec" | "tests" | "code"
        content: str       # what they wrote
        result: dict       # the test_result that followed (passed, exit code, logs)
  State gains `ledger: Annotated[list, add]`; each agent APPENDS its event (reducer = add,
  Lesson 001) instead of overwriting. The checkpointer (Lesson 005) gives a timestamped
  per-step history for free - this adds the explicit author -> artifact label it lacks.
- USER-INTERVENTION ON STUCK: when the brake trips, print a truthful build report FROM the
  ledger (each agent's artifact + the failure logs + attempts), then ask the user how to
  proceed (e.g. edit the spec / retry / quit). Replaces today's silent "give up + END".
- The ledger is introduced in v2 (it fuels the HUMAN handoff) and REUSED in v3 (it fuels
  AUTO-routing). Same data, two consumers - which is why the ledger is a v2 concern, not v3.

DESIGN NOTE - one-shot task vs conversation: the builder is a ONE-SHOT TASK (runs to
completion in ONE invoke), not a conversation. Its working memory (the retry feedback) lives
WITHIN that invoke, in the loop's state. So a while-True loop + a checkpointer are OPTIONAL
enhancements (multi-build sessions; resumable/inspectable builds), NOT required - unlike the
foundations CHAT agent, which needs both to persist memory ACROSS turns. (See Lesson 012.)

## v3 (run): the surgical, stateful build loop   [COMPLETE 26-Jun]
PROBLEM (Lesson 015): v2 regenerates the WHOLE pipeline each intervention round (no MEMORY)
and sprays feedback at every agent (no TARGETING), so the target re-randomises and it
whack-a-moles. v3 = MEMORY + TARGETING: persist the build and EDIT ONLY the wrong artifact.

DECISIONS (26-Jun):
- TARGETING = the HUMAN picks when stuck (stay the judge - the v2 principle). A judge-agent
  that picks automatically is a LATER autonomy upgrade.
- MEMORY = MANUAL carry-forward in main.py (mirrors Lesson 003's chat loop saving `messages`).
  A checkpointer is the LATER Milestone-3-style upgrade.

THE SHIFT: v2's graph is ONE-SHOT (always spec->tests->code). v3's graph is RESUMABLE +
TARGETED - given the current build + a fix_target, re-run ONLY the needed path and PIN the rest.

New pieces:
- STATE: add `fix_target: str` ("" fresh | "spec" | "tests" | "code"). `feedback` already
  exists. Artifacts (spec/tests/code/test_result) are carried forward by main.py.
- MEMORY (main.py): the intervention loop holds the current build dict and feeds it back into
  the next invoke (like Lesson 003 saved `messages = result["messages"]`). A fresh build =
  empty artifacts + fix_target="".
- DISPATCHER (graph entry, a conditional edge - Lesson 007): route on fix_target ->
    ""/"spec" -> write_spec ; "tests" -> write_tests ; "code" -> write_code
- EDIT-AWARE agents: in edit mode each agent gets its PREVIOUS artifact + the feedback and
  EDITS (not regenerates) - exactly as the coder already does on a retry.
- TOPOLOGY: write_spec -> write_tests -> (after_write_tests: if fix_target=="tests" ->
  run_sandbox REUSING existing code; else -> write_code) -> run_sandbox -> route_after_tests
  {retry->write_code, done->END}. So fixing TESTS keeps the code and just re-checks; fixing
  SPEC cascades downstream; fixing CODE is the existing loop.

v3 ROADMAP (vertical slices - one fix-target at a time):
- STEP 1 - the "tests" path   [DONE + PROVEN]: dispatcher + after_write_tests
  conditional + edit-aware write_tests + main.py carry-forward + a "fix which? [spec/tests/
  code]" prompt. Fixes the exact Lesson 015 pain (QA wrote a bad test; spec+code fine).
  SUCCESS CRITERIA: on a "tests" fix the spec stays BYTE-IDENTICAL (pinned, write_spec never
  runs), the QA EDITS (not regenerates) the suite (log shows "[qa] editing tests"), code is
  untouched - and the headline test: does is_palindrome now CONVERGE once the spec holds still
  (where v2 could not)?
- STEP 2 - the "spec" path    [DONE + PROVEN]: make write_spec EDIT-aware (prev spec + feedback ->
  edit, not regenerate). Today "spec" routes correctly via dispatch but still REGENERATES.
  When the spec changes, tests + code cascade (regenerate downstream - that is correct).
- STEP 3 - the "code" path    [DONE]: thread the human feedback into write_code's edit. Today
  the coder edits from TEST FAILURES only; this adds the human's targeted correction. "code"
  already routes + edits via the retry loop - this just opens the human-feedback channel.
After all three: every artifact is human-targetable + edit-aware = v3 complete. Then v3+
(autonomy/breadth, below).

NOTE - what STEP 1 does NOT yet do (so the test reads honestly): a "spec" fix regenerates the
spec (step 2), and a "code" fix uses the existing coder retry without the human's words yet
(step 3). Only the "tests" path is fully surgical right now.

UPDATE (26-Jun, Change 015) - V3 COMPLETE; the NOTE above is now SUPERSEDED: all three
fix_targets are surgical (tests/code/spec edit-aware). PROVEN live - tests-fix converges (spec
pinned/byte-identical, QA edits the suite, code cascades); spec-fix EDITS the spec
(near-identical, only the targeted rule changed) and cascaded to green on attempt 1. The
Lesson 015 whack-a-mole is dissolved. CAVEAT (Lesson 016): edit QUALITY is model-gated -
llama3.2 only PARTIALLY applied the spec edit (kept one contradictory line); it converged only
because no test discriminated on it (Lesson 011). Wiring proven; quality wants a stronger /
per-agent model.

## v3+ (later autonomy / breadth, parked)
- JUDGE agent sets fix_target automatically (full autonomy; the undecidable case) - "replace
  the human judge with machinery". CATCH: once it can blame the TEST and re-invoke the QA, the
  coder and QA can endlessly re-break each other (PING-PONG) - needs a SINGLE combined retry
  cap, and it gives up v2's fixed-target TDD guarantee. Build only after the human-directed v3.
  UPDATE (26-Jun, Change 016 / Lesson 017): BUILT + VALIDATED on qwen2.5-coder. The `judge` node
  has a decidable (load-error -> tests) + LLM (code|tests|spec) tier; ONE combined brake
  (`attempts`, incremented in the judge) = 3 auto-rounds then escalate to the human. Proven: it
  routed -> tests on a dropped `import pytest` and the QA recovered autonomously in 1 round - the
  value the coder-only loop never had. Ping-pong is bounded by the brake; the judge is
  model-gated (needs qwen-class, not llama3.2).
- CHECKPOINTER for build-state (Milestone-3-style upgrade; time-travel/inspect each round).
- parallelise coder/QA from the spec (reducer fan-in, Lesson 001); add a research/spec agent
  (internet/prior-art); structured results with sources/confidence.
- Domain port: backtester agent (yfinance data, Sharpe/drawdown/turnover, forward-bias
  prevention by engine structure). "correct" != "profitable".
