# Evaluation

Two tiers, deliberately separated:

## `deterministic/` — ground-truth the *plumbing* (no LLM, no network, exact)
Proves the **verifier itself** is trustworthy before it grades any AI output. Exact reconciliation against
hand-computed answers — no model, no randomness.

| Script | Checks |
|---|---|
| `check_engine.py` | the backtest engine: buy-and-hold must track the normalized price exactly |
| `check_contributions.py` | the contribution engine: deposit math + monthly/signal schedules on toy series |
| `check_contribution_flow.py` | the `contribution_run` graph node in isolation (toy code + prices) |
| `check_runner.py` | `run_strategy`: loads + runs strategy code (good & bad cases) |

## `probabilistic/` — ground-truth the *AI* (LLM, pass-rate over N runs)
The model is non-deterministic (temp 0.5), so these report a **pass rate**, not a single yes/no. Graded by
**target-series agreement** vs a documented point-in-time baseline (bar-by-bar) — not outcome metrics, so an
off-by-one a return-only check would miss is caught. Verdicts: `CORRECT / NEAR (review) / BROKEN / UNSOUND /
MISROUTED`.

Most of these read a **shared versioned dataset** — `dataset/requests.json` (one `(request -> expected)`
bank), so classify + spec are graded from a single source of truth. `python -m evals.coverage` prints what it
covers + which eval grades which agent (no LLM).

| Script (per agent) | Checks |
|---|---|
| `check_classify.py` | **classify** — router + param extraction (mode / ticker / start / amount), incl. the feasibility gate |
| `eval_spec.py` | **spec** — orchestrator faithfulness: does the spec carry the request's key facts (catches dropped params)? |
| `eval_suite.py` | **coder** — multi-baseline **ruler** + **model-ceiling experiment** (swap models, same baselines) |
| `check_judge.py` | **judge** — does a failure message route to the right culprit (code/spec/tests)? |
| `check_contribution_eval.py` | the contribution **ruler** — agent vs `baselines/buy_the_dip.py` |

*(The position-mode ruler lives in `src/evals.py`, run as `python -m src.evals` — it's tightly coupled to the
graph it tests.)*

**Finding (29-Jun):** the spec eval passes 21/21 on **both** 32B and 7B → the model ceiling is a *coder*
ceiling, not a *spec* ceiling. The 7B interprets intent into a faithful spec fine; it breaks at writing
correct point-in-time code (where `eval_suite` *does* split 7B from 32B). Keyword-presence is a proxy for
faithfulness (catches dropped params), not deep semantic quality — that's the LLM-as-judge next step.

## Running
Run **from the `backtester-app/` root** as modules (so `src/` and `baselines/` resolve):

```bash
# deterministic (instant, no deps)
python -m evals.deterministic.check_engine
python -m evals.deterministic.check_contributions

# probabilistic (needs the LLM; network for the rulers)
python -m evals.probabilistic.check_classify
python -m evals.probabilistic.eval_suite                                    # local 7B
DEEPINFRA_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct LLM_PROVIDER=deepinfra python -m evals.probabilistic.eval_suite
DEEPINFRA_MODEL=deepseek-ai/DeepSeek-V3         LLM_PROVIDER=deepinfra python -m evals.probabilistic.eval_suite
```

Results + the cross-model comparison: see the main [README](../README.md#evaluation--results).
