# 📈 Quant Backtester

> **🚀 Live:** [quant-backtester.streamlit.app](https://quant-backtester.streamlit.app)
> An AI quant backtester that's **honest about the three ways backtests lie** — look-ahead, overfitting, and costs — not just plausible-looking code.

Describe a strategy or a money question in plain English. The system writes the strategy, runs it **point-in-time** (look-ahead is structurally impossible), grades it against a **known-correct baseline**, and lets you **stress-test it out-of-sample** — net of transaction costs.

The domain port of a multi-agent [code-builder](../README.md): same orchestrator → coder → self-healing judge skeleton, retargeted from "make the tests pass" to "produce an *honest* backtest."

---

## Why this is different — the honesty stack

NL→backtest is a crowded space ("vibe trading"). Most tools generate code, plot a pretty equity curve, and stop. The differentiator here is **eval-first**: a backtest only earns trust by clearing four gates, in order.

| Gate | The lie it catches | How it's enforced |
|------|--------------------|-------------------|
| **1. Sound** | code that crashes, or quietly peeks at the future | a point-in-time loop (`strategy(prices.iloc[:t+1])`) + a 1-bar position lag (`.shift(1)`) → the strategy *cannot* see data it wouldn't have had. Look-ahead is **structural**, not a guideline. |
| **2. Correct** | plausible-but-wrong logic (the code runs, the number is wrong) | graded against a **human-verified baseline** — a "ruler" — not against self-written tests. Catches the bugs you'd otherwise eyeball. |
| **3. Robust** | curve-fit parameters that won't generalize | rolling **out-of-sample** windows + a per-window buy-and-hold benchmark → *"positive in 4/12 windows, beats buy-and-hold in 3 — looks like beta, not alpha."* |
| **4. Net-of-cost** | high-turnover strategies flattered by a frictionless engine | a default **~5 bps/side** transaction cost. A 240-trade churn strategy's Sharpe flips **+0.38 → −0.38** — it can no longer masquerade as profitable. Buy-and-hold (1 trade) is untouched. |

> **The throughline:** *sound* ≠ *correct* ≠ *profitable-gross* ≠ *robust* ≠ *profitable-net*. Each gate is a different way to be wrong. A request that falls outside a gate's assumptions is **detected and refused** (or surfaced for confirmation), never silently degraded into a wrong-but-plausible answer.

---

## Three engines, one chat box

The router classifies each request and dispatches to the right paradigm:

- **Position** — a single-asset strategy `strategy(history) → target ∈ [-1, 1]`; growth-of-$1 equity, benchmarked vs buy-and-hold. *(SMA / RSI / breakout …)*
- **Contribution** — a cash-flow / DCA comparison: per-leg ticker, cadence, and amount → compare the *multiple*, not the absolute. *("$1000/mo into SPY vs buying every 3-day dip.")*
- **Pairs** — a dollar-neutral spread `strategy_pair(history_a, history_b) → spread position`; log-spread, market-neutral (no buy-and-hold benchmark). *("XLF vs XLI when the spread hits +2 SD.")*

---

## How it works

```
request ─▶ classify ─▶ write spec ─▶ [confirm + edit spec]  ◀─ human-in-the-loop on intent
                                          │
                                          ▼
                            write strategy code  ◀─┐
                                          │        │ self-healing judge:
                                          ▼        │ routes each failure to
                              run through the engine│ the agent at fault
                                          │        │ (code / spec) until sound
                                          ▼        │
                                  soundness check ─┘
                                          │
                                          ▼
                        grade vs ruler · stress-test out-of-sample · net-of-cost
```

- **Editable-spec confirm flow** — the interpretation is shown *before* running; your edited words go straight to the coder (no LLM re-interpretation on the last mile).
- **AST exec-sandbox** — LLM-written strategy code is validated (no dangerous imports / calls / dunders) *before* `exec`, since it runs in-process on a public app.
- **Prompt-driven** — ticker, dates, and amounts are extracted from your request; company names resolve to symbols (Exxon Mobil → XOM) with a confirm step.

---

## Evaluation & results

Two tiers of eval — *deterministic* (the plumbing) and *probabilistic* (the AI):

**Deterministic** (no LLM, exact): the engine, contribution engine, and runner are ground-truthed against hand-computed answers — e.g. buy-and-hold must track the normalized price to **0.00000**. This proves the *verifier itself* is trustworthy before it grades anything.

**Probabilistic** (LLM, pass-rate over N runs): agents graded against **human-verified baselines** — the "ruler." Grading is by **target-series agreement** vs a documented point-in-time rule (bar-by-bar), *not* outcome metrics — so it catches an off-by-one a return-only check would miss. Verdicts: `CORRECT / NEAR (review) / BROKEN / UNSOUND / MISROUTED`.

### Per-agent evals (a shared dataset, not just the final output)
The app is a pipeline (`classify → spec → coder → judge`), so each agent is graded **on its own station** against a versioned dataset (`evals/dataset/requests.json`) — when a run is wrong, you know *which* agent failed. `python -m evals.coverage` shows coverage + which eval grades which agent.

A useful decomposition fell out of this: the **spec writer passes 21/21 on *both* 32B and 7B** (it carries every key fact, including a "Wilder" pin), while the **coder** splits 7B from 32B (below). So the model ceiling is a *coder* ceiling, **not** a *spec* ceiling — the 7B interprets intent fine; it breaks at writing correct point-in-time code. *(Spec grading is keyword-presence — a proxy for dropped params, not deep semantic quality; that's the LLM-as-judge next step.)*

### Model-ceiling experiment
The model is a clean variable (swap `DEEPINFRA_MODEL`, same baselines + data). 3 models × the suite, N=3:

| Strategy | qwen-7B (local) | **Qwen-32B (deployed)** | DeepSeek-V3 (frontier) |
|---|:---:|:---:|:---:|
| SMA 50/200 regime | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 |
| 20-day breakout *(off-by-one causality)* | ❌ 0/3 | ✅ 3/3 | ✅ 3/3 |
| RSI-14 *(any valid variant)* | ❌ broken | ✅ 3/3 | ✅ 3/3 |
| RSI-14 *(Wilder-pinned — spec-following)* | — | — | ⚠️ ~50% (high variance) |

**Findings:**
1. **There's a capability *threshold* between 7B and 32B** — above it, both the code-specialist (32B) *and* the frontier reasoner (V3) respect "prior 20 days *excluding* today"; the 7B fires 15% vs the baseline's 23%. It's a threshold, not a reasoning-vs-coding axis (running the 3rd column *falsified* the initial 2-column hypothesis).
2. **32B already matches V3 — at ~¼ the cost.** Qwen-32B ≈ \$0.66/1M tokens ≈ **~\$0.01/build**; DeepSeek-V3 ≈ **~4× the cost** for *zero* measured gain on this suite → the app is deployed on **32B**.
3. **RSI divergence was a *spec* flaw, not a model gap.** All three capable models independently chose simple-average RSI (Cutler's) over the Wilder baseline → identical ~81% divergence. The fix was the *ruler/spec* (pin the variant or accept both), not the model. No model can resolve an underspecified spec — itself a finding about spec fidelity.

### Reproduce
Run from the `backtester-app/` root as modules (see [`evals/`](evals/README.md) for the full layout):
```bash
DEEPINFRA_MODEL=Qwen/Qwen3-32B                  LLM_PROVIDER=deepinfra python -m evals.probabilistic.eval_suite
DEEPINFRA_MODEL=deepseek-ai/DeepSeek-V3         LLM_PROVIDER=deepinfra python -m evals.probabilistic.eval_suite
python -m evals.probabilistic.eval_suite     # local 7B
```
*(Numbers as of 27-Jun-2026; cost figures are DeepInfra list-price estimates.)*

---

## Run locally

```bash
cd backtester-app
pip install -r requirements.txt

# uses DeepInfra (set DEEPINFRA_API_KEY in .env) or local Ollama
LLM_PROVIDER=deepinfra streamlit run src/ui.py     # or omit the prefix for local Ollama

# ground-truth evals (run as modules from this dir — see evals/README.md)
python -m evals.deterministic.check_engine          # instant, no LLM/network
python -m evals.probabilistic.eval_suite            # the "ruler" + model-ceiling experiment
```

Put `LANGFUSE_*` keys in `.env` to trace every run. The engine, baselines, and `deterministic/` evals have no LLM dependency — only the agents do.

---

## Stack
LangGraph + LangChain · DeepInfra (Qwen3-32B) or Ollama · pandas/numpy engine · yfinance (daily, auto-adjusted) · Streamlit · Langfuse · Python 3.11+

## Layout
```
backtester-app/
├── src/
│   ├── core/            # THE DETERMINISTIC VERIFIER — no LLM dependency (the thesis, made structural)
│   │   ├── engine.py        #   point-in-time loop, 1-bar lag, net-of-cost metrics
│   │   ├── pairs.py         #   dollar-neutral spread engine
│   │   ├── contributions.py #   cash-flow / DCA engine
│   │   ├── robustness.py    #   rolling out-of-sample windows (+ per-window benchmark)
│   │   ├── metrics.py       #   Sharpe / drawdown / CAGR / annual returns
│   │   └── data.py          #   yfinance loader (daily, auto-adjusted)
│   ├── agents.py        # the LLM layer: classify / spec / coder / judge + scope guard + extraction
│   ├── runner.py        # AST sandbox + strategy loader (the bridge: AI code -> core engine)
│   ├── graph.py         # the LangGraph wiring (dispatch + self-healing loop)
│   ├── export.py        # standalone-script generator
│   ├── evals.py         # position-mode ruler (python -m src.evals) — runs GROSS by design
│   └── ui.py            # Streamlit chat, confirm flow, results
├── baselines/           # human-verified reference implementations (the rulers)
├── evals/               # see evals/README.md
│   ├── deterministic/   #   plumbing checks — engine / contributions / runner (no LLM)
│   └── probabilistic/   #   AI rulers — classify / judge / contribution-eval / eval_suite (pass-rate)
└── demos/               # demo_*.py — quick "does the whole thing produce sane output"
```
