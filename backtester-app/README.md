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

## Run locally

```bash
cd backtester-app
pip install -r requirements.txt

# uses DeepInfra (set DEEPINFRA_API_KEY in .env) or local Ollama
LLM_PROVIDER=deepinfra streamlit run src/ui.py     # or omit the prefix for local Ollama

# ground-truth evals (the "ruler" in action; needs network for market data)
python eval_suite.py
python check_contribution_eval.py
```

Put `LANGFUSE_*` keys in `.env` to trace every run. The engine, baselines, and evals have no LLM dependency — only the agents do.

---

## Stack
LangGraph + LangChain · DeepInfra (Qwen2.5-Coder-32B) or Ollama · pandas/numpy engine · yfinance (daily, auto-adjusted) · Streamlit · Langfuse · Python 3.11+

## Layout
```
backtester-app/
├── src/
│   ├── agents.py        # classify / spec / coder / judge + ticker & leg extraction
│   ├── engine.py        # the trusted verifier: point-in-time loop, 1-bar lag, net-of-cost metrics
│   ├── pairs.py         # the dollar-neutral spread engine
│   ├── contributions.py # the cash-flow / DCA engine
│   ├── robustness.py    # rolling out-of-sample windows (+ per-window benchmark)
│   ├── runner.py        # AST sandbox + strategy loader
│   ├── graph.py         # the LangGraph wiring (dispatch + self-healing loop)
│   ├── evals.py         # ground-truth grading (the "ruler") — runs GROSS by design
│   └── ui.py            # Streamlit chat, confirm flow, results
├── baselines/           # human-verified reference implementations (the rulers)
├── check_*.py           # ground-truth eval harnesses per flow
└── eval_suite.py        # the suite runner
```
