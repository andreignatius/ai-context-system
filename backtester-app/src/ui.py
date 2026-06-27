import os, sys
import streamlit as st
st.set_page_config(page_title="Quant Backtester", page_icon="📈", layout="centered")

# make `src` importable regardless of how Streamlit launches this file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# bridge Streamlit Cloud secrets -> env BEFORE importing src (config reads LLM_PROVIDER at import)
try:
    for _k in ("LLM_PROVIDER", "DEEPINFRA_API_KEY", "DEEPINFRA_MODEL",
               "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass                       # no secrets.toml locally -> fall back to .env / shell env

import pandas as pd
from src.graph import build_run_graph
from src.agents import classify, write_spec, extract_legs, _leg_label
from src.runner import _load_strategy
from src.engine import run_backtest
from src.data import load_prices
from src.metrics import buy_and_hold, longest_drawdown_days, annual_returns, dca
from src.export import full_script, full_contribution_script
from src.config import get_langfuse_handler

st.title("📈 Quant Backtester")

@st.cache_resource
def _run_graph():
    return build_run_graph()

@st.cache_resource
def _handler():                       # Langfuse tracer (no-op if LANGFUSE_* secrets are unset)
    return get_langfuse_handler()

@st.cache_data
def _prices(ticker, period, start):
    return load_prices(ticker, period, start)

# --- sidebar: data config ---
with st.sidebar:
    st.header("Data")
    _t = st.selectbox("Ticker", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"], index=0)
    ticker = st.text_input("…or a custom ticker", _t).strip().upper()
    period = st.selectbox("Period", ["6m", "1y", "2y", "3y", "5y"], index=2)
    use_start = st.checkbox("Use a start date (overrides period)")
    start = st.date_input("Start date", value=pd.Timestamp("2021-01-01")) if use_start else None

_range = f"from {start}" if start else period
st.caption(f"{ticker} · daily close (auto-adjusted) · yfinance · {_range} · "
           f"equity = growth of $1, fully invested when long")

st.session_state.setdefault("history", [])
st.session_state.setdefault("draft", None)   # pending interpretation awaiting confirm/fix


def render(b):
    if b["status"] == "ok":
        st.success("✅ sound — the strategy runs and is valid")
    else:
        st.error("❌ stuck — the judge gave up after retries")

    # CONTRIBUTION mode: a dollar comparison (different shape from the position render)
    if b.get("mode") == "contribution":
        cr = b.get("contribution_result")
        px = b.get("prices")
        if cr:
            legs = cr.get("legs")
            if legs:                                   # generic two-leg shape (cadence + own amount)
                idx = [l["label"] for l in legs]
                per_dep = [l["amount"] for l in legs]
                rows = legs
            else:                                      # legacy shape (signal vs monthly @ one amount)
                rows = [cr["signal"], cr["dca"]]
                idx = ["Buy-the-signal", "Monthly DCA"]
                per_dep = [cr.get("amount", 1000.0)] * 2
            if px is not None:
                st.caption(f"{px.index[0].date()} -> {px.index[-1].date()}")
            comp = pd.DataFrame(
                {"per deposit": per_dep,
                 "deposits": [r["n"] for r in rows],
                 "invested": [r["invested"] for r in rows],
                 "final": [r["final"] for r in rows],
                 "multiple": [r["final"] / r["invested"] if r["invested"] else 0.0 for r in rows]},
                index=idx)
            st.dataframe(comp.style.format(
                {"per deposit": "${:,.0f}", "invested": "${:,.0f}", "final": "${:,.0f}", "multiple": "{:.2f}x"}))
            if cr.get("signal_curve") is not None:      # portfolio value over time (leg A vs leg B)
                st.line_chart(pd.DataFrame({idx[0]: cr["signal_curve"], idx[1]: cr["dca_curve"]}))
            st.caption("Compare the MULTIPLE, not the absolute final (deposit totals differ).")
            if px is not None:
                with st.expander("data (preview + download)"):
                    st.dataframe(px.rename("close").to_frame().tail(10))
                    st.download_button("Download prices (CSV)", px.to_csv(),
                                       file_name=f"{b.get('ticker', 'data')}.csv", mime="text/csv")
                with st.expander("📄 full contribution script (reproducible end-to-end)"):
                    script = full_contribution_script(b["strategy_code"], b.get("ticker", "SPY"),
                                                      px.index[0].date(), cr.get("legs"))
                    st.code(script, language="python")
                    st.download_button("Download full script (.py)", script,
                                       file_name=f"contribution_{b.get('ticker', 'SPY')}.py",
                                       mime="text/x-python")
        elif b.get("run_result", {}).get("failures"):
            st.warning(b["run_result"]["failures"])
        st.code(b["strategy_code"], language="python")
        with st.expander("strategy spec"):
            st.text(b["spec"])
        return

    m = b["run_result"]["metrics"]
    eq = b.get("equity")
    px = b.get("prices")
    if m:
        c = st.columns(5)
        c[0].metric("Total return", f"{m['total_return']:+.1%}")
        c[1].metric("CAGR", f"{m['ann_return']:+.1%}")
        c[2].metric("Sharpe", f"{m['sharpe']:.2f}")
        c[3].metric("Max drawdown", f"{m['max_drawdown']:.1%}")
        if eq is not None:
            c[4].metric("Longest DD", f"{longest_drawdown_days(eq)}d")
        st.caption(f"{m['n_trades']} trades")

    # benchmark overlay: strategy vs buy-and-hold (same $1 lump-sum basis -> directly comparable)
    if eq is not None and px is not None:
        st.line_chart(pd.DataFrame({"Strategy": eq, "Buy & Hold": buy_and_hold(px)}))
        # all three multiples on one line so the STRATEGY return is comparable to the benchmarks
        _, dca_final, dca_invested = dca(px, 100.0)
        strat_mult = 1 + m["total_return"]
        bh_mult = px.iloc[-1] / px.iloc[0]
        st.caption(f"**Strategy {strat_mult:.2f}x**  ·  buy & hold {bh_mult:.2f}x  ·  "
                   f"DCA \\$100/mo {dca_final / dca_invested:.2f}x "
                   f"(\\${dca_final:,.0f} on \\${dca_invested:,.0f})")
        st.caption(f"period: {px.index[0].date()} -> {px.index[-1].date()}  ·  {len(px)} trading days")
        with st.expander("annual returns"):
            st.dataframe(annual_returns(eq).to_frame("return").style.format("{:+.1%}"))
        with st.expander("data (preview + download)"):
            st.dataframe(px.rename("close").to_frame().tail(10))
            st.download_button("Download prices (CSV)", px.to_csv(),
                               file_name=f"{b.get('ticker', 'data')}_{b.get('period', '')}.csv",
                               mime="text/csv")
            st.download_button("Download strategy (.py)", b["strategy_code"],
                               file_name="strategy.py", mime="text/x-python")
        with st.expander("📄 full backtest script (reproducible end-to-end)"):
            script = full_script(b["strategy_code"], b.get("ticker", "SPY"), b.get("period", "2y"))
            st.caption("Data load -> signal -> engine -> metrics -> plot, in one standalone file.")
            st.code(script, language="python")
            st.download_button("Download full script (.py)", script,
                               file_name=f"backtest_{b.get('ticker', 'SPY')}.py",
                               mime="text/x-python")

    st.code(b["strategy_code"], language="python")
    with st.expander("strategy spec"):
        st.text(b["spec"])
    if b["run_result"]["failures"]:
        st.warning(b["run_result"]["failures"])


# replay the conversation
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.write(turn["text"])
        else:
            render(turn["build"])

# --- confirm flow: a pending DRAFT (the interpretation) the user approves or corrects BEFORE running ---
draft = st.session_state.get("draft")
if draft:
    with st.chat_message("user"):
        st.write(draft["request"])
    with st.chat_message("assistant"):
        st.info("Here's how I read your request — **edit the spec if needed, then run.** "
                "Your words go straight to the coder (no LLM re-interpretation):")
        st.caption(f"engine: **{draft['mode']}**  ·  start: **{draft.get('start_date') or '(sidebar period)'}**")

        # CONTRIBUTION: two editable legs (cadence + own amount) -> "weekly $250 vs monthly $1000".
        # Direct user control (c2 philosophy); prefilled by extract_legs, but the user has final say.
        leg_inputs = None
        if draft.get("mode") == "contribution":
            st.markdown("**Compare two deposit schedules** — set each leg's cadence + dollar amount:")
            hint = list(draft.get("legs") or [])
            while len(hint) < 2:                       # default the second leg to monthly DCA
                hint.append({"cadence": "monthly", "amount": draft.get("amount", 1000.0)})
            cad_opts = ["signal", "weekly", "monthly"]
            leg_inputs = []
            for i in range(2):
                c1, c2 = st.columns(2)
                hc = hint[i].get("cadence") if hint[i].get("cadence") in cad_opts else "monthly"
                cad = c1.selectbox(f"Leg {chr(65 + i)} cadence", cad_opts,
                                   index=cad_opts.index(hc), key=f"leg{i}_cad")
                amt = c2.number_input(f"Leg {chr(65 + i)} $ / deposit", min_value=1.0,
                                      value=float(hint[i].get("amount") or 1000.0), step=50.0, key=f"leg{i}_amt")
                leg_inputs.append({"cadence": cad, "amount": amt})
            if any(l["cadence"] == "signal" for l in leg_inputs):
                st.caption("A **signal** leg deposits when the strategy fires — edit the spec below to define it.")
        else:
            st.caption(f"${draft.get('amount', 1000):,.0f} per deposit")

        edited_spec = st.text_area("interpreted spec (editable)", value=draft["spec"], height=320, key="spec_edit")
        run_clicked = st.button("✅ Run it", type="primary")

    if run_clicked:
        prices = _prices(ticker, period, draft.get("start_date") or start)
        with st.spinner("running… (coder → engine → self-heal)"):
            state = {**draft, "spec": edited_spec, "prices": prices, "ticker": ticker, "period": period,
                     "fix_target": "", "feedback": ""}     # the EDITED spec is what the coder builds from
            if leg_inputs:                                 # user-confirmed legs -> per-leg cadence + amount
                state["legs"] = [{"cadence": l["cadence"], "amount": l["amount"],
                                  "label": _leg_label(l["cadence"], l["amount"])} for l in leg_inputs]
            b = _run_graph().invoke(state, config={"callbacks": [_handler()]})
            b["prices"] = prices
            b["equity"] = None
            if b.get("mode") != "contribution" and b["status"] == "ok":
                b["equity"] = run_backtest(prices, _load_strategy(b["strategy_code"])).equity_curve
        st.session_state.history += [{"role": "user", "text": draft["request"]},
                                     {"role": "assistant", "build": b}]
        st.session_state.draft = None
        st.rerun()

# chat input -> DRAFT (read the request + show the interpretation; wait for confirm)
if req := st.chat_input("Describe a strategy, or a money question…"):
    s = {"request": req}
    with st.spinner("reading your request…"):
        s.update(classify(s))                 # mode, start_date, amount (extracted)
        s.update(write_spec(s))               # the interpreted spec
        if s.get("mode") == "contribution":   # prefill the two leg controls (UI-only; eval path untouched)
            s.update(extract_legs(s))
    st.session_state.draft = s
    st.rerun()
