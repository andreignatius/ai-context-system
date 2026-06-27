import os, sys
import streamlit as st
st.set_page_config(page_title="Quant Backtester", page_icon="📈", layout="centered")

# make `src` importable regardless of how Streamlit launches this file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# bridge Streamlit Cloud secrets -> env BEFORE importing src (config reads LLM_PROVIDER at import)
try:
    for _k in ("LLM_PROVIDER", "DEEPINFRA_API_KEY",
               "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass                       # no secrets.toml locally -> fall back to .env / shell env

import pandas as pd
from src.graph import build_run_graph
from src.agents import classify, write_spec
from src.runner import _load_strategy
from src.engine import run_backtest
from src.data import load_prices
from src.metrics import buy_and_hold, longest_drawdown_days, annual_returns, dca
from src.export import full_script, full_contribution_script

st.title("📈 Quant Backtester")

@st.cache_resource
def _run_graph():
    return build_run_graph()

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
            s, d = cr["signal"], cr["dca"]
            rng = f"  ·  {px.index[0].date()} -> {px.index[-1].date()}" if px is not None else ""
            st.caption(f"${cr['amount']:,.0f} per deposit{rng}")
            comp = pd.DataFrame(
                {"deposits": [s["n"], d["n"]],
                 "invested": [s["invested"], d["invested"]],
                 "final": [s["final"], d["final"]],
                 "multiple": [s["final"] / s["invested"] if s["invested"] else 0.0,
                              d["final"] / d["invested"] if d["invested"] else 0.0]},
                index=["Buy-the-signal", "Monthly DCA"])
            st.dataframe(comp.style.format(
                {"invested": "${:,.0f}", "final": "${:,.0f}", "multiple": "{:.2f}x"}))
            if cr.get("signal_curve") is not None:      # portfolio value over time
                st.line_chart(pd.DataFrame({"Buy-the-signal": cr["signal_curve"],
                                            "Monthly DCA": cr["dca_curve"]}))
            st.caption("Compare the MULTIPLE, not the absolute final (deposit totals differ).")
            if px is not None:
                with st.expander("data (preview + download)"):
                    st.dataframe(px.rename("close").to_frame().tail(10))
                    st.download_button("Download prices (CSV)", px.to_csv(),
                                       file_name=f"{b.get('ticker', 'data')}.csv", mime="text/csv")
                with st.expander("📄 full contribution script (reproducible end-to-end)"):
                    script = full_contribution_script(b["strategy_code"], b.get("ticker", "SPY"),
                                                      px.index[0].date(), cr["amount"])
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
        st.caption(f"engine: **{draft['mode']}**  ·  start: **{draft.get('start_date') or '(sidebar period)'}**"
                   f"  ·  ${draft.get('amount', 1000):,.0f} per deposit")
        edited_spec = st.text_area("interpreted spec (editable)", value=draft["spec"], height=320, key="spec_edit")
        run_clicked = st.button("✅ Run it", type="primary")

    if run_clicked:
        prices = _prices(ticker, period, draft.get("start_date") or start)
        with st.spinner("running… (coder → engine → self-heal)"):
            state = {**draft, "spec": edited_spec, "prices": prices, "ticker": ticker, "period": period,
                     "fix_target": "", "feedback": ""}     # the EDITED spec is what the coder builds from
            b = _run_graph().invoke(state)
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
    st.session_state.draft = s
    st.rerun()
