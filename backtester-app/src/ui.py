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
from src.graph import build_graph
from src.runner import _load_strategy
from src.engine import run_backtest
from src.data import load_prices
from src.metrics import buy_and_hold, longest_drawdown_days, annual_returns, dca

st.title("📈 Quant Backtester")

@st.cache_resource
def _graph():
    return build_graph()

@st.cache_data
def _prices(ticker, period):
    return load_prices(ticker, period)

# --- sidebar: data config ---
with st.sidebar:
    st.header("Data")
    _t = st.selectbox("Ticker", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"], index=0)
    ticker = st.text_input("…or a custom ticker", _t).strip().upper()
    period = st.selectbox("Period", ["6m", "1y", "2y", "3y", "5y"], index=2)

st.caption(f"{ticker} · daily close (auto-adjusted) · yfinance · {period} · "
           f"equity = growth of $1, fully invested when long")

st.session_state.setdefault("history", [])


def render(b):
    if b["status"] == "ok":
        st.success("✅ sound — the strategy runs and is valid")
    else:
        st.error("❌ stuck — the judge gave up after retries")

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
        # DCA is a different cash-flow basis -> report as an end metric, not a same-axis line
        _, dca_final, dca_invested = dca(px, 100.0)
        bh_mult = px.iloc[-1] / px.iloc[0]
        st.caption(f"DCA \\$100/mo: \\${dca_final:,.0f} on \\${dca_invested:,.0f} invested "
                   f"({dca_final / dca_invested:.2f}x)  ·  buy & hold: {bh_mult:.2f}x")
        with st.expander("annual returns"):
            st.dataframe(annual_returns(eq).to_frame("return").style.format("{:+.1%}"))

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

# chat input -> build
if req := st.chat_input("e.g. 'go long when the 20-day return is positive, else flat'"):
    prices = _prices(ticker, period)
    with st.spinner("building… (orchestrator → coder → sandbox → judge)"):
        b = _graph().invoke({"request": req, "prices": prices, "ticker": ticker, "period": period})
        b["prices"] = prices
        b["equity"] = None
        if b["status"] == "ok":
            b["equity"] = run_backtest(prices, _load_strategy(b["strategy_code"])).equity_curve
    st.session_state.history += [{"role": "user", "text": req},
                                 {"role": "assistant", "build": b}]
    st.rerun()
