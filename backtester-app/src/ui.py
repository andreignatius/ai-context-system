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

from src.graph import build_graph
from src.runner import _load_strategy
from src.engine import run_backtest
from src.data import load_prices

st.title("📈 Quant Backtester")
st.caption("Describe a strategy in plain English -> spec -> code -> backtest -> self-healing judge.")

@st.cache_resource
def _graph():
    return build_graph()

@st.cache_data
def _prices():
    return load_prices("SPY", "2y")

st.session_state.setdefault("history", [])

def render(b):
    if b["status"] == "ok":
        st.success("✅ sound — the strategy runs and is valid")
    else:
        st.error("❌ stuck — the judge gave up after retries")
    m = b["run_result"]["metrics"]
    if m:
        c = st.columns(4)
        c[0].metric("Total return", f"{m['total_return']:+.1%}")
        c[1].metric("Ann. return", f"{m['ann_return']:+.1%}")
        c[2].metric("Sharpe", f"{m['sharpe']:.2f}")
        c[3].metric("Max drawdown", f"{m['max_drawdown']:.1%}")
        st.caption(f"{m['n_trades']} trades")
    if b.get("equity") is not None:
        st.line_chart(b["equity"])
    st.code(b["strategy_code"], language="python")
    with st.expander("strategy spec"):
        st.text(b["spec"])
    if b["run_result"]["failures"]:
        st.warning(b["run_result"]["failures"])

# replay the conversation
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["text"]) if turn["role"] == "user" else render(turn["build"])

# chat input -> build
if req := st.chat_input("e.g. 'go long when the 20-day return is positive, else flat'"):
    with st.spinner("building… (orchestrator → coder → sandbox → judge)"):
        b = _graph().invoke({"request": req})
        b["equity"] = None
        if b["status"] == "ok":                       # re-run to get the equity curve for the plot
            b["equity"] = run_backtest(_prices(), _load_strategy(b["strategy_code"])).equity_curve
    st.session_state.history += [{"role": "user", "text": req},
                                 {"role": "assistant", "build": b}]
    st.rerun()
