import requests
import streamlit as st
import os, sys

# API = "http://localhost:8000/build"
st.set_page_config(page_title="Code Builder", page_icon="🛠️", layout="centered")
st.title("🛠️ Code Builder")

st.session_state.setdefault("history", [])   # list of {"role", "text"|"build"}
st.session_state.setdefault("last", None)     # latest build result (for fixing)
st.session_state.setdefault("request", "")

# make `src` importable no matter how Streamlit launches this file (adds code-builder/ to path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# bridge Streamlit Cloud secrets -> env BEFORE importing src (config reads LLM_PROVIDER at import)
try:
    for _k in ("LLM_PROVIDER", "DEEPINFRA_API_KEY"):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass            # no secrets.toml locally -> fall back to .env / shell env

from src.graph import build_graph
from src.config import get_langfuse_handler

@st.cache_resource                  # build the graph ONCE, not on every Streamlit rerun
def _get_graph():
    return build_graph()

_graph = _get_graph()
_handler = get_langfuse_handler()

def run_build(payload: dict) -> dict:
    """In-process replacement for POST /build - same input/output shape as api.py."""
    state = {"request": payload["request"], "feedback": payload.get("feedback", ""),
             "fix_target": payload.get("fix_target", "")}
    if state["fix_target"]:         # a human fix -> carry prior artifacts forward (memory)
        state.update({"spec": payload.get("spec", ""), "tests": payload.get("tests", ""),
                      "code": payload.get("code", ""), "test_result": payload.get("test_result", {})})
    r = _graph.invoke(state, config={"callbacks": [_handler]})
    return {"status": r["status"], "spec": r["spec"], "code": r["code"],
            "tests": r["tests"], "test_result": r.get("test_result", {})}

def render_build(b):
    if b["status"] == "ok":
        st.success("✅ verified — all tests pass")
    else:
        st.error("❌ stuck — the auto-judge gave up")
    st.code(b["code"], language="python")
    with st.expander("tests"): st.code(b["tests"], language="python")
    with st.expander("spec"):  st.text(b["spec"])


# 1. replay the conversation as bubbles
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.write(turn["text"])
        else:
            render_build(turn["build"])


# 2. fix panel — appears in an assistant bubble when the last build is stuck
last = st.session_state.last
if last and last["status"] != "ok":
    with st.chat_message("assistant"):
        st.warning("I'm stuck. Pick what to fix and tell me what's wrong:")
        target   = st.selectbox("Fix which?", ["tests", "code", "spec"])
        feedback = st.text_area("What should change?")
        if st.button("Apply fix"):
            # b = requests.post(API, json={
            b = run_build({
                "request": st.session_state.request,
                "spec": last["spec"], "tests": last["tests"], "code": last["code"],
                "test_result": last.get("test_result", {}),
                "fix_target": target, "feedback": feedback,
            # }).json()
            })
            st.session_state.history += [{"role": "user", "text": f"fix **{target}**: {feedback}"},
                                         {"role": "assistant", "build": b}]
            st.session_state.last = b
            st.rerun()

# 3. the docked chat input — a fresh build
if req := st.chat_input("What function should I build?"):
    st.session_state.request = req
    with st.spinner("building… (orchestrator → QA → coder → judge)"):
        # b = requests.post(API, json={"request": req}).json()
        b = run_build({"request": req})
    st.session_state.history += [{"role": "user", "text": req},
                                 {"role": "assistant", "build": b}]
    st.session_state.last = b
    st.rerun()
