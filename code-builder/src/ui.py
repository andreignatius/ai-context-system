import requests
import streamlit as st

API = "http://localhost:8000/build"
st.set_page_config(page_title="Code Builder", page_icon="🛠️", layout="centered")
st.title("🛠️ Code Builder")

st.session_state.setdefault("history", [])   # list of {"role", "text"|"build"}
st.session_state.setdefault("last", None)     # latest build result (for fixing)
st.session_state.setdefault("request", "")

def render_build(b):
    st.success("✅ verified — all tests pass") if b["status"] == "ok" \
        else st.error("❌ stuck — the auto-judge gave up")
    st.code(b["code"], language="python")
    with st.expander("tests"): st.code(b["tests"], language="python")
    with st.expander("spec"):  st.text(b["spec"])

# 1. replay the conversation as bubbles
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["text"]) if turn["role"] == "user" else render_build(turn["build"])

# 2. fix panel — appears in an assistant bubble when the last build is stuck
last = st.session_state.last
if last and last["status"] != "ok":
    with st.chat_message("assistant"):
        st.warning("I'm stuck. Pick what to fix and tell me what's wrong:")
        target   = st.selectbox("Fix which?", ["tests", "code", "spec"])
        feedback = st.text_area("What should change?")
        if st.button("Apply fix"):
            b = requests.post(API, json={
                "request": st.session_state.request,
                "spec": last["spec"], "tests": last["tests"], "code": last["code"],
                "test_result": last.get("test_result", {}),
                "fix_target": target, "feedback": feedback,
            }).json()
            st.session_state.history += [{"role": "user", "text": f"fix **{target}**: {feedback}"},
                                         {"role": "assistant", "build": b}]
            st.session_state.last = b
            st.rerun()

# 3. the docked chat input — a fresh build
if req := st.chat_input("What function should I build?"):
    st.session_state.request = req
    with st.spinner("building… (orchestrator → QA → coder → judge)"):
        b = requests.post(API, json={"request": req}).json()
    st.session_state.history += [{"role": "user", "text": req},
                                 {"role": "assistant", "build": b}]
    st.session_state.last = b
    st.rerun()
