import requests
import streamlit as st

API = "http://localhost:8000/build"
st.title("Code Builder")

st.session_state.setdefault("build", None)
st.session_state.setdefault("request", "")

req = st.text_input("What function should I build?", "write a function to reverse a string")
if st.button("Build (fresh)"):
    st.session_state.request = req
    st.session_state.build = requests.post(API, json={"request": req}).json()

b = st.session_state.build
if b:
    st.subheader(f"Status: {b['status']}")
    st.code(b["code"], language="python")
    with st.expander("Tests"): st.code(b["tests"], language="python")
    with st.expander("Spec"):  st.text(b["spec"])

    if b["status"] != "ok":                       # <-- the human-in-the-loop
        st.warning("Stuck - the auto-judge gave up after 3 rounds. You fix it:")
        target   = st.selectbox("Fix which?", ["tests", "code", "spec"])
        feedback = st.text_area("Feedback / correction")
        if st.button("Apply fix"):
            st.session_state.build = requests.post(API, json={
                "request": st.session_state.request,
                "spec": b["spec"], "tests": b["tests"], "code": b["code"],
                "test_result": b.get("test_result", {}),
                "fix_target": target, "feedback": feedback,
            }).json()
            st.rerun()                            # re-render with the new build
