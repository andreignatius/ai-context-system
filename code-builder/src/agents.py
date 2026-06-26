"""The three agents, each an ISOLATED-context function.

Key pattern: every agent builds its OWN fresh message list (its own system prompt + the
scoped input). No agent reads a shared conversation - there isn't one. Each returns ONLY
its artifact into BuilderState. That is ISOLATE in code.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from .config import get_llm
from .sandbox import run_tests
from .state import BuildEvent

import re

llm = get_llm()

# ORCHESTRATOR_PROMPT = (
#     "Kindly assume role of software orchestrator. Produce a SPEC that a SEPARATE engineer will "
#     "implement from. Output ONLY these fields, as plain text - nothing else:\n"
#     "1. Function name and signature (with type hints)\n"
#     "2. One-line description of what it does\n"
#     "3. Edge cases to handle - these MUST be consistent with the input range in the request; "
#     "treat ALL out-of-range inputs the same way (e.g. raise ValueError), do not invent "
#     "special return values for them\n" 
#     "4. 2-4 concrete input->output examples\n"
# )

ORCHESTRATOR_PROMPT = (
    "Kindly assume role of software orchestrator. Produce a SPEC with EXACTLY these four "
    "sections and nothing else:\n"
    "1. Function name and signature (with type hints)\n"
    "2. One-line description\n"
    "3. Edge cases\n"
    "4. 2-4 concrete input->output examples\n\n"
    "Notes:\n"
    "a. edge cases must be consistent with the input range\n"
    "b. treat all out-of-range inputs the same way (raise an error)\n"
    "c. do not invent special return values.\n"
    "d. Output ONLY the sections 1-4.\n"
)

# CODER_PROMPT = (
#     "Kindly assume role of Python coder. Implement exactly the spec you are given. "
#     "Please write only the python code that can be written straight to .py file"
# )

# QA_PROMPT = (
#     "Kindly assume role of QA engineer. Given a SPEC document, assume TDD and write "
#     "pytest test suite that will verify implementation against spec, covering examples "
#     "and edge cases. Assume code is importable as `from solution import <function>`. "
#     "Write only the test file python code."
# )

CODER_PROMPT = (
    "You are a Python coder. Implement EXACTLY the spec you are given. "
    "Return the code as a SINGLE ```python fenced block and nothing else - no prose."
)

QA_PROMPT = (
    "You are a QA engineer doing TDD. From the SPEC ALONE, write a pytest suite that "
    "verifies an implementation against the spec - cover the spec's examples and edge "
    "cases, and test ONLY behaviour the spec defines (do not invent requirements). "
    "Include EVERY import your tests use (e.g. `import pytest` if you use pytest.raises), "
    "including `from solution import <function>`. Return the test file as a SINGLE "
    "```python fenced block and nothing else - no notes, no prose."
    "Test only OBSERVABLE BEHAVIOUR (return values, raised exceptions). "
    "Do NOT test the docstring, __name__, __annotations__, or other implementation details."
)

JUDGE_PROMPT = (
    "You are a triage judge for a code-builder. A pytest test failed. Decide WHO is at fault:\n"
    "- code  : the implementation is wrong (the spec + test are reasonable)\n"
    "- tests : the test is wrong - a wrong expected value, an invented requirement the SPEC "
    "does not state, or broken pytest syntax\n"
    "- spec  : the spec is ambiguous or self-contradictory, so code and tests diverged "
    "legitimately\n"
    "Reply EXACTLY two lines and nothing else:\n"
    "CULPRIT: <code|tests|spec>\n"
    "REASON: <one short sentence>"
)

def _extract_code(text: str) -> str:
    """Pull runnable code from an LLM reply: the FIRST ```fenced``` block (ignoring any
    prose around it); if there is no fence, fall back to the whole reply."""
    text = text.strip()
    if "```" in text:
        after = text.split("```", 1)[1]            # content after the first ```
        block = after.split("```", 1)[0]            # up to the closing ```
        lines = block.splitlines()
        if lines and lines[0].strip().isalpha():    # drop a leading "python" language tag
            lines = lines[1:]
        return "\n".join(lines).strip()
    return text

def _strip_think(text: str) -> str:
    # some models (the 32B coder on DeepInfra) emit chain-of-thought in
    # <think>...</think> tags; keep only the answer after it
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def write_spec(state) -> dict:
    """ORCHESTRATOR (isolated context): request -> spec.
    EDIT mode (fix_target=='spec'): given its OWN previous spec + the human feedback, it EDITS
    the spec (keep the good parts) instead of regenerating from scratch."""
    if state.get("fix_target") == "spec" and state.get("spec"):
        task = (
            f"Original request:\n{state['request']}\n\n"
            f"Your PREVIOUS spec:\n{state['spec']}\n\n"
            f"A human reviewer says it needs changing:\n{state.get('feedback', '')}\n\n"
            "Edit the spec to apply this; keep the parts that were already correct. "
            "Return the full corrected spec in the same four-section format."
        )
        print("[orchestrator] editing spec (human-directed)")
    else:
        task = state["request"]
        if state.get("feedback"):
            task += f"\n\nIMPORTANT human feedback to apply:\n{state['feedback']}"
        print("[orchestrator] wrote spec")

    messages = [SystemMessage(content=ORCHESTRATOR_PROMPT), HumanMessage(content=task)]
    spec = _strip_think(llm.invoke(messages).content)
    return {"spec": spec, "ledger": [BuildEvent("orchestrator", "spec", spec)]}


def write_code(state) -> dict:
    """CODER (isolated context): spec -> code.
    On a retry it also gets its previous code + the test failures, so it fixes (not blindly
    rewrites). The round counter now lives in the JUDGE, not here."""
    prev = state.get("test_result")
    if prev and not prev.get("passed", True):
        # retry: feed back the last attempt + why it failed
        task = (
            f"SPEC:\n{state['spec']}\n\n"
            f"Your previous code:\n{state['code']}\n\n"
            f"It FAILED these tests:\n{prev['failures']}\n\n"
            "Kindly fix the code so all the tests pass. Return the corrected function."
        )
    else:
        # first attempt: just the spec
        task = f"SPEC:\n{state['spec']}"

    # judge/human-directed code fix: surface the reviewer's guidance to the coder
    if state.get("fix_target") == "code" and state.get("feedback"):
        task += (f"\n\nA reviewer says, specifically about the CODE:\n"
                 f"{state['feedback']}\nApply this.")

    messages = [
        SystemMessage(content=CODER_PROMPT),
        HumanMessage(content=task),
    ]
    code = _extract_code(_strip_think(llm.invoke(messages).content))
    print("[coder] wrote code")
    return {"code": code, "ledger": [BuildEvent("coder", "code", code)]}


def write_tests(state) -> dict:
    """QA (isolated context): spec -> pytest tests.
    EDIT mode (fix_target=='tests'): given its OWN previous tests + the human feedback, the
    QA fixes the suite (keep the good tests, repair the bad) instead of regenerating blind."""
    if state.get("fix_target") == "tests" and state.get("tests"):
        task = (
            f"SPEC:\n{state['spec']}\n\n"
            f"Your PREVIOUS test suite:\n{state['tests']}\n\n"
            f"A human reviewer says it is wrong:\n{state.get('feedback', '')}\n\n"
            "Fix ONLY what the feedback calls out; keep the tests that were already correct. "
            "Return the full corrected test file."
        )
        print("[qa] editing tests (human-directed)")
    else:
        task = f"SPEC:\n{state['spec']}"
        if state.get("feedback"):
            task += f"\n\nIMPORTANT human feedback about the TESTS - apply it:\n{state['feedback']}"
        print("[qa] wrote tests")
    messages = [SystemMessage(content=QA_PROMPT), HumanMessage(content=task)]
    tests = _extract_code(_strip_think(llm.invoke(messages).content))
    return {"tests": tests, "ledger": [BuildEvent("qa", "tests", tests)]}


def _parse_verdict(reply: str) -> tuple:
    culprit, reason = "code", "(no reason parsed)"          # default to code = the old behaviour
    for line in reply.splitlines():
        low = line.strip().lower()
        if low.startswith("culprit:"):
            val = low.split(":", 1)[1]
            culprit = "tests" if "test" in val else "spec" if "spec" in val else "code"
        elif low.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
    return culprit, reason

def _is_load_error(failures: str) -> bool:
    f = failures.lower()
    return "error collecting" in f or "errors during collection" in f or "==== errors ====" in f

def judge(state) -> dict:
    """Decide which agent is at fault and route a fix to it (auto version of the human's
    fix_target). Decidable tier: a non-loading test file -> the QA. Undecidable tier: an LLM
    weighs code vs tests vs spec on an assertion failure."""
    attempts = state.get("attempts", 0) + 1
    failures = state["test_result"]["failures"]

    if _is_load_error(failures):
        print(f"[judge] round {attempts}: test file did not LOAD -> tests (mechanical)")
        return {"fix_target": "tests", "attempts": attempts,
                "feedback": f"The test file failed to load:\n{failures}"}
    
    task = (f"SPEC:\n{state['spec']}\n\nTESTS:\n{state['tests']}\n\n"
            f"CODE:\n{state['code']}\n\nFAILURE:\n{failures}")
    reply = _strip_think(llm.invoke([SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=task)]).content)
    culprit, reason = _parse_verdict(reply)
    print(f"[judge] round {attempts}: -> {culprit} ({reason})")
    return {"fix_target": culprit, "feedback": reason, "attempts": attempts}


if __name__ == "__main__":
    request = "write a function is_palindrome(s) that checks if a string is a palindrome"

    spec = write_spec({"request": request})["spec"]
    code = write_code({"spec": spec})["code"]
    tests = write_tests({"spec": spec})["tests"]
    print("=== spec ===\n", spec)
    print("\n=== code ===\n", code)
    print("\n=== tests ===\n" + tests)

    result = run_tests(code, tests)
    print("\n=== test result ===")
    print("passed:", result["passed"])
    print(result["failures"][:1500])


