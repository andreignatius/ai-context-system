"""The three agents, each an ISOLATED-context function.

Key pattern: every agent builds its OWN fresh message list (its own system prompt + the
scoped input). No agent reads a shared conversation - there isn't one. Each returns ONLY
its artifact into BuilderState. That is ISOLATE in code.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from .config import get_llm
from .sandbox import run_tests
from .state import BuildEvent

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


def write_spec(state) -> dict:
    # orchestrator (isolated context): request -> spec
    messages = [
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        HumanMessage(content=state["request"]),
    ]
    spec = llm.invoke(messages).content
    print("[orchestrator] wrote spec")
    return {"spec": spec,
            "ledger": [BuildEvent("orchestrator", "spec", spec)]}

def write_code(state) -> dict:
    """CODER (isolated context): spec -> code.
    on a retry, the coder also gets its previous code + test failures,
    so it can fix (not blindly rewrite)
    increments the attempts counter each call
    """
    attempts = state.get("attempts", 0) + 1

    prev = state.get("test_result")
    if prev and not prev.get("passed", True):
        # retry: feedback the last attempt + why it failed
        task = (
            f"SPEC:\n{state['spec']}\n\n"
            f"Your previous code:\n{state['code']}\n\n"
            f"It FAILED these tests:\n{prev['failures']}\n\n"
            "Kindly fix the code so all the tests pass. Return the corrected function."
        )
    else:
        # first attempt: just the spec
        task = f"SPEC:\n{state['spec']}"

    messages = [
        SystemMessage(content=CODER_PROMPT),
        HumanMessage(content=task),
    ]
    code = _extract_code(llm.invoke(messages).content)
    print(f"[coder] wrote code (attempt {attempts})")
    return {"code": code, "attempts": attempts,
            "ledger": [BuildEvent("coder", "code", code)]}

def write_tests(state) -> dict:
    """QA (isolated context): spec -> pytest tests."""
    # task = f"SPEC:\n{state['spec']}\n\nCODE:\n{state['code']}"
    task = f"SPEC:\n{state['spec']}"
    messages = [
        SystemMessage(content=QA_PROMPT),
        HumanMessage(content=task),
    ]
    tests = _extract_code(llm.invoke(messages).content)
    print("[qa] wrote tests")
    return {"tests": tests,
            "ledger": [BuildEvent("qa", "tests", tests)]}



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


