"""Sandbox: run the QA tests against the coder's code and return a verdict.

v1 safety = temp dir + subprocess + timeout. This is NOT a real sandbox - it executes
LLM-written code, so a hostile model could still do harm (no network/fs isolation).
Acceptable for local learning on trivial tasks; harden later (container, no-net, rlimits).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

def run_tests(code: str, tests: str, timeout: int=30) -> dict:
    # write code + tests to temp dir, run pytest
    # return {"passed", "failures"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "solution.py").write_text(code)
        (tmp / "test_solution.py").write_text(tests)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=tmp,    # run inside temp dir -> `from solution import ...` resolves
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "failures": f"timed out after {timeout}s (possible infinite loop)"}
        
        passed = result.returncode == 0 # pytest: 0 = all passed, non-zero = failures / errors
        output = (result.stdout + "\n" + result.stderr).strip()
        return {"passed": passed, "failures": "" if passed else output}