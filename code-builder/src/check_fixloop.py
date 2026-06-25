# check_fixloop.py - prove the coder's RETRY branch converges a real, fixable bug
from src.agents import write_code
from src.sandbox import run_tests

spec = (
    "1. is_prime(n: int) -> bool\n"
    "2. Returns True if n is prime, else False.\n"
    "3. Edge cases: n < 2 returns False (0, 1, negatives are not prime).\n"
    "4. Examples: is_prime(2) -> True; is_prime(1) -> False; is_prime(9) -> False\n"
)

# a GENUINE, fixable bug: returns True for n < 2 (range(2,n) is empty)
buggy_code = (
    "def is_prime(n: int) -> bool:\n"
    "    for i in range(2, n):\n"
    "        if n % i == 0:\n"
    "            return False\n"
    "    return True\n"
)

tests = (
    "from solution import is_prime\n\n"
    "def test_primes():\n"
    "    assert is_prime(2) is True\n"
    "    assert is_prime(3) is True\n"
    "    assert is_prime(9) is False\n\n"
    "def test_edge_cases():\n"
    "    assert is_prime(1) is False\n"
    "    assert is_prime(0) is False\n"
    "    assert is_prime(-5) is False\n"
)

before = run_tests(buggy_code, tests)
print("planted code passes?", before["passed"])        # expect False (sanity: bug is real)

# drive the RETRY branch: spec + previous (buggy) code + the REAL failures
out = write_code({"spec": spec, "code": buggy_code,
                  "test_result": before, "attempts": 1})
fixed = out["code"]
print("\n--- coder's fix ---\n" + fixed)

after = run_tests(fixed, tests)
print("\nfixed code passes?", after["passed"])          # expect True == convergence proven
if not after["passed"]:
    print(after["failures"][:800])
