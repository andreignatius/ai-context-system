from src.agents import judge

# CASE 1 - a real CODE bug (spec + test are fine, code is wrong)
code_bug = {
    "spec": "factorial(n): returns n!; factorial(0) == 1",
    "tests": "from solution import factorial\ndef test_zero():\n    assert factorial(0) == 1",
    "code": "def factorial(n):\n    return 0",                       # wrong
    "test_result": {"passed": False, "failures": "assert 0 == 1\n where 0 = factorial(0)"},
}

# CASE 2 - a TEST bug (the QA invented a requirement the spec never stated)
test_bug = {
    "spec": "gcd(a, b): returns the greatest common divisor of a and b",
    "tests": "from solution import gcd\ndef test_neg():\n    with pytest.raises(ValueError):\n        gcd(-4, 6)",
    "code": "import math\ndef gcd(a, b):\n    return math.gcd(a, b)",  # correct
    "test_result": {"passed": False, "failures": "DID NOT RAISE ValueError"},
}

for name, st in [("code_bug", code_bug), ("test_bug", test_bug)]:
    v = judge(st)
    print(f"{name:10} -> fix_target={v['fix_target']!r:8} | {v['feedback']}")
