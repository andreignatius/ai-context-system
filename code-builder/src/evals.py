"""A backtest for the builder: run a fixed task suite ONE-SHOT (no human), measure pass-rate.
    python -m src.evals
"""
from src.graph import build_graph

TASKS = [
    # "write is_prime(n: int) -> bool: True if n is prime; n < 2 return False",
    # "write factorial(n: int) -> int: n!; factorial(0) == 1; raise ValueError if n < 0",
    # "write reverse_string(s: str) -> str: return s reversed",
    # "write gcd(a: int, b: int) -> int: the greatest common divisor of a and b",
    # "write celsius_to_fahrenheit(c: float) -> float: F = c * 9/5 + 32",
    "write function to determine if integer is prime number",
    "write function to compute factorial of integer",
    "write function to reverse a string given a string input",
    "write function to compute the greatest common divisor of two integer inputs a and b",
    "write function to convert float value from celsius to fahrenheit",
]

def run_eval():
    app = build_graph()
    results = []
    for task in TASKS:
        out = app.invoke({"request": task})
        results.append({
            "task": task,
            "passed": out["status"] == "ok",
            "attempts": out.get("attempts", 0),
        })
    print("\n===== EVAL REPORT =====")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] attempts={r['attempts']:<2} {r['task'][:55]}")
    passed = sum(r["passed"] for r in results)
    n = len(results)
    print(f"\nPASS RATE: {passed}/{n} = {round(100 * passed / n)}%")

if __name__ == "__main__":
    run_eval()
