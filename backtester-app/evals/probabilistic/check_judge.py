from src.agents import judge

cases = ["load error: name 'strategy' is not defined",
         "runtime error: KeyError 'Close'",
         "positions out of range [-1, 1]",
         "strategy is not deterministic"]

for failure in cases:
    state = {"run_result": {"passed": False, "failures": failure}, "attempts": 0,
             "request": "x", "spec": "x", "strategy_code": "x"}
    out = judge(state)
    print(f"{failure[:32]:34s} -> {out['fix_target']}  (attempt {out['attempts']})")
    assert out["fix_target"] == "code", f"expected code for: {failure}"

print("\nPASS: implementation errors route to code (mechanical, no LLM).")
