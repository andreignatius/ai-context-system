# from .data import load_prices
# from .runner import run_strategy
# from .agents import write_spec, write_code
from .graph import build_graph

# prices = load_prices("SPY", "2y")
# request = input("Strategy idea? ") or "long when price is above its 50-day moving average, else flat"

# state = {"request": request}
# state.update(write_spec(state))
# state.update(write_code(state))
# result = run_strategy(state["strategy_code"], prices)

# print("\n=== spec ===\n" + state["spec"])
# print("\n=== strategy code ===\n" + state["strategy_code"])
# print("\n=== run result ===")
# print("passed:", result["passed"], "| failures:", result["failures"] or "none")
# print("metrics:", {k: round(v, 3) for k, v in result["metrics"].items()})



app = build_graph()

def run_one():
    request = input("Strategy idea? ") or "long when price is above its 50-day moving average, else flat"
    result = app.invoke({"request": request})
    print("\n=== spec ===\n" + result["spec"])
    print("\n=== strategy code ===\n" + result["strategy_code"])
    print("\n=== result ===")
    print("status:", result["status"])
    rr = result["run_result"]
    print("passed:", rr["passed"], "| failures:", rr["failures"] or "none")
    print("metrics:", {k: round(v, 3) for k, v in rr["metrics"].items()})

if __name__ == "__main__":
    run_one()
