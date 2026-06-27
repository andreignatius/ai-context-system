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
    print("status:", result["status"], "| mode:", result.get("mode"))
    if result.get("mode") == "contribution" and result.get("contribution_result"):
        cr = result["contribution_result"]
        s, d = cr["signal"], cr["dca"]
        print(f"buy-the-signal: {s['n']:3d} deposits  ${s['invested']:>10,.0f} -> ${s['final']:>11,.0f}  "
              f"({s['final'] / s['invested']:.2f}x)")
        print(f"monthly DCA   : {d['n']:3d} deposits  ${d['invested']:>10,.0f} -> ${d['final']:>11,.0f}  "
              f"({d['final'] / d['invested']:.2f}x)")
    else:
        rr = result["run_result"]
        print("passed:", rr["passed"], "| failures:", rr["failures"] or "none")
        print("metrics:", {k: round(v, 3) for k, v in rr["metrics"].items()})

if __name__ == "__main__":
    run_one()
