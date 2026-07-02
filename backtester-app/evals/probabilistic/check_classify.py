"""Per-agent eval: the CLASSIFIER (routing + param extraction). Reads the SHARED dataset
(evals/dataset/requests.json) and checks classify's `mode` for every case, plus the extracted params
(start-presence / amount / named ticker) for the backtest modes. Pass-rate over N runs (EVAL_N, default 1).
Needs the LLM.

The test cases live in the dataset (not inline here) so classify + spec are graded from ONE source of truth -
add a case once and it tests both agents.

Run (from the backtester-app/ root):  python -m evals.probabilistic.check_classify
"""
import os
import json
import pathlib

from src.agents import classify

N_RUNS = int(os.getenv("EVAL_N", "1"))
DATA = pathlib.Path(__file__).resolve().parents[1] / "dataset" / "requests.json"


def check(out, exp):
    """Grade one classify output vs the dataset's expected `classify` block. Returns (ok, detail)."""
    if out.get("mode") != exp["mode"]:
        return False, f"mode={out.get('mode')} (want {exp['mode']})"
    if exp["mode"] in ("help", "out_of_scope"):
        return True, "mode ok"                       # these short-circuit: no params to check
    bad = []
    if (out.get("start_date") is not None) != (exp.get("start") is not None):
        bad.append("start")
    if out.get("amount") != exp.get("amount"):
        bad.append("amount")
    if exp.get("ticker") and (out.get("ticker") or "").upper() != exp["ticker"].upper():
        bad.append("ticker")
    return (not bad), ("ok" if not bad else "wrong: " + ", ".join(bad))


def main():
    cases = json.loads(DATA.read_text())
    print(f"\nCLASSIFY eval: {len(cases)} cases x {N_RUNS} run(s)\n")
    correct = total = 0
    for c in cases:
        for _ in range(N_RUNS):
            ok, detail = check(classify({"request": c["request"]}), c["classify"])
            correct += ok
            total += 1
            print(f"  [{'ok  ' if ok else 'MISS'}] {c['id']:20} {detail}")
    print(f"\n{correct}/{total} correct (mode + params)")


if __name__ == "__main__":
    main()
