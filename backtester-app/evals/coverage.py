"""Dataset coverage + the per-agent eval index (NO LLM). Shows what the shared dataset covers and which eval
grades which agent - the one-screen "are we testing every agent?" view.

Run (from the backtester-app/ root):  python -m evals.coverage
"""
import json
import pathlib
from collections import Counter

DATA = pathlib.Path(__file__).resolve().parent / "dataset" / "requests.json"


def main():
    cases = json.loads(DATA.read_text())
    modes = Counter(c["classify"]["mode"] for c in cases)
    tags = Counter(t for c in cases for t in c["tags"])
    spec_n = sum(1 for c in cases if c.get("spec_must_include"))

    print(f"\nDATASET  evals/dataset/requests.json  -  {len(cases)} cases")
    print(f"  by mode:  {dict(modes)}")
    print(f"  by tag:   {dict(tags)}")
    print(f"  spec-gradeable (have spec_must_include): {spec_n}")

    print("\nPER-AGENT EVALS  (run from the backtester-app/ root)")
    rows = [
        ("classify", "router + params", f"all {len(cases)} cases", "evals.probabilistic.check_classify"),
        ("spec",     "orchestrator faithfulness", f"{spec_n} cases", "evals.probabilistic.eval_spec"),
        ("coder",    "behaviour vs baseline (discriminates by model)", "suite", "evals.probabilistic.eval_suite"),
        ("judge",    "failure -> culprit", "fixed cases", "evals.probabilistic.check_judge"),
    ]
    for agent, what, n, mod in rows:
        print(f"  {agent:9} {what:46} {n:12}  python -m {mod}")
    print("\n  deterministic plumbing checks: python -m evals.deterministic.<check_engine|check_contributions|...>\n")


if __name__ == "__main__":
    main()
