"""Per-agent eval: the SPEC writer (orchestrator). Does the written spec FAITHFULLY carry the request's
key facts? Reads the shared dataset (evals/dataset/requests.json) and grades the spec TEXT against each case's
`spec_must_include` (regexes that MUST appear) and `spec_must_not` (regexes that must NOT). Reports exactly
which facts were DROPPED or wrongly ADDED. Pass-RATE over N runs (the orchestrator is non-deterministic).

This is where SILENT SCOPE-FIT is born (Lessons 028/029/037): a spec that quietly drops a parameter (the
'200', the 'long-only', a ticker) produces a sound-but-wrong backtest downstream. Grading the spec writer
directly is the eval analog of the editable-spec confirm flow.

HONEST LIMIT: keyword-presence is a PROXY for 'captured the intent' - it catches dropped/wrong PARAMETERS (the
real failure mode), NOT deep semantic faithfulness. The semantic check is LLM-as-judge (v2).

Run (from the backtester-app/ root):  python -m evals.probabilistic.eval_spec
"""
import os
import re
import json
import pathlib
from collections import Counter

from src.agents import write_spec

N_RUNS = int(os.getenv("EVAL_N", "3"))
DATA = pathlib.Path(__file__).resolve().parents[1] / "dataset" / "requests.json"


def grade_spec(spec: str, must_include, must_not):
    """Returns (dropped, added): must_include regexes that did NOT match, must_not regexes that DID match."""
    text = spec.lower()
    dropped = [p for p in must_include if not re.search(p.lower(), text)]
    added = [p for p in (must_not or []) if re.search(p.lower(), text)]
    return dropped, added


def main():
    cases = [c for c in json.loads(DATA.read_text()) if c.get("spec_must_include")]
    print(f"\nSPEC eval (orchestrator faithfulness): {len(cases)} cases x {N_RUNS} runs\n")
    summary = {}
    for c in cases:
        verdicts = []
        for i in range(N_RUNS):
            spec = write_spec({"request": c["request"]}).get("spec", "")
            dropped, added = grade_spec(spec, c["spec_must_include"], c.get("spec_must_not"))
            ok = not dropped and not added
            verdicts.append(ok)
            detail = "" if ok else f"  dropped={dropped or '-'} added={added or '-'}"
            print(f"  {c['id']:20} run {i+1}/{N_RUNS}: {'PASS' if ok else 'FAIL'}{detail}")
        passed = sum(verdicts)
        summary[c["id"]] = passed
        print(f"  => {c['id']}: {passed}/{N_RUNS}\n")

    total = sum(summary.values())
    print(f"{'='*60}\nSPEC writer: {total}/{len(cases) * N_RUNS} spec-checks passed")
    for cid, p in summary.items():
        print(f"  {cid:20} {p}/{N_RUNS}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
