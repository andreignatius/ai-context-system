"""Backtester agents: orchestrator (strategy spec) + coder (strategy code).
Same ISOLATE pattern as the code-builder - each builds its own message list."""
import re
from langchain_core.messages import SystemMessage, HumanMessage
from .config import get_llm
from .state import BuildEvent

llm = get_llm()

def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _extract_code(text):
    text = text.strip()
    if "```" in text:
        block = text.split("```", 1)[1].split("```", 1)[0]
        lines = block.splitlines()
        if lines and lines[0].strip().isalpha():   # drop a leading "python" tag
            lines = lines[1:]
        return "\n".join(lines).strip()
    return text


# ONE PROMPT, ONE JOB: mode classification kept separate from param extraction (combining the two
# degraded mode accuracy - the critical routing decision). See Lesson 029.
MODE_PROMPT = (
    "You route a quant request to the right engine. Decide ONE word:\n"
    "- position : the user wants a trading STRATEGY's performance (return / Sharpe / drawdown).\n"
    "- contribution : the user asks about MONEY DEPOSITED over time - dollars, 'how much money', "
    "putting in $X on a schedule or on a signal, dollar-cost averaging (DCA), 'total value'.\n"
    "RULE: if the request mentions depositing dollars OR 'how much money', it is CONTRIBUTION "
    "regardless of the trading signal it describes.\n"
    "Reply with EXACTLY one word: position OR contribution. Nothing else."
)

PARAMS_PROMPT = (
    "Extract two fields from a quant request. Reply EXACTLY two lines and nothing else:\n"
    "START: the start date as YYYY-MM-DD if the user gives one ('since 2021' -> 2021-01-01), else none\n"
    "AMOUNT: the dollars-per-deposit as a plain number if given ('$1k' -> 1000), else none\n"
)

def classify(state):
    """M8 tool-selection + param extraction. TWO calls (one prompt, one job): mode classification
    stays separate from param extraction so the critical routing decision is not degraded (Lesson 029)."""
    req = state["request"]
    mode_reply = _strip_think(llm.invoke([SystemMessage(content=MODE_PROMPT),
                                          HumanMessage(content=req)]).content)
    mode = "contribution" if "contribution" in mode_reply.lower() else "position"

    params_reply = _strip_think(llm.invoke([SystemMessage(content=PARAMS_PROMPT),
                                            HumanMessage(content=req)]).content)
    start, amount = None, None
    for line in params_reply.splitlines():
        low = line.strip().lower()
        if low.startswith("start:"):
            val = line.split(":", 1)[1].strip()
            start = val if ("-" in val and val.lower() != "none") else None   # a YYYY-MM-DD date
        elif low.startswith("amount:"):
            val = line.split(":", 1)[1].strip().replace("$", "").replace(",", "")
            try:
                amount = float(val)
            except ValueError:
                amount = None
    print(f"[classify] mode={mode} start={start} amount={amount}")
    out = {"mode": mode}
    if start:
        out["start_date"] = start
    if amount:
        out["amount"] = amount
    return out

ORCHESTRATOR_PROMPT = (
    "You are a quant strategist. Given a trading idea, produce a SPEC for a Python function "
    "strategy(history) that returns a target position. Output EXACTLY these sections, nothing else:\n"
    "1. Strategy name + one-line description\n"
    "2. Signal logic (indicator + rule)\n"
    "3. Position mapping (long=1.0 / flat=0.0 / short=-1.0)\n"
    "4. Parameters (e.g. window lengths)\n"
    "PRESERVE the user's EXACT quantitative definitions: 'N-day' means N consecutive days (NOT N%); "
    "do not substitute or reinterpret the user's numbers or units.\n"
)

CODER_PROMPT = (
    "You are a Python quant coder. Implement the spec as a function `strategy(history)`:\n"
    "- `history` is a pandas Series of prices up to AND INCLUDING the current bar.\n"
    "- return a float target position in [-1, 1] to hold from the NEXT bar.\n"
    "- CRITICAL: use ONLY `history` - NEVER index beyond it or peek at future data.\n"
    "- handle the warm-up period (return 0.0 until there are enough bars).\n"
    "Return ONLY a single ```python fenced block defining strategy(history) - no prose.\n"
    "- `history` is a pandas Series: use history.iloc[-1] (and .iloc[-2]) for the latest values; "
    "do NOT use history[-1] - that is LABEL indexing and will KeyError.\n"
    "- POINT-IN-TIME: decide for the CURRENT (last) bar using ONLY the MOST RECENT bars. Do NOT loop or "
    "scan over all of history (that makes the signal fire on almost every bar). For 'N consecutive down "
    "days', check the LAST N daily changes: (history.diff().iloc[-N:] < 0).all().\n"
)

JUDGE_PROMPT = (
    "You are a triage judge for a quant strategy builder. A generated strategy(history) function "
    "failed its soundness checks. Decide WHO is at fault:\n"
    "- code : the strategy IMPLEMENTATION is buggy (the idea IS a valid position strategy)\n"
    "- spec : the request/spec is NOT a position strategy at all (e.g. a comparison or research "
    "question), so no sound strategy(history) can satisfy it\n"
    "Reply EXACTLY two lines and nothing else:\n"
    "CULPRIT: <code|spec>\n"
    "REASON: <one short sentence>"
)

def write_spec(state):
    if state.get("fix_target") == "spec" and state.get("spec"):
        task = (f"Original request:\n{state['request']}\n\n"
                f"Your PREVIOUS spec:\n{state['spec']}\n\n"
                f"A reviewer says it needs changing:\n{state.get('feedback', '')}\n\n"
                "Edit the spec; keep the good parts. Return the full corrected 4-section spec.")
        print("[orchestrator] editing spec")
    else:
        task = state["request"]
        print("[orchestrator] wrote strategy spec")
    msgs = [SystemMessage(content=ORCHESTRATOR_PROMPT), HumanMessage(content=task)]
    spec = _strip_think(llm.invoke(msgs).content)
    return {"spec": spec, "ledger": [BuildEvent("orchestrator", "spec", spec)]}

def write_code(state):
    if state.get("fix_target") == "code" and state.get("strategy_code"):
        task = (f"SPEC:\n{state['spec']}\n\n"
                f"Your PREVIOUS strategy code:\n{state['strategy_code']}\n\n"
                f"It FAILED soundness checks:\n{state.get('feedback', '')}\n\n"
                "Fix the strategy so it passes. Return ONLY the corrected strategy(history) "
                "in a single ```python block.")
        print("[coder] fixing strategy code")
    else:
        task = f"SPEC:\n{state['spec']}"
        print("[coder] wrote strategy code")
    msgs = [SystemMessage(content=CODER_PROMPT), HumanMessage(content=task)]
    code = _extract_code(_strip_think(llm.invoke(msgs).content))
    return {"strategy_code": code, "ledger": [BuildEvent("coder", "strategy_code", code)]}


def _parse_verdict(reply):
    culprit, reason = "code", "(no reason parsed)"      # default to code
    for line in reply.splitlines():
        low = line.strip().lower()
        if low.startswith("culprit:"):
            culprit = "spec" if "spec" in low.split(":", 1)[1] else "code"
        elif low.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
    return culprit, reason

def judge(state):
    """Route a soundness failure to the agent at fault. Decidable tier: a clear implementation
    error -> code (trace the origin). Undecidable tier: an LLM weighs code vs spec."""
    attempts = state.get("attempts", 0) + 1
    failures = state["run_result"]["failures"]
    f = failures.lower()
    if any(k in f for k in ["load error", "runtime error", "out of range",
                            "non-finite", "not deterministic"]):
        print(f"[judge] round {attempts}: implementation error -> code (mechanical)")
        return {"fix_target": "code", "attempts": attempts, "feedback": failures}
    # inert / ambiguous -> LLM decides code vs spec
    task = (f"REQUEST:\n{state['request']}\n\nSPEC:\n{state['spec']}\n\n"
            f"STRATEGY CODE:\n{state['strategy_code']}\n\nFAILURE:\n{failures}")
    reply = _strip_think(llm.invoke([SystemMessage(content=JUDGE_PROMPT),
                                     HumanMessage(content=task)]).content)
    culprit, reason = _parse_verdict(reply)
    print(f"[judge] round {attempts}: -> {culprit} ({reason})")
    return {"fix_target": culprit, "feedback": reason, "attempts": attempts}
