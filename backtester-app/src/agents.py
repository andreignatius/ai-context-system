"""Backtester agents: orchestrator (strategy spec) + coder (strategy code).
Same ISOLATE pattern as the code-builder - each builds its own message list."""
import re
from langchain_core.messages import SystemMessage, HumanMessage
from .config import get_llm
from .state import BuildEvent

_PAIRS_WORDS = ("pairs", "spread", "relative value", "market neutral", "cointegrat", "ratio")
_NOT_TICKERS = {"RSI","SMA","EMA","MACD","DCA","ETF","USD","AND","THE","SD","ATR","ADX","VWAP","OBV"}

# ONE PROMPT, ONE JOB: mode classification kept separate from param extraction (combining the two
# degraded mode accuracy - the critical routing decision). See Lesson 029.
MODE_PROMPT = (
    "You route a request to the right engine. Reply EXACTLY one word: position OR contribution OR help.\n"
    "- position : the user wants a trading STRATEGY's performance (return / Sharpe / drawdown), EVEN IF they "
    "mention STARTING CAPITAL in dollars ('start with $10k', 'trade 100% of equity each signal').\n"
    "- contribution : money DEPOSITED REPEATEDLY over time - DCA, '$X every week/month', 'deposit $X on each "
    "dip', comparing two DCA schedules, 'how much money would I have'.\n"
    "- help : the message is NOT a backtest request - a greeting, a question ABOUT the app, 'what can you do', "
    "or anything off-topic.\n"
    "KEY TEST: a dollar amount tied to a CADENCE or a repeated event ('$1k monthly', '$250 weekly', '$1k on "
    "each dip') = CONTRIBUTION. A one-time STARTING CAPITAL ('start with $X', 'trade equity') = POSITION. A "
    "message that describes NO strategy or comparison at all = HELP.\n"
    "Examples:\n"
    "  'what can you do?'                                   -> help\n"
    "  'hi' / 'how does this work?'                         -> help\n"
    "  'long SPY when RSI<30, start with $10k'              -> position\n"
    "  '50/200 SMA crossover on SPY'                        -> position\n"
    "  'DCA $1k monthly into SPY'                           -> contribution\n"
    "  'compare GOOG monthly vs SPY monthly, $1k each'      -> contribution\n"
    "  'buy the dip: $1000 on each drawdown vs $1k monthly' -> contribution\n"
)

PARAMS_PROMPT = (
    "Extract three fields from a quant request. Reply EXACTLY three lines and nothing else:\n"
    "TICKER: the stock/ETF/crypto symbol in UPPERCASE if the user names ONE asset (e.g. IWM, QQQ, BTC-USD), "
    "else none\n"
    "START: the start date as YYYY-MM-DD if the user gives one ('since 2021' -> 2021-01-01), else none\n"
    "AMOUNT: the dollars-per-deposit as a plain number if given ('$1k' -> 1000), else none\n"
)

_CADENCES = ("signal", "weekly", "monthly")

LEGS_PROMPT = (
    "A user is comparing TWO money-deposit schedules. Extract both. Reply EXACTLY two lines, nothing else:\n"
    "LEG1: <ticker> <cadence> <amount>\n"
    "LEG2: <ticker> <cadence> <amount>\n"
    "ticker is the stock/ETF/crypto symbol in UPPERCASE (GOOG, SPY, AAPL, BTC-USD). If the user names only "
    "ONE asset, use it for BOTH legs; if NONE is named, write SPY.\n"
    "cadence is ONE of: signal (deposit on a trading signal, e.g. 'buy the dip'), weekly, monthly.\n"
    "amount is the dollars per deposit, a plain number.\n"
    "Examples:\n"
    "  'GOOG monthly vs SPY monthly, $1k each'   -> LEG1: GOOG monthly 1000 / LEG2: SPY monthly 1000\n"
    "  'weekly $250 vs monthly $1000 into QQQ'   -> LEG1: QQQ weekly 250 / LEG2: QQQ monthly 1000\n"
    "  'buy the dip $1000 vs DCA $1000 monthly'  -> LEG1: SPY signal 1000 / LEG2: SPY monthly 1000\n"
)


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

CODER_PAIRS_PROMPT = (
    "You are a Python quant coder. Implement the spec as a PAIRS function "
    "`strategy_pair(history_a, history_b)`:\n"
    "- history_a, history_b are pandas Series of the two assets' prices up to AND INCLUDING the current bar.\n"
    "- return a float SPREAD position in [-1, 1]: +1 = long the spread (long A / short B), -1 = short, 0 = flat.\n"
    "- CRITICAL: use ONLY history_a / history_b - NEVER index beyond them or peek at the future.\n"
    "- handle warm-up (return 0.0 until enough bars).\n"
    "- SPREAD: ALWAYS use the LOG-spread `np.log(history_a) - np.log(history_b)` (scale-invariant, the "
    "stat-arb standard) - even if the spec writes 'A - B' or 'spread = A - B', treat that as SHORTHAND for "
    "the log-spread (raw price differences are scale-broken). Use a price RATIO (A / B) or a hedge-ratio "
    "regression ONLY if the spec says 'ratio' or 'hedge ratio' in words.\n"
    "- POINT-IN-TIME: compute any rolling stat (z-score, hedge ratio) over the MOST RECENT bars only "
    "(e.g. .iloc[-lookback:]), NOT the full history. Use .iloc[-1] for the latest value, not [-1].\n"
    "Return ONLY a single ```python fenced block defining strategy_pair(history_a, history_b) - no prose.\n"
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

def is_multi_asset_position(request: str, mode: str) -> bool:
    if mode != "position":
        return False
    r = request.lower()
    if any(w in r for w in _PAIRS_WORDS):                       # "pairs", "spread", ...
        return True
    if re.search(r"\blong\b.+\bshort\b", r):                    # "long X ... short Y"
        return True
    toks = set(re.findall(r"\b[A-Z]{2,5}(?:-USD)?\b", request)) - _NOT_TICKERS
    return len(toks) >= 2                                       # 2+ distinct tickers named

def extract_pair_tickers(request: str) -> list:
    """Distinct ticker-like tokens in order (for pairs: the first two = A, B)."""
    seen = []
    for t in re.findall(r"\b[A-Z]{2,5}(?:-USD)?\b", request):
        if t not in _NOT_TICKERS and t not in seen:
            seen.append(t)
    return seen


PAIR_TICKERS_PROMPT = """You map a pairs-trading request to the TWO Yahoo Finance ticker symbols it names.
Map company NAMES to symbols: Exxon Mobil -> XOM, Shell -> SHEL, Apple -> AAPL, Coca-Cola -> KO.
If an asset is already a symbol, keep it as-is.
Reply with EXACTLY two uppercase symbols separated by a comma (e.g. "XOM, SHEL"), nothing else.
If you cannot identify two distinct assets, reply "NONE"."""

def resolve_pair_tickers(request: str) -> list:
    """The two pair tickers. Cheap regex first (already-symbols); fall back to the LLM to map NAMES -> symbols."""
    syms = extract_pair_tickers(request)
    if len(syms) >= 2:
        return syms[:2]
    reply = _strip_think(llm.invoke([SystemMessage(content=PAIR_TICKERS_PROMPT),
                                     HumanMessage(content=request)]).content).strip().upper()
    if "NONE" in reply:
        return []
    skip = _NOT_TICKERS | {"VS", "OR"}                  # drop connectives the model may leak ("XOM and SHEL")
    toks = [t for t in re.split(r"[^A-Z.\-]+", reply) if any(c.isalpha() for c in t) and t not in skip]
    return toks[:2] if len(toks) >= 2 else []


TICKER_PROMPT = """Return the Yahoo Finance ticker symbol for the asset named below.
Examples: Exxon Mobil -> XOM, Shell -> SHEL, Apple -> AAPL, S&P 500 -> SPY, Bitcoin -> BTC-USD.
Reply with ONLY the uppercase symbol, nothing else. If unsure, reply NONE."""

def resolve_ticker(name: str) -> str:
    """A single asset NAME -> its Yahoo Finance symbol (LLM); '' if unsure. Caller MUST validate it loads
    (an LLM lookup can hallucinate a plausible-but-wrong symbol)."""
    name = (name or "").strip()
    if not name:
        return ""
    reply = _strip_think(llm.invoke([SystemMessage(content=TICKER_PROMPT),
                                     HumanMessage(content=name)]).content).strip().upper()
    if "NONE" in reply:
        return ""
    m = re.findall(r"[A-Z][A-Z.\-]{0,7}", reply)
    return m[0] if m else ""

def classify(state):
    """M8 tool-selection + param extraction. TWO calls (one prompt, one job): mode classification
    stays separate from param extraction so the critical routing decision is not degraded (Lesson 029)."""
    req = state["request"]
    mode_reply = _strip_think(llm.invoke([SystemMessage(content=MODE_PROMPT),
                                          HumanMessage(content=req)]).content)
    ml = mode_reply.lower()
    if "help" in ml:                      # a META / non-backtest message -> no params, no draft (UI shows welcome)
        print("[classify] mode=help")
        return {"mode": "help"}
    mode = "contribution" if "contribution" in ml else "position"

    params_reply = _strip_think(llm.invoke([SystemMessage(content=PARAMS_PROMPT),
                                            HumanMessage(content=req)]).content)
    start, amount, ticker = None, None, None
    for line in params_reply.splitlines():
        low = line.strip().lower()
        if low.startswith("ticker:"):
            val = line.split(":", 1)[1].strip().upper().strip(".,")
            ticker = val if val and val.lower() != "none" else None
        elif low.startswith("start:"):
            val = line.split(":", 1)[1].strip()
            start = val if ("-" in val and val.lower() != "none") else None   # a YYYY-MM-DD date
        elif low.startswith("amount:"):
            val = line.split(":", 1)[1].strip().replace("$", "").replace(",", "")
            try:
                amount = float(val)
            except ValueError:
                amount = None
    print(f"[classify] mode={mode} ticker={ticker} start={start} amount={amount}")
    out = {"mode": mode}
    if ticker:
        out["ticker"] = ticker
    if start:
        out["start_date"] = start
    if amount:
        out["amount"] = amount
    return out


def _leg_label(cadence, amount, ticker=None):
    nice = {"signal": "Signal", "weekly": "Weekly", "monthly": "Monthly"}.get(cadence, str(cadence).title())
    pre = f"{ticker} " if ticker else ""
    return f"{pre}{nice} ${amount:,.0f}"


def _parse_leg(text):
    """Pull (ticker, cadence, amount) from a LEG line, order-independent: a cadence is a known word, a number
    is the amount, the leftover alpha token is the ticker (may be absent -> None, the UI fills the default)."""
    ticker, cadence, amount = None, None, None
    for p in text.strip().split():
        low = p.lower()
        if low in _CADENCES and cadence is None:
            cadence = low
            continue
        try:
            amount = float(p.replace("$", "").replace(",", ""))
            continue
        except ValueError:
            pass
        if ticker is None and any(c.isalpha() for c in p):
            ticker = p.upper().strip(".,")
    return ticker, cadence, amount


def extract_legs(state):
    """UI-ONLY helper (NOT a graph node): best-effort parse of the TWO schedules being compared (ticker +
    cadence + amount), to PREFILL the editable leg controls in the confirm panel. The user confirms/edits
    before running, so a mis-parse is harmless. The CLI/eval path never calls this -> legacy stays intact."""
    reply = _strip_think(llm.invoke([SystemMessage(content=LEGS_PROMPT),
                                     HumanMessage(content=state["request"])]).content)
    legs = []
    for line in reply.splitlines():
        if line.strip().lower()[:4] in ("leg1", "leg2"):
            ticker, cadence, amount = _parse_leg(line.split(":", 1)[1])
            if cadence:
                amt = amount or 1000.0
                leg = {"cadence": cadence, "amount": amt, "label": _leg_label(cadence, amt, ticker)}
                if ticker:
                    leg["ticker"] = ticker
                legs.append(leg)
    print(f"[extract_legs] {legs}")
    return {"legs": legs}


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
    pairs = state.get("mode") == "pairs"
    prompt = CODER_PAIRS_PROMPT if pairs else CODER_PROMPT
    sig = "strategy_pair(history_a, history_b)" if pairs else "strategy(history)"
    if state.get("fix_target") == "code" and state.get("strategy_code"):
        task = (f"SPEC:\n{state['spec']}\n\n"
                f"Your PREVIOUS strategy code:\n{state['strategy_code']}\n\n"
                f"It FAILED soundness checks:\n{state.get('feedback', '')}\n\n"
                f"Fix the strategy so it passes. Return ONLY the corrected {sig} in a single ```python block.")
        print("[coder] fixing strategy code")
    else:
        task = f"SPEC:\n{state['spec']}"
        print(f"[coder] wrote {'pairs ' if pairs else ''}strategy code")
    msgs = [SystemMessage(content=prompt), HumanMessage(content=task)]
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
