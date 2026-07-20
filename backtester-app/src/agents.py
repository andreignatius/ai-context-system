"""Backtester agents: orchestrator (strategy spec) + coder (strategy code).
Same ISOLATE pattern as the code-builder - each builds its own message list."""
import re
from langchain_core.messages import SystemMessage, HumanMessage
from .config import get_llm
from .state import BuildEvent

_PAIRS_WORDS = ("pairs", "spread", "relative value", "market neutral", "cointegrat", "ratio")
_NOT_TICKERS = {"RSI","SMA","EMA","MACD","DCA","ETF","USD","AND","THE","SD","ATR","ADX","VWAP","OBV",
                # common all-caps words in strategy prose that are NOT tickers (avoid false "2-ticker" pairs)
                "PRIOR","NOT","LOG","LONG","SHORT","FLAT","STD","MEAN","OU","DAY","DAYS","HIGH","BUY","SELL"}

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
    "A date range like 'since 2019' or 'last 5 years' is the BACKTEST PERIOD, NOT a rolling-window length. "
    "For a z-score / rolling-stat rule (spreads, bands, mean-reversion) with NO lookback given, default the "
    "rolling window to ~20-60 days - NEVER the backtest length or a large value like 1000 (that makes the "
    "band so wide it never triggers -> the strategy is inert).\n"
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
    "- For an 'N-day return' (the return over the last N days), use history.pct_change(N).iloc[-1] "
    "(= price_today / price_N_bars_ago - 1). Do NOT write history.iloc[-N] - that is only N-1 bars back "
    "(off by one); N bars back is history.iloc[-(N+1)].\n"
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
    if any(w in r for w in _PAIRS_WORDS):                       # "pairs", "spread", "relative value", ...
        return True
    if re.search(r"\b(vs|versus)\b", r):                        # "X vs Y" - an explicit pairing connective
        return True
    # else: 2+ distinct ticker-like tokens (catches "XLF XLI" without a keyword). NOTE: dropped the old
    # "long ... short" heuristic - a SINGLE-asset long/short (e.g. RSI oversold/overbought) is NOT a pair.
    toks = set(re.findall(r"\b[A-Z]{2,5}(?:-USD)?\b", request)) - _NOT_TICKERS
    return len(toks) >= 2

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


def resolve_symbol(raw, is_ok):
    """First-class ticker resolution: the LLM PROPOSES a symbol, the FETCH VALIDATES it (never trust the
    model's existence claim). `is_ok(sym)` is the caller's fetch-validator (the UI passes its cached
    check). Returns (symbol, changed): the raw if it already loads (NO llm call), else an LLM-resolved
    symbol that actually LOADS, else the raw unchanged (the caller surfaces the bad ticker)."""
    raw = (raw or "").strip().upper()
    if raw and is_ok(raw):
        return raw, False                       # already valid - skip the LLM
    cand = resolve_ticker(raw)                  # LLM name/alias -> Yahoo symbol
    if cand and cand != raw and is_ok(cand):
        return cand, True                       # resolved to something that actually loads
    return raw, False                           # nothing better; the caller handles the bad ticker


# --- FEASIBILITY GATE (gate 0 of the honesty stack) -------------------------------------------------
# A SEPARATE call from MODE_PROMPT (Lesson 029: keep the routing decision from being degraded). Runs only
# on non-help requests inside classify. "Can this be answered with daily close of 1-2 listed tickers?"
SCOPE_PROMPT = (
    "You are a feasibility gate for a quant BACKTESTER. Its ONLY data is daily CLOSE prices of listed tickers "
    "(stocks / ETFs / major crypto), 1-2 at a time. Nothing else.\n"
    "Decide if the request can be answered with THAT data alone. Reply EXACTLY one word: IN or OUT.\n"
    "OUT (out of scope) if it needs ANY other data, or is not a testable price rule:\n"
    "- fundamentals (P/E, earnings, short interest, float); credit / CDS / trade-finance spreads; factor data "
    "(Fama-French); macro / rates; intraday; options / derivatives; VOLUME; news / sentiment; alt-data\n"
    "- a single-asset strategy GATED on a SECOND series ('SPY only when VIX < 30')\n"
    "- a markets RESEARCH QUESTION or explanation ('what is the transmission mechanism / lag structure?'), "
    "not a testable rule\n"
    "IN otherwise - ANY price-based strategy on listed tickers: SMA / RSI / breakout / momentum / mean-"
    "reversion, single-asset OR a pairs / spread on TWO listed tickers, OR a DCA / contribution comparison.\n"
    "When genuinely unsure, reply IN. Only reply OUT when it CLEARLY needs unavailable data or is a question.\n"
    "Examples:\n"
    "  '50/200 SMA crossover on SPY'                         -> IN\n"
    "  'RSI(14) mean reversion on AAPL since 2020'           -> IN\n"
    "  'pairs trade KO vs PEP, z-score +/-2'                 -> IN\n"
    "  'DCA $1k monthly into SPY vs buying every 3-day dip'  -> IN\n"
    "  'backtest using P/E ratios and short interest'        -> OUT\n"
    "  'long SPY only when VIX < 30'                         -> OUT\n"
    "  'buy when volume spikes 2x the 20-day average'        -> OUT\n"
    "  'trade-finance spreads / CDS on commodity banks as a leading signal' -> OUT\n"
    "  'explain the lead-lag between oil and airlines'       -> OUT\n"
)

def scope_check(request: str) -> bool:
    """Feasibility gate. True = OUT of scope (needs data we lack, or is a question); False = IN scope.
    Biased to IN when unsure - the editable spec + the ruler are downstream nets."""
    reply = _strip_think(llm.invoke([SystemMessage(content=SCOPE_PROMPT),
                                     HumanMessage(content=request)]).content).strip().lower()
    out = reply.startswith("out")
    print(f"[scope] {'OUT' if out else 'IN'}")
    return out


SCOPE_REFUSAL_PROMPT = (
    "You are the scope guard for a quant BACKTESTER that has ONLY daily CLOSE prices of listed tickers "
    "(stocks / ETFs), 1-2 at a time. The user's request is OUT OF SCOPE. Write a short, honest reply in "
    "markdown (~4 brief bullet points), in this order:\n"
    "1. say plainly WHAT it cannot do and WHY - name the missing data / the boundary.\n"
    "2. if the request includes a research QUESTION, decline it explicitly - do NOT answer or guess at it.\n"
    "3. if a sensible PRICE PROXY exists using WELL-KNOWN, liquid ETFs, propose SPECIFIC tickers and the "
    "testable version (e.g. credit stress -> HYG or the HYG/LQD ratio; financials -> XLF / KRE; commodities "
    "-> DBC / XLE; so a HYG-vs-DBC lead-lag or pair). If NO sensible proxy exists, say so - never invent "
    "tickers or data sources.\n"
    "4. CAVEAT the proxy: it is a crude price simplification that DROPS the original thesis - a null result "
    "would not disprove it, a positive one would not confirm it.\n"
    "End by asking if they want to try the proxy, or rephrase with listed tickers. Be concise and honest."
)

def scope_refusal(request: str) -> str:
    """A 4-part honest refusal for an out_of_scope request: boundary + decline-question + proxy + caveat."""
    return _strip_think(llm.invoke([SystemMessage(content=SCOPE_REFUSAL_PROMPT),
                                    HumanMessage(content=request)]).content).strip()

# --- METHOD-FEASIBILITY GATE (extends the DATA-scope gate to METHOD-scope) --------------------------
# A SEPARATE call (Lesson 029), like scope_check. The engine runs a strategy as a numpy+pandas function on
# daily closes - it has NO copula / GARCH / Kalman / ML libraries. scope_check asks "is the DATA available?";
# this asks "can we faithfully build the METHOD?". If the request NAMES a technique from the curated list we
# DISCLOSE the gap (the proxy still runs) rather than silently downgrade it. See method-feasibility-plan.md,
# Lesson 044. CURATED LIST, not a fuzzy "too fancy?" - precise on named methods so plain price rules (z-score
# / SMA / RSI) do NOT false-flag (banner fatigue would decay this into the silent-downgrade it prevents).
METHOD_PROMPT = (
    "You gate a quant BACKTESTER. Its strategy is a Python function using ONLY numpy + pandas on daily CLOSE "
    "prices (1-2 tickers). It CANNOT fit or use: copulas / vine copulas, GARCH / EGARCH / stochastic-vol "
    "models, Kalman filters, cointegration / Johansen tests, HMM / regime-switching models, ML / regression "
    "models (trees, neural nets, SVMs), options / implied-vol models, or factor models (Fama-French).\n"
    "Does the request NAME one of those techniques? Reply EXACTLY one line:\n"
    "  NONE\n"
    "  METHOD: <name> | NEEDS: <what a faithful version needs> | INTENT: <what it was for>\n"
    "Flag ONLY a NAMED technique from the list above. Plain price rules are NOT flagged - "
    "z-score / SMA / EMA / RSI / MACD / Bollinger / breakout / momentum / mean-reversion / DCA -> NONE.\n"
    "Examples:\n"
    "  'vine copula pairs trade NVDA vs AMD for tail dependence'\n"
    "     -> METHOD: vine copula | NEEDS: a copula library + return-distribution fitting | INTENT: tail dependence\n"
    "  'GARCH vol-targeting on QQQ'\n"
    "     -> METHOD: GARCH | NEEDS: a volatility-model library | INTENT: volatility forecasting\n"
    "  'Kalman-filter dynamic hedge ratio for KO/PEP'\n"
    "     -> METHOD: Kalman filter | NEEDS: a state-space / filtering library | INTENT: a dynamic hedge ratio\n"
    "  'z-score pairs trade KO vs PEP'                     -> NONE\n"
    "  '50/200 SMA crossover on SPY'                       -> NONE\n"
    "  'RSI(14) mean reversion on AAPL'                    -> NONE\n"
)

def _parse_method(reply):
    """Parse the method-gate reply -> {method, needs, intent} or None. Robust to a small model that may wrap
    the line in prose: scan for a NONE / METHOD: line anywhere. Missing NEEDS/INTENT default to ''."""
    for line in reply.splitlines():
        low = line.strip().lower()
        if low.startswith("none"):
            return None
        if low.startswith("method:"):
            segs = [s.strip() for s in line.split(":", 1)[1].split("|")]
            fields = {"method": segs[0] if segs else "", "needs": "", "intent": ""}
            for seg in segs[1:]:
                if ":" in seg:
                    k, v = seg.split(":", 1)
                    if k.strip().lower() in ("needs", "intent"):
                        fields[k.strip().lower()] = v.strip()
            return fields if fields["method"] else None
    return None

def method_check(request: str):
    """METHOD-feasibility gate. Returns {method, needs, intent} if the request NAMES a technique the engine
    can't faithfully build (-> disclose the substitution), else None. Separate call (un-degraded, Lesson 029)."""
    reply = _strip_think(llm.invoke([SystemMessage(content=METHOD_PROMPT),
                                     HumanMessage(content=request)]).content)
    note = _parse_method(reply)
    print(f"[method] {note['method'] if note else 'NONE'}")
    return note


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
    if scope_check(req):                   # GATE 0: feasibility, a SEPARATE call (MODE_PROMPT untouched, Lesson 029)
        print("[classify] mode=out_of_scope")
        return {"mode": "out_of_scope"}
    mode = "contribution" if "contribution" in ml else "position"
    # GATE (method feasibility): does the request NAME a technique we can't faithfully build? A SEPARATE call
    # (Lesson 029). Sequential for now - could run in PARALLEL off `req` on a multi-worker backend, but the
    # local single-model server serialises anyway, so threading buys ~nothing here. See method-feasibility-plan.md.
    mnote = method_check(req)

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
    # PAIRS: a multi-asset "position" request is really the pairs engine. Detect it HERE (not UI-only) so the
    # CLI + eval + UI all route identically - the pairs engine was previously UNREACHABLE from classify.
    if mode == "position" and is_multi_asset_position(req, mode):
        tks = resolve_pair_tickers(req)
        print(f"[classify] mode=pairs {tks}")
        out = {"mode": "pairs", "ticker": tks[0] if len(tks) >= 1 else "",
               "ticker_b": tks[1] if len(tks) >= 2 else ""}
        if start:
            out["start_date"] = start
        if mnote:
            out["method_note"] = mnote
        return out
    print(f"[classify] mode={mode} ticker={ticker} start={start} amount={amount}")
    out = {"mode": mode}
    if ticker:
        out["ticker"] = ticker
    if start:
        out["start_date"] = start
    if amount:
        out["amount"] = amount
    if mnote:
        out["method_note"] = mnote
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
