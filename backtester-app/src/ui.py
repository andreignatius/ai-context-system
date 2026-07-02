import os, sys
import streamlit as st
st.set_page_config(page_title="Quant Backtester", page_icon="📈", layout="centered")

# make `src` importable regardless of how Streamlit launches this file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# bridge Streamlit Cloud secrets -> env BEFORE importing src (config reads LLM_PROVIDER at import)
try:
    for _k in ("LLM_PROVIDER", "DEEPINFRA_API_KEY", "DEEPINFRA_MODEL",
               "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass                       # no secrets.toml locally -> fall back to .env / shell env

import pandas as pd
from src.graph import build_run_graph
from src.agents import (classify, write_spec, extract_legs, _leg_label, is_multi_asset_position,
                        resolve_pair_tickers, resolve_ticker, scope_refusal)
from src.runner import _load_strategy, _load_strategy_pair
from src.core.robustness import rolling_robustness, rolling_robustness_pairs
from src.core.engine import run_backtest
from src.core.data import load_prices
from src.core.metrics import buy_and_hold, longest_drawdown_days, annual_returns, dca
from src.export import full_script, full_contribution_script, full_pairs_script
from src.config import get_langfuse_handler

st.title("📈 Quant Backtester")

@st.cache_resource
def _run_graph():
    return build_run_graph()

@st.cache_resource
def _handler():                       # Langfuse tracer (no-op if LANGFUSE_* secrets are unset)
    return get_langfuse_handler()

@st.cache_data
def _prices(ticker, period, start):
    return load_prices(ticker, period, start)

def _ticker_ok(t, start):
    """True if a ticker actually loads data (cached) - validates pairs BEFORE the coder runs."""
    if not t:
        return False
    try:
        return len(_prices(t, "5y", start)) > 0
    except Exception:
        return False

# friendly labels for live progress - the graph emits one event per node as it finishes
_STEP = {
    "write_spec": "📝 interpreting the strategy…",
    "write_code": "⌨️ writing the strategy code…",
    "run": "⚙️ running the backtest engine…",
    "contribution_run": "💵 running the cash-flow engine…",
    "judge": "🔍 checking soundness (self-healing)…",
}

st.caption("An AI quant backtester — **honest about look-ahead, costs, and out-of-sample robustness**, "
           "not just plausible-looking code.")
st.caption("daily close (auto-adjusted) · yfinance · **prompt-driven** (name the ticker in your request, "
           "e.g. \"backtest IWM…\" or \"GOOG vs SPY\") · equity = growth of $1, fully invested when long")

st.session_state.setdefault("history", [])
st.session_state.setdefault("draft", None)   # pending interpretation awaiting confirm/fix
st.session_state.setdefault("show_help", False)   # show the welcome (first run, or a 'what can you do' query)


def _robustness_ui(uid, key, compute, has_bh):
    """Controls + button + results for an out-of-sample robustness run. compute(tw, sp) -> (table, summary)."""
    with st.expander("📈 out-of-sample robustness — does it generalize across regimes?"):
        cc = st.columns(2)
        tw = cc[0].number_input("Test window (months)", 6, 36, 12, key=f"{key}_tw_{uid}")
        sp = cc[1].number_input("Step (months)", 1, 12, 6, key=f"{key}_sp_{uid}")
        if st.button("Run robustness", key=f"{key}_btn_{uid}"):
            with st.spinner("rolling out-of-sample windows…"):
                tbl, s = compute(int(tw), int(sp))
            if not s["windows"]:
                st.warning("Not enough data for that window/step — try a shorter window."); return
            cols = st.columns(4 if has_bh else 3)
            i = 0
            cols[i].metric("Positive windows", f"{s['pct_positive']:.0%}"); i += 1
            if has_bh:
                cols[i].metric("Beat buy & hold", f"{s.get('beat_bh', 0)}/{s['windows']}"); i += 1
            cols[i].metric("Avg window Sharpe", f"{s['avg_sharpe']:.2f}"); i += 1
            cols[i].metric("Worst window", f"{s['worst_return']:+.1%}")
            ww = s["worst_window"]
            st.caption(f"{s['windows']} rolling windows  ·  full-sample Sharpe **{s['full_sharpe']:.2f}**  ·  "
                       f"worst: {ww[0]} → {ww[1]}")
            chart_cols = ["sharpe", "bh_sharpe"] if has_bh else ["sharpe"]
            st.line_chart(tbl.set_index("test_end")[chart_cols].rename(
                columns={"sharpe": "Strategy", "bh_sharpe": "Buy & Hold"}))
            fmt = {"total_return": "{:+.1%}", "sharpe": "{:.2f}", "max_drawdown": "{:.1%}"}
            if has_bh:
                fmt.update({"bh_return": "{:+.1%}", "bh_sharpe": "{:.2f}"})
            st.dataframe(tbl.style.format(fmt))


def render(b, uid=""):                  # uid keeps widget IDs unique across replayed history turns
    if b.get("scope_error"):              # out of scope (e.g. identical legs / cross-asset) - refuse clearly
        st.warning("⚠️ out of scope — " + b.get("run_result", {}).get("failures",
                                                                       "this request isn't supported yet."))
        with st.expander("interpreted spec"):
            st.text(b.get("spec", ""))
        return
    if b["status"] == "ok":
        st.success("✅ sound — the strategy runs point-in-time and is valid")
        # honest disclosure: live requests are SOUND-checked, never CORRECTNESS-graded (the ground-truth
        # ruler only covers the offline eval suite). Don't let "sound" overstate trust on a novel strategy.
        st.caption("**Sound, not verified-correct** — this ran and didn't peek at the future, but the logic "
                   "was **not** graded against a known-correct baseline. Skim the strategy code below before "
                   "trusting the numbers.")
    else:
        st.error("❌ stuck — the judge gave up after retries")

    if b.get("mode") == "pairs":
        pr = b.get("pairs_result")
        if pr:
            st.caption(f"**{pr['ticker_a']} vs {pr['ticker_b']}**  ·  log-spread z-score · dollar-neutral")
            m = pr["metrics"]
            c = st.columns(4)
            c[0].metric("Total return", f"{m['total_return']:+.1%}")
            c[1].metric("CAGR", f"{m['ann_return']:+.1%}")
            c[2].metric("Sharpe", f"{m['sharpe']:.2f}")
            c[3].metric("Max drawdown", f"{m['max_drawdown']:.1%}")
            st.caption(f"{m['n_trades']} trades  ·  **net of ~5 bps/side** (~10 bps round-trip) — costs scale "
                       "with turnover, so this no longer flatters churn")
            if pr.get("equity_curve") is not None:
                st.line_chart(pr["equity_curve"].rename("spread equity (growth of $1)"))
            with st.expander("📄 full pairs script (reproducible end-to-end)"):
                script = full_pairs_script(b["strategy_code"], pr["ticker_a"], pr["ticker_b"],
                                           pr["equity_curve"].index[0].date())
                st.code(script, language="python")
                st.download_button("Download full script (.py)", script, key=f"pairs_script_{uid}",
                                   file_name=f"pairs_{pr['ticker_a']}_{pr['ticker_b']}.py",
                                   mime="text/x-python")
            if pr.get("equity_curve") is not None:
                _start = pr["equity_curve"].index[0].date()
                _robustness_ui(uid, "prob",
                               lambda tw, sp: rolling_robustness_pairs(
                                   _prices(pr["ticker_a"], "5y", _start), _prices(pr["ticker_b"], "5y", _start),
                                   _load_strategy_pair(b["strategy_code"]), tw, sp),
                               has_bh=False)

        elif b.get("run_result", {}).get("failures"):
            st.warning(b["run_result"]["failures"])
        st.code(b["strategy_code"], language="python")
        with st.expander("strategy spec"):
            st.text(b["spec"])
        return


    # CONTRIBUTION mode: a dollar comparison (different shape from the position render)
    if b.get("mode") == "contribution":
        cr = b.get("contribution_result")
        px = b.get("prices")
        if cr:
            legs = cr.get("legs")
            if legs:                                   # generic two-leg shape (cadence + own amount)
                idx = [l["label"] for l in legs]
                per_dep = [l["amount"] for l in legs]
                rows = legs
            else:                                      # legacy shape (signal vs monthly @ one amount)
                rows = [cr["signal"], cr["dca"]]
                idx = ["Buy-the-signal", "Monthly DCA"]
                per_dep = [cr.get("amount", 1000.0)] * 2
            curve = cr.get("signal_curve")               # the actual run window (cross-asset = common window)
            if curve is not None and len(curve):
                st.caption(f"{curve.index[0].date()} -> {curve.index[-1].date()}")
            elif px is not None:
                st.caption(f"{px.index[0].date()} -> {px.index[-1].date()}")
            if cr.get("warning"):                        # overlap-coverage notice (cross-asset)
                st.warning("⚠️ " + cr["warning"])
            comp = pd.DataFrame(
                {"per deposit": per_dep,
                 "deposits": [r["n"] for r in rows],
                 "invested": [r["invested"] for r in rows],
                 "final": [r["final"] for r in rows],
                 "multiple": [r["final"] / r["invested"] if r["invested"] else 0.0 for r in rows]},
                index=idx)
            st.dataframe(comp.style.format(
                {"per deposit": "${:,.0f}", "invested": "${:,.0f}", "final": "${:,.0f}", "multiple": "{:.2f}x"}))
            if cr.get("signal_curve") is not None:      # portfolio value over time (leg A vs leg B)
                st.line_chart(pd.DataFrame({idx[0]: cr["signal_curve"], idx[1]: cr["dca_curve"]}))
            st.caption("Compare the MULTIPLE, not the absolute final (deposit totals differ).")
            if px is not None:
                with st.expander("data (preview + download)"):
                    st.dataframe(px.rename("close").to_frame().tail(10))
                    st.download_button("Download prices (CSV)", px.to_csv(), key=f"c_prices_{uid}",
                                       file_name=f"{b.get('ticker', 'data')}.csv", mime="text/csv")
                with st.expander("📄 full contribution script (reproducible end-to-end)"):
                    script = full_contribution_script(b["strategy_code"], b.get("ticker", "SPY"),
                                                      px.index[0].date(), cr.get("legs"))
                    st.code(script, language="python")
                    st.download_button("Download full script (.py)", script, key=f"c_script_{uid}",
                                       file_name=f"contribution_{b.get('ticker', 'SPY')}.py",
                                       mime="text/x-python")
        elif b.get("run_result", {}).get("failures"):
            st.warning(b["run_result"]["failures"])
        st.code(b["strategy_code"], language="python")
        with st.expander("strategy spec"):
            st.text(b["spec"])
        return

    m = b["run_result"]["metrics"]
    eq = b.get("equity")
    px = b.get("prices")
    if m:
        c = st.columns(5)
        c[0].metric("Total return", f"{m['total_return']:+.1%}")
        c[1].metric("CAGR", f"{m['ann_return']:+.1%}")
        c[2].metric("Sharpe", f"{m['sharpe']:.2f}")
        c[3].metric("Max drawdown", f"{m['max_drawdown']:.1%}")
        if eq is not None:
            c[4].metric("Longest DD", f"{longest_drawdown_days(eq)}d")
        st.caption(f"{m['n_trades']} trades  ·  **net of ~5 bps/side** (~10 bps round-trip) — costs scale "
                   "with turnover, so this no longer flatters churn")

    # benchmark overlay: strategy vs buy-and-hold (same $1 lump-sum basis -> directly comparable)
    if eq is not None and px is not None:
        st.line_chart(pd.DataFrame({"Strategy": eq, "Buy & Hold": buy_and_hold(px)}))
        # all three multiples on one line so the STRATEGY return is comparable to the benchmarks
        _, dca_final, dca_invested = dca(px, 100.0)
        strat_mult = 1 + m["total_return"]
        bh_mult = px.iloc[-1] / px.iloc[0]
        st.caption(f"**Strategy {strat_mult:.2f}x**  ·  buy & hold {bh_mult:.2f}x  ·  "
                   f"DCA \\$100/mo {dca_final / dca_invested:.2f}x "
                   f"(\\${dca_final:,.0f} on \\${dca_invested:,.0f})")
        st.caption(f"period: {px.index[0].date()} -> {px.index[-1].date()}  ·  {len(px)} trading days")
        with st.expander("annual returns"):
            st.dataframe(annual_returns(eq).to_frame("return").style.format("{:+.1%}"))
        with st.expander("data (preview + download)"):
            st.dataframe(px.rename("close").to_frame().tail(10))
            st.download_button("Download prices (CSV)", px.to_csv(), key=f"p_prices_{uid}",
                               file_name=f"{b.get('ticker', 'data')}_{b.get('period', '')}.csv",
                               mime="text/csv")
            st.download_button("Download strategy (.py)", b["strategy_code"], key=f"p_strat_{uid}",
                               file_name="strategy.py", mime="text/x-python")
        with st.expander("📄 full backtest script (reproducible end-to-end)"):
            script = full_script(b["strategy_code"], b.get("ticker", "SPY"), b.get("period", "2y"))
            st.caption("Data load -> signal -> engine -> metrics -> plot, in one standalone file.")
            st.code(script, language="python")
            st.download_button("Download full script (.py)", script, key=f"p_script_{uid}",
                               file_name=f"backtest_{b.get('ticker', 'SPY')}.py",
                               mime="text/x-python")
        # RE-exec of code already sandbox-vetted upstream in run_strategy (sandbox-plan.md decision b) - the
        # dangerous FIRST exec was bounded, so this in-process re-run of known-good code can't newly hang.
        _robustness_ui(uid, "rob",
                       lambda tw, sp: rolling_robustness(px, _load_strategy(b["strategy_code"]), tw, sp),
                       has_bh=True)

    st.code(b["strategy_code"], language="python")
    with st.expander("strategy spec"):
        st.text(b["spec"])
    if b["run_result"]["failures"]:
        st.warning(b["run_result"]["failures"])


TEMPLATES = [
    {"name": "📈 Trend following", "prompt": "Long SPY when the 50-day SMA is above the 200-day SMA, since 2020"},
    {"name": "📊 Mean reversion",  "prompt": "Go long SPY when RSI(14) is below 30 and exit when above 70, since 2020"},
    {"name": "💵 DCA comparison",  "prompt": "Compare investing $1000 in SPY on every 3-day dip vs $1000 monthly DCA, since 2021"},
    {"name": "🔗 Pairs trade",     "prompt": "Pairs trade XLF vs XLI: long the weaker when the spread widens to +2 SD, since 2019"},
]


def show_welcome(rich=False):
    """The intro + one-click example templates. Two flavours so the two entry points feel different:
    - rich=False (FIRST visit): a short greeting -- don't wall-of-text a brand-new visitor.
    - rich=True ('what can you do?'): a fuller, expressive answer (the honesty stack + the 3 request kinds),
      so asking actually produces something NEW rather than re-showing the same banner."""
    with st.chat_message("assistant"):
        if rich:
            st.markdown(
                "Here's what I do — I turn a plain-English strategy or money question into a **real, graded "
                "backtest**, and I'm honest about the three ways backtests usually lie:\n\n"
                "- 📝 **Write & run it** — I write the strategy code and run it *point-in-time* (each bar sees only "
                "the past), so look-ahead bias is **structurally impossible**, not just avoided.\n"
                "- 📏 **Grade it honestly** — every run is checked against a **known-correct baseline** (a "
                "ground-truth ruler), so plausible-but-wrong code gets caught instead of trusted.\n"
                "- 🌍 **Stress-test it** — roll it across **out-of-sample** windows to see if it *generalizes* or is "
                "just curve-fit (\"positive in 4/12 windows, beats buy & hold in 3 — looks like beta, not alpha\").\n\n"
                "**Three kinds of request I handle:**\n"
                "- **Single-asset strategies** — SMA / RSI / breakout (e.g. *\"long SPY when the 50-day SMA is above "
                "the 200-day\"*)\n"
                "- **DCA comparisons** — dollar-cost-averaging questions (e.g. *\"$1000/mo into SPY vs buying every "
                "3-day dip\"*)\n"
                "- **Pairs trades** — market-neutral spreads (e.g. *\"XLF vs XLI when the spread hits +2 SD\"*)\n\n"
                "Try one 👇")
        else:
            st.markdown("👋 **An AI quant backtester** — describe a strategy or a money question in plain English, "
                        "and I'll write it, run it point-in-time (no look-ahead), grade it against a known-correct "
                        "baseline, and let you stress-test it out-of-sample. Try an example:")
        cols = st.columns(2)
        for i, t in enumerate(TEMPLATES):
            if cols[i % 2].button(t["name"], key=f"tpl_{i}", use_container_width=True):
                st.session_state.pending = t["prompt"]      # routed through the SAME pipeline as typing
                st.rerun()
        if not rich:
            st.caption("I handle single-asset strategies (SMA / RSI / breakout), DCA comparisons, and pairs trades.")


def process_request(req):
    """Read a request -> a confirmable DRAFT. The SAME path for typed prompts AND template clicks. A META
    message (classify -> 'help') shows the welcome instead of forcing it into a strategy."""
    st.session_state.show_help = False
    s = {"request": req}
    with st.status("Reading your request…", expanded=True) as status:
        s.update(classify(s))
        if s.get("mode") == "help":                         # not a backtest request -> welcome, no draft
            status.update(label="Here's what I can do 👇", state="complete")
            st.session_state.show_help = True
            st.session_state.help_query = req               # echoed as a user bubble so the prompt isn't lost
            st.session_state.draft = None
            return
        if s.get("mode") == "out_of_scope":                 # GATE 0: can't honestly answer -> refuse + help
            status.update(label="Out of scope", state="complete")
            s["scope_error"] = True
            s["scope_msg"] = scope_refusal(req)             # 4-part: boundary / decline-question / proxy / caveat
            st.session_state.draft = s
            return
        status.write(f"✓ understood — engine: **{s.get('mode')}**")
        s.update(write_spec(s)); status.write("✓ drafted the spec")
        if is_multi_asset_position(req, s.get("mode")):
            tks = resolve_pair_tickers(req)             # maps NAMES -> symbols (Exxon Mobil -> XOM); [] if unsure
            s["mode"] = "pairs"
            s["ticker"]   = tks[0] if len(tks) >= 1 else ""
            s["ticker_b"] = tks[1] if len(tks) >= 2 else ""
            status.write(f"✓ pairs: **{s['ticker'] or '?'} vs {s['ticker_b'] or '?'}**  (confirm below)")
        elif s.get("mode") == "contribution":
            s.update(extract_legs(s)); status.write("✓ parsed the legs")
        status.update(label="Ready ✓", state="complete")
    st.session_state.draft = s


# INPUT IS HANDLED FIRST (before any rendering): a submit must process + rerun BEFORE we draw the
# welcome, or st.rerun() fires mid-run after keyed widgets already streamed -> blank next render.
# a template click sets `pending` -> process it through the same pipeline as a typed prompt
if st.session_state.get("pending"):
    process_request(st.session_state.pop("pending"))
    st.rerun()

# chat input (the widget still pins to the bottom of the page wherever it's called) -> same pipeline
if req := st.chat_input("Describe a strategy, a money question, or ask 'what can you do?'…"):
    process_request(req)
    st.rerun()


# replay the conversation
for i, turn in enumerate(st.session_state.history):
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.write(turn["text"])
        else:
            render(turn["build"], uid=str(i))

# --- confirm flow: a pending DRAFT (the interpretation) the user approves or corrects BEFORE running ---
draft = st.session_state.get("draft")

# welcome screen, two entry points:
#  - 'what can you do?' (show_help): echo the question + the RICH answer, so asking produces something new
#  - first visit (empty history): the short greeting, to fix blank-box paralysis
if not draft:
    if st.session_state.get("show_help"):
        with st.chat_message("user"):
            st.write(st.session_state.get("help_query", "what can you do?"))
        show_welcome(rich=True)
    elif not st.session_state.history:
        show_welcome(rich=False)

if draft:
    if draft.get("scope_error"):
        with st.chat_message("user"):
            st.write(draft["request"])
        with st.chat_message("assistant"):
            st.warning(draft["scope_msg"])              # the 4-part honest refusal (boundary/proxy/caveat)
            st.caption("What I can do: single-asset strategies (SMA / RSI / breakout), DCA comparisons, and "
                       "pairs trades — on listed tickers. Try an example, or rephrase:")
            cols = st.columns(2)
            for i, t in enumerate(TEMPLATES):           # detect-and-HELP: offer concrete in-scope paths
                if cols[i % 2].button(t["name"], key=f"scope_tpl_{i}", use_container_width=True):
                    st.session_state.pending = t["prompt"]; st.session_state.draft = None; st.rerun()
            if st.button("Dismiss"):
                st.session_state.draft = None; st.rerun()
        st.stop()           # don't render the confirm panel / Run button

    ticker = draft.get("ticker") or "SPY"      # prompt-extracted asset (default SPY) - fully prompt-driven now
    period = "5y"                              # default window; an extracted start_date overrides it
    start = None
    with st.chat_message("user"):
        st.write(draft["request"])
    with st.chat_message("assistant"):
        st.info("Here's how I read your request — **edit the spec if needed, then run.** "
                "Your words go straight to the coder (no LLM re-interpretation):")
        st.caption(f"engine: **{draft['mode']}**  ·  ticker: **{draft.get('ticker') or 'SPY'}**  ·  "
                   f"start: **{draft.get('start_date') or '(default 5y)'}**")

        # CONTRIBUTION: two editable legs (cadence + own amount) -> "weekly $250 vs monthly $1000".
        # Direct user control (c2 philosophy); prefilled by extract_legs, but the user has final say.
        leg_inputs = None
        pair_a = pair_b = None
        if draft.get("mode") == "contribution":
            st.markdown("**Compare two deposit schedules** — set each leg's ticker, cadence, and amount "
                        "(different tickers = a cross-asset comparison, e.g. GOOG vs SPY):")
            hint = list(draft.get("legs") or [])
            while len(hint) < 2:                       # default the second leg to monthly DCA
                hint.append({"cadence": "monthly", "amount": draft.get("amount", 1000.0)})
            cad_opts = ["signal", "weekly", "monthly"]
            leg_inputs = []
            for i in range(2):
                c1, c2, c3 = st.columns([1, 1, 1])
                tkr = c1.text_input(f"Leg {chr(65 + i)} ticker", value=(hint[i].get("ticker") or ticker),
                                    key=f"leg{i}_tkr").strip().upper()
                hc = hint[i].get("cadence") if hint[i].get("cadence") in cad_opts else "monthly"
                cad = c2.selectbox(f"Leg {chr(65 + i)} cadence", cad_opts,
                                   index=cad_opts.index(hc), key=f"leg{i}_cad")
                amt = c3.number_input(f"Leg {chr(65 + i)} $ / dep", min_value=1.0,
                                      value=float(hint[i].get("amount") or 1000.0), step=50.0, key=f"leg{i}_amt")
                leg_inputs.append({"ticker": tkr, "cadence": cad, "amount": amt})
            if any(l["cadence"] == "signal" for l in leg_inputs):
                st.caption("A **signal** leg deposits when the strategy fires — edit the spec below. "
                           "(Signal cadence runs on a single asset; cross-asset uses weekly/monthly.)")
        elif draft.get("mode") == "pairs":
            st.markdown("**Pair** — confirm or correct the two tickers "
                        "(Yahoo symbols, e.g. **XOM** = Exxon Mobil, **SHEL** = Shell):")
            pca, pcb = st.columns(2)
            pair_a = pca.text_input("Ticker A", value=draft.get("ticker") or "", key="pair_a").strip().upper()
            pair_b = pcb.text_input("Ticker B", value=draft.get("ticker_b") or "", key="pair_b").strip().upper()
        # (position mode: no per-deposit line - it's growth-of-$1, fully invested when long; shown in the header)

        # the spec/strategy is only used by a 'signal' leg. For a calendar-only DCA comparison there is no
        # strategy, so hide the (irrelevant, often misframed) spec box rather than confuse the user with it.
        needs_spec = (draft.get("mode") != "contribution") or any(l["cadence"] == "signal" for l in (leg_inputs or []))
        if needs_spec:
            edited_spec = st.text_area("interpreted spec (editable)", value=draft["spec"], height=320, key="spec_edit")
        else:
            st.caption("No strategy needed — this is a calendar DCA comparison (driven by the leg controls above).")
            edited_spec = draft["spec"]                 # carried through but unused by the engine for calendar legs
        run_clicked = st.button("✅ Run it", type="primary")

    # pre-check: identical legs = an asset-vs-itself comparison -> refuse instantly (no wasted coder call).
    # identity includes the TICKER, so GOOG-vs-SPY is NOT identical. keeps the draft so the user can adjust.
    identical_legs = bool(leg_inputs) and len({(l["ticker"], l["cadence"], l["amount"]) for l in leg_inputs}) == 1
    # pairs: the editable fields OVERRIDE the extracted tickers; validate BOTH load before the (costly) coder run
    pair_bad = None
    if draft.get("mode") == "pairs":
        ticker = pair_a or ""                                  # Ticker A drives the single-load `prices`
        pair_bad = [t or "(blank)" for t in (pair_a, pair_b) if not _ticker_ok(t, draft.get("start_date"))]

    if run_clicked and identical_legs:
        st.error("⚠️ Both legs are identical (same ticker + cadence + amount) — that compares the asset to "
                 "itself. Change a ticker, cadence, or amount to make a real comparison.")
    elif run_clicked and pair_bad:
        # v1 name-help: map each bad entry NAME -> a symbol that actually loads, and SUGGEST it (resolve ->
        # validate -> show; never auto-swap, since an LLM lookup can hallucinate a plausible-but-wrong symbol)
        hits = []
        for typed in (pair_a, pair_b):
            if typed and not _ticker_ok(typed, draft.get("start_date")):
                sym = resolve_ticker(typed)
                if sym and _ticker_ok(sym, draft.get("start_date")):
                    hits.append((typed, sym))
        if hits:
            sugg = "  ·  ".join(f"**{typed}** -> **{sym}**" for typed, sym in hits)
            st.error(f"⚠️ Those don't look like Yahoo symbols. Did you mean:  {sugg}  —  "
                     "type the symbol(s) above, then Run.")
        else:
            st.error("⚠️ Couldn't load price data for: **" + "**, **".join(pair_bad) + "**. "
                     "Use Yahoo Finance symbols (Exxon Mobil = **XOM**, Shell = **SHEL**, S&P 500 = **SPY**).")
    elif run_clicked:
        prices = _prices(ticker, period, draft.get("start_date") or start)
        state = {**draft, "spec": edited_spec, "prices": prices, "ticker": ticker, "period": period,
                 "fix_target": "", "feedback": ""}         # the EDITED spec is what the coder builds from
        if draft.get("mode") == "pairs":
            state["ticker_b"] = pair_b                         # the edited Ticker B into the engine
        if leg_inputs:                                     # user-confirmed legs -> per-leg ticker+cadence+amount
            state["legs"] = [{"ticker": l["ticker"], "cadence": l["cadence"], "amount": l["amount"],
                              "label": _leg_label(l["cadence"], l["amount"], l["ticker"])} for l in leg_inputs]
        b = state
        with st.status("Working…", expanded=True) as status:
            # stream the graph: 'updates' = WHICH node just finished (live label); 'values' = the accumulated
            # state, whose LAST value is the final result. Real progress, not a fixed spinner.
            for smode, chunk in _run_graph().stream(state, stream_mode=["updates", "values"],
                                                    config={"callbacks": [_handler()]}):
                if smode == "updates":
                    status.write(_STEP.get(next(iter(chunk)), next(iter(chunk))))
                else:
                    b = chunk
            status.update(label="Done ✓", state="complete")
        b["prices"] = prices
        b["equity"] = None
        if b.get("mode") == "position" and b["status"] == "ok":
            # RE-exec of code already sandbox-vetted (status=="ok" means it cleared run_strategy) - decision b
            b["equity"] = run_backtest(prices, _load_strategy(b["strategy_code"])).equity_curve

        st.session_state.history += [{"role": "user", "text": draft["request"]},
                                     {"role": "assistant", "build": b}]
        st.session_state.draft = None
        st.rerun()

# (chat input is handled at the TOP of the script, before any rendering -- see the input block above)
