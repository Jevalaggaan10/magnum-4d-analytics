"""
Magnum / Damacai / 4dyes3 4D Analytics — Friendly Edition.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Page config + global banner
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="4D Number Explorer (Honest Edition)",
    page_icon="🎲",
    layout="wide",
)

st.title("🎲 4D Number Explorer")
st.markdown(
    """
    <div style="background:#fff4e5;border-left:6px solid #d97706;padding:14px;border-radius:6px;
                font-size:15px;margin-bottom:10px;color:#1B1B1F;">
    <b>Heads up.</b> 4D lottery draws are random. No tool in the world — including this one —
    can predict the winning number. For every RM10 you bet, the lottery keeps about
    RM3.60 on average. This app helps you <i>understand</i> the numbers, not beat them.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

DATA_DIR = "c:/codex/lottery"


@st.cache_data(show_spinner=True)
def load_data(cache_bust: int = 0):
    from data_loader import load_combined_data, extract_top3_prizes
    df_full = load_combined_data(DATA_DIR, include_cache=True)
    top3 = extract_top3_prizes(df_full)
    return df_full, top3


@st.cache_data(show_spinner=False)
def get_last_sync_info():
    from data_sources.manager import get_last_sync
    return get_last_sync()


@st.cache_data(show_spinner=True)
def run_predictors(df_hash: int, _df: pd.DataFrame, include_ml: bool, top_n: int):
    from predictor_suite import run_all
    return run_all(_df, include_ml=include_ml, top_n=top_n, lstm_epochs=2)


@st.cache_data(show_spinner=False)
def build_consensus_table(_results, top_n: int):
    from consensus import build_consensus
    return build_consensus(_results, top_n=top_n)


@st.cache_data(show_spinner=True)
def run_randomness(df_hash: int, _df: pd.DataFrame):
    from randomness_tests import run_all, summary_table
    res = run_all(_df)
    return res, summary_table(res)


# ---------------------------------------------------------------------------
# Sidebar — data sources + sync
# ---------------------------------------------------------------------------

if 'cache_bust' not in st.session_state:
    st.session_state['cache_bust'] = 0

st.sidebar.header("Data sources")

last_sync = get_last_sync_info()
for op in ['damacai', 'magnum', 'fouryes3']:
    ts = last_sync.get(op, "never")
    st.sidebar.caption(f"• {op}: {ts}")

sync_choice = st.sidebar.selectbox("Sync from operator",
                                   ["damacai", "fouryes3", "magnum", "all"],
                                   index=0)
sync_recent = st.sidebar.slider("How many recent draws", 5, 200, 30, 5)

if st.sidebar.button("⬇ Sync latest results", use_container_width=True):
    from data_sources.manager import sync, sync_all
    with st.spinner(f"Syncing {sync_choice}..."):
        if sync_choice == "all":
            reports = sync_all(recent=sync_recent, verbose=False)
        else:
            reports = [sync(sync_choice, recent=sync_recent, verbose=False)]
    for r in reports:
        if r.available:
            st.sidebar.success(f"{r.operator}: +{r.fetched} new, {r.skipped_existing} already cached")
        else:
            st.sidebar.error(f"{r.operator}: {r.error or 'unavailable'}")
    st.session_state['cache_bust'] += 1
    st.cache_data.clear()

st.sidebar.divider()
include_ml = st.sidebar.checkbox("Use ML models (slower)", value=False,
                                  help="Adds XGBoost + LSTM. Same conclusion, longer wait.")
fouryes3_key = st.sidebar.text_input("4dyes3 API key (optional)", type="password",
                                      help="If 4dyes3 needs auth, paste your key here. Stays in this session only.")
if fouryes3_key:
    import os
    os.environ['FOURYES3_API_KEY'] = fouryes3_key


# Load data
with st.spinner("Loading historical draws..."):
    df_full, top3 = load_data(st.session_state['cache_bust'])
df_hash = len(top3)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_home, tab_check, tab_bet, tab_how, tab_adv = st.tabs([
    "🏠 Home",
    "🔍 Check My Number",
    "💰 Should I Bet?",
    "📚 How It Works",
    "⚙️ Advanced",
])


# --- Home -------------------------------------------------------------------
with tab_home:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total draws on file", f"{len(top3):,}")
    c2.metric("Oldest draw", str(top3['DrawDate'].min().date()))
    c3.metric("Latest draw", str(top3['DrawDate'].max().date()))

    st.subheader("🎯 Today's most-agreed-on numbers")
    st.caption("These are the 10 numbers that the most different methods rank highly. **This is not a prediction** — it's just where the methods agree. The Backtest in Advanced shows that agreement doesn't translate to wins.")

    with st.spinner("Crunching numbers..."):
        results = run_predictors(df_hash, top3, include_ml, 100)
        cons = build_consensus_table(results, 50)

    if cons.empty:
        st.warning("Couldn't compute consensus. Try clicking Sync above.")
    else:
        top10 = cons.head(10).reset_index(drop=True)
        max_agree = int(top10['agreement_count'].max())
        cols = st.columns(5)
        for i, row in top10.iterrows():
            col = cols[i % 5]
            shade = int(255 - (180 * row['agreement_count'] / max(max_agree, 1)))
            color = f"rgb(46, 134, 171, {0.3 + 0.7 * row['agreement_count'] / max(max_agree, 1)})"
            col.markdown(
                f"""
                <div style="background:{color};padding:18px;border-radius:10px;
                            text-align:center;color:white;margin-bottom:10px;">
                    <div style="font-size:30px;font-weight:700;font-family:monospace;">{row['number']}</div>
                    <div style="font-size:12px;opacity:0.9;">{int(row['agreement_count'])} methods agree</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Recent draws")
    st.dataframe(
        top3.sort_values('DrawDate', ascending=False).head(15)[
            ['DrawDate', '1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']
        ],
        use_container_width=True,
        hide_index=True,
    )


# --- Check My Number --------------------------------------------------------
with tab_check:
    st.subheader("Tell me about my number")
    guess = st.text_input("Enter a 4-digit number", value="1234", max_chars=4)

    if st.button("Analyze this number", type="primary", use_container_width=True):
        try:
            guess_clean = str(guess).strip().zfill(4)
            if len(guess_clean) != 4 or not guess_clean.isdigit():
                st.error("Please enter exactly 4 digits (0000–9999).")
            else:
                results = run_predictors(df_hash, top3, include_ml, 100)
                cons = build_consensus_table(results, 100)

                from user_guess_analyzer import analyze
                report = analyze(guess_clean, df_full, method_results=results, consensus_df=cons)

                c1, c2, c3 = st.columns(3)
                c1.metric("Total appearances", report.total_appearances,
                          help="How many times this exact number has appeared across all 23 prize positions, ever.")
                c2.metric("Permutation family", report.permutation_appearances,
                          help=f"How often any arrangement of digits {{{','.join(sorted(set(guess_clean)))}}} has appeared.")
                if report.last_seen is not None:
                    c3.metric("Last seen", f"{report.days_since_last} days ago",
                              help=str(report.last_seen.date()))
                else:
                    c3.metric("Last seen", "Never (top 3)")

                st.markdown("### What history says")
                if report.total_appearances == 0:
                    st.write(f"**{guess_clean} has never appeared** in any of the prize positions on record. That's not unusual — there are 10,000 possible numbers and many haven't come up yet.")
                else:
                    last_str = f"on {report.last_seen.date()}" if report.last_seen is not None else "(not in top 3 ever)"
                    st.write(
                        f"**{guess_clean} has appeared {report.total_appearances} times** across all 23 prize positions since 1985. "
                        f"It was last seen {last_str}. "
                        f"Any arrangement of these digits has come up **{report.permutation_appearances} times** in total."
                    )

                st.markdown("### What this means for tonight's draw")
                empirical = report.empirical_next_draw_hit_prob
                baseline = report.random_baseline
                st.write(
                    f"After each historical occurrence of {guess_clean}, the chance of it appearing in the very next draw "
                    f"has been **{empirical * 100:.2f}%**. The pure-random baseline is **{baseline * 100:.2f}%** "
                    f"({int(round(baseline * 10000))} prize slots out of 10,000)."
                )
                if abs(empirical - baseline) < baseline * 0.5:
                    st.success("That's right around what you'd expect from random chance — no detectable advantage.")
                elif empirical > baseline:
                    st.info(f"It's a touch above baseline, but with only {report.total_appearances} samples this is noise, not a pattern.")
                else:
                    st.info(f"It's a touch below baseline — also noise. Lottery numbers don't 'use up their luck'.")

                st.markdown("### Where it ranks across all methods")
                rank_rows = [
                    {'method': m, 'rank': str(r) if r is not None else "not in top 100"}
                    for m, r in report.method_ranks.items()
                ]
                with st.expander(
                    f"Method rankings ({report.agreement_count}/{len(report.method_ranks)} methods include it in top 100)"
                ):
                    if rank_rows:
                        st.dataframe(
                            pd.DataFrame(rank_rows).astype(str),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.write("_no method rankings available_")

                st.markdown("### If you bet RM1 on this number every draw")
                from ev_simulator import simulate
                with st.spinner("Simulating 1,000 draws..."):
                    sim = simulate([guess_clean], n_draws=1000, bet_per_number=1.0,
                                    starting_bankroll=10000.0, seed=42)
                c1, c2, c3 = st.columns(3)
                c1.metric("Net after 1,000 draws", f"RM {sim.net:.2f}")
                c2.metric("ROI", f"{sim.roi * 100:.1f}%")
                c3.metric("Number of wins", sum(sim.hits_per_tier.values()))

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=sim.pnl_curve, mode='lines', name='cumulative net (RM)',
                                          line=dict(color='#2E86AB')))
                fig.add_hline(y=0, line_dash="dash", line_color="grey")
                fig.update_layout(title=f"Cumulative profit/loss betting RM1 on {guess_clean} every draw",
                                  xaxis_title="draws played", yaxis_title="RM",
                                  height=350, margin=dict(t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)

                st.warning(
                    f"**Bottom line:** {guess_clean} carries {report.agreement_count}/{len(report.method_ranks)} methods of historical-pattern "
                    f"support — but that means nothing for the next draw. Don't bet expecting profit."
                )

        except Exception as e:
            st.error(f"Something went wrong: {e}")


# --- Should I Bet? ----------------------------------------------------------
with tab_bet:
    st.subheader("How much will I really lose?")
    st.caption("Run a realistic simulation of betting on 4D over time. Spoiler: the line goes down.")

    c1, c2 = st.columns(2)
    bet_amount = c1.number_input("RM per draw", min_value=1.0, value=10.0, step=1.0,
                                  help="How much you'd spend per draw, total.")
    num_draws = c2.slider("Number of draws", 100, 5000, 1000, 100,
                          help="Magnum/Damacai draws happen ~3 times a week.")

    c1, c2 = st.columns(2)
    strategy = c1.selectbox("How do you pick numbers?", [
        "Random numbers each draw",
        "Always the same 10 hot numbers",
        "Always the same 10 'overdue' numbers",
        "Today's top 10 consensus numbers",
    ])
    n_runs = c2.slider("Repeat the experiment N times", 1, 50, 10,
                        help="Each run picks fresh random draws so you see the spread.")

    if st.button("Run simulation", type="primary", use_container_width=True):
        from ev_simulator import simulate, theoretical_ev_per_rm1, house_edge

        if strategy.startswith("Random"):
            rng = np.random.default_rng()
            numbers = [f'{i:04d}' for i in rng.choice(10000, 10, replace=False)]
        elif "hot" in strategy:
            from predictor_suite import method_frequency
            numbers = method_frequency(top3, top_n=10).numbers(10)
        elif "overdue" in strategy:
            from predictor_suite import method_overdue
            numbers = method_overdue(top3, top_n=10).numbers(10)
        else:
            results = run_predictors(df_hash, top3, include_ml, 100)
            cons = build_consensus_table(results, 50)
            numbers = cons.head(10)['number'].tolist()

        per_number_bet = bet_amount / len(numbers)
        st.write(f"Betting RM{per_number_bet:.2f} on each of these 10 numbers: `{numbers}`")

        nets, curves = [], []
        with st.spinner(f"Running {n_runs} simulations of {num_draws} draws each..."):
            for i in range(n_runs):
                res = simulate(numbers, n_draws=num_draws, bet_per_number=per_number_bet,
                               starting_bankroll=10000.0, seed=i)
                nets.append(res.net)
                curves.append(res.pnl_curve)

        fig = go.Figure()
        for i, curve in enumerate(curves):
            fig.add_trace(go.Scatter(y=curve, mode='lines', opacity=0.5,
                                      line=dict(color='#2E86AB'),
                                      showlegend=False))
        fig.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="break even")
        fig.update_layout(title="Cumulative profit/loss per simulated run",
                          xaxis_title="draws played", yaxis_title="RM",
                          height=420, margin=dict(t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

        avg_loss = float(np.mean(nets))
        median_loss = float(np.median(nets))
        worst = float(np.min(nets))
        best = float(np.max(nets))

        c1, c2, c3 = st.columns(3)
        c1.metric("Average outcome", f"RM {avg_loss:.0f}",
                  delta=f"{avg_loss / (bet_amount * num_draws) * 100:.1f}% return")
        c2.metric("Best outcome (lucky run)", f"RM {best:.0f}")
        c3.metric("Worst outcome", f"RM {worst:.0f}")

        st.markdown(f"""
        ### Plain-English summary
        - You'd spend a total of **RM {bet_amount * num_draws:,.0f}** across {num_draws:,} draws.
        - On average across {n_runs} runs, you'd lose **RM {-avg_loss:,.0f}**.
        - The lottery's theoretical edge is **{house_edge() * 100:.1f}%**, meaning you keep about
          **RM{theoretical_ev_per_rm1():.2f}** for every **RM1** you bet, on average.
        - Even your *luckiest* simulated run ended with **RM {best:,.0f}** net.
        """)
        if avg_loss < 0:
            st.error("The chart trends down because the math says it must. There is no strategy here that flips a negative-EV game.")


# --- How It Works -----------------------------------------------------------
with tab_how:
    st.subheader("How this works, in plain English")

    st.markdown("""
**1. The lottery is random.**
Every draw, the operator picks numbers from a machine designed to be unpredictable.
"Random" has a precise mathematical meaning: knowing every past draw tells you
**nothing** about the next one. We test this on the data — see the randomness
section in **Advanced**.

**2. Why we built this anyway.**
The data is interesting even if it isn't predictive. You can ask things like:
- Has my favorite number ever won?
- Which numbers are statistically "hot" right now?
- If different methods agree, is that meaningful?

This app answers all of those — honestly.

**3. The methods.**
We run 10 different statistical and machine-learning methods over 40+ years of draws.
They each rank the 10,000 possible 4-digit numbers using their own logic
(frequency, gap, neural network, etc.). Some methods agree on "good" numbers.
That's just what the **methods think look good in the past** — it has nothing
to do with what the next draw will be.

**4. The backtest.**
The most important number in this app is in **Advanced → Backtest**: when we
test all these methods on past data they hadn't seen, **none of them beat
random chance**. That's not because we built it badly. That's because the
data is genuinely random.

**5. The math of losing.**
A Magnum 4D "Big" bet pays out:
- RM 2,500 for 1st prize (1 number per draw)
- RM 1,000 for 2nd, RM 500 for 3rd
- RM 180 each for the 10 "Special" prizes
- RM 60 each for the 10 "Consolation" prizes

Multiply the payout by the chance of each (1/10,000 per slot) and you get
**RM 0.64 expected return for every RM 1 bet**. The lottery keeps **36 sen
of every ringgit** on average. That's the house edge. No strategy beats it
because nothing in your strategy changes the payout-to-probability ratio.

**6. Use this app like a museum, not a casino.**
Click around. Check your favorite number. Watch the simulator slowly drain
a fictional RM10,000. Have fun with the data. Just don't change your
betting habits because of anything you see here.
""")


# --- Advanced ---------------------------------------------------------------
with tab_adv:
    st.subheader("Power-user view")

    with st.expander("Per-method top numbers"):
        results = run_predictors(df_hash, top3, include_ml, 100)
        for name, res in results.items():
            with st.container():
                st.markdown(f"**{name}** _({res.family})_ — {res.explanation}")
                if res.top:
                    st.dataframe(pd.DataFrame(res.top[:30], columns=['number', 'score']),
                                  use_container_width=True, hide_index=True, height=200)
                else:
                    st.write(f"_no results: {res.explanation}_")

    with st.expander("Full consensus table (all numbers)"):
        results = run_predictors(df_hash, top3, include_ml, 100)
        cons = build_consensus_table(results, 100)
        st.dataframe(cons, use_container_width=True, hide_index=True, height=420)

    with st.expander("Randomness tests"):
        raw, table = run_randomness(df_hash, top3)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("With ~30 years of data, even a 0.5% bias is detectable. The data passes; it's genuinely random.")

    with st.expander("Walk-forward backtest"):
        c1, c2 = st.columns(2)
        years_str = c1.text_input("Years to backtest (e.g. 2020-2023)", value="2020-2023")
        bt_top_n = c2.slider("Top-N per method", 20, 300, 100, 10)
        if st.button("Run backtest", key='adv_bt'):
            try:
                start, end = (int(s) for s in years_str.split('-'))
                years = list(range(start, end + 1))
                from backtest import run as run_bt, summary
                with st.spinner(f"Backtesting {years}..."):
                    bt = run_bt(top3, years, top_n=bt_top_n, include_ml=False)
                st.dataframe(bt, use_container_width=True, hide_index=True)
                s = summary(bt)
                st.dataframe(s, use_container_width=True, hide_index=True)
                n_sig = int(bt['significant'].sum()) if 'significant' in bt.columns else 0
                if n_sig == 0:
                    st.success("No method beat random after Bonferroni. As predicted by i.i.d. theory.")
                else:
                    st.info(f"{n_sig} cells flagged 'significant' — likely multiple-testing artefact.")
            except Exception as e:
                st.error(f"Error: {e}")

    with st.expander("Data sources detail"):
        last = get_last_sync_info()
        st.json(last or {"info": "no sync performed yet"})
        st.caption("Sync from the sidebar. Damacai uses Hong's public endpoint. 4dyes3 is config-driven via `data_sources/fouryes3_config.yaml`.")
