# ======================================================================
# AURELINE LABS — DASHBOARD v2.0
# Quantitative Research & Intelligence Platform
# Ateneo de Manila University · Applied Mathematics · Math Finance
# ======================================================================

# SECTION 1: IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import os
import sys
import glob
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.data_handler import get_price_data
from src.indicators import add_sma, add_atr
from src.risk import calculate_shares
from src.features import build_all_features as build_features
from src.features import all_feature_cols as feature_cols


# ======================================================================
# SECTION 2: PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Aureline Labs",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================================
# SECTION 3: VISUAL IDENTITY — CSS
# ======================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
.stApp { background-color:#060d1f; font-family:'DM Sans',sans-serif; color:#e8f0fe; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color:#0d1b35 !important; border-right:1px solid #1a3357; }
[data-testid="stSidebar"] * { color:#e8f0fe !important; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    background-color:#060d1f !important; border:1px solid #1a3357 !important;
    color:#e8f0fe !important; font-family:'JetBrains Mono',monospace !important;
    font-size:0.8rem !important; border-radius:4px;
}
[data-testid="stSidebar"] .stTextInput input:focus { border-color:#00d4aa !important; }
[data-testid="stSidebar"] hr { border-color:#1a3357 !important; }

/* ── Typography ── */
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; font-weight:700 !important; color:#e8f0fe !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color:#0f2040; border:1px solid #1a3357;
    border-top:2px solid #00d4aa; border-radius:6px; padding:16px 20px 12px 20px;
}
[data-testid="stMetricLabel"] {
    font-family:'JetBrains Mono',monospace !important; font-size:0.62rem !important;
    color:#7b9bc0 !important; text-transform:uppercase !important; letter-spacing:0.12em !important;
}
[data-testid="stMetricValue"] {
    font-family:'JetBrains Mono',monospace !important; font-size:1.3rem !important;
    font-weight:700 !important; color:#00d4aa !important;
}
[data-testid="stMetricDelta"] { font-family:'JetBrains Mono',monospace !important; font-size:0.72rem !important; }

/* ── Section labels — signature element ── */
.al-section {
    font-family:'JetBrains Mono',monospace; font-size:0.62rem; font-weight:500;
    color:#00d4aa; text-transform:uppercase; letter-spacing:0.2em;
    padding-bottom:6px; border-bottom:1px solid #00d4aa;
    margin-bottom:14px; display:block;
}

/* ── Divider ── */
hr { border:none !important; border-top:1px solid #1a3357 !important; margin:16px 0 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color:#0d1b35; border:1px solid #1a3357;
    border-radius:6px; padding:4px; gap:2px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family:'JetBrains Mono',monospace; font-size:0.72rem;
    font-weight:500; color:#7b9bc0; border-radius:4px;
}
[data-testid="stTabs"] [aria-selected="true"] { background-color:#00d4aa !important; color:#060d1f !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border:1px solid #1a3357 !important; border-radius:6px; }

/* ── Buttons ── */
.stButton button {
    background-color:#00d4aa !important; color:#060d1f !important; border:none !important;
    font-family:'JetBrains Mono',monospace !important; font-weight:700 !important;
    font-size:0.78rem !important; letter-spacing:0.05em !important;
    border-radius:4px !important; padding:10px 24px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background-color:#0f2040 !important; border:1px solid #1a3357 !important;
    border-left:3px solid #00d4aa !important; border-radius:4px !important;
    color:#7b9bc0 !important; font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important;
}

/* ── Radio (mode toggle) ── */
[data-testid="stRadio"] label p { font-family:'JetBrains Mono',monospace !important; font-size:0.75rem !important; }

/* ── Beginner card ── */
.bcard {
    background:#0f2040; border:1px solid #1a3357; border-radius:8px;
    padding:18px; margin-bottom:10px; font-family:'DM Sans',sans-serif;
}
.bcard-label { font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#7b9bc0; text-transform:uppercase; letter-spacing:0.1em; }
.bcard-value { font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700; color:#00d4aa; margin:4px 0; }
.bcard-desc { font-size:0.8rem; color:#7b9bc0; line-height:1.5; }

/* ── News headline row ── */
.news-row {
    display:flex; align-items:center; gap:10px; padding:8px 12px;
    background:#0f2040; border:1px solid #1a3357; border-radius:4px;
    margin-bottom:4px;
}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# SECTION 4: HELPERS
# ======================================================================
def section(label):
    st.markdown(f'<span class="al-section">{label}</span>', unsafe_allow_html=True)


def aureline_chart_style(ax, fig):
    fig.patch.set_facecolor("#0f2040")
    ax.set_facecolor("#060d1f")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1a3357")
    ax.spines["bottom"].set_color("#1a3357")
    ax.tick_params(colors="#7b9bc0", labelsize=8)
    ax.yaxis.label.set_color("#7b9bc0")
    ax.xaxis.label.set_color("#7b9bc0")
    ax.grid(True, color="#0d1b35", linewidth=0.8, linestyle="--")


# ── Beginner translation layer ──
def plain_signal(signal):
    return {
        "BUY":          "✅ Worth considering",
        "HOLD":         "⏸️ Wait and watch",
        "AVOID":        "⛔ Too risky right now",
        "STRONG BUY":   "🟢 Strong opportunity",
        "STRONG AVOID": "🔴 Avoid — high risk",
        "WATCH ▲":      "👀 Positive news, mixed price signal",
        "WATCH ▼":      "👀 Negative news, watch carefully",
    }.get(signal, signal)


def plain_regime(regime):
    return {
        "BULL_TRENDING":   "📈 Rising steadily",
        "BEAR_TRENDING":   "📉 Falling trend",
        "HIGH_VOLATILITY": "⚠️ Very choppy — unpredictable",
        "SIDEWAYS":        "➡️ No clear direction",
    }.get(regime, regime)


def plain_sharpe(s):
    if s >= 2.0:   return f"Sharpe {s:.2f} — Excellent (very consistent returns)"
    elif s >= 1.0: return f"Sharpe {s:.2f} — Good risk-adjusted performance"
    elif s >= 0.5: return f"Sharpe {s:.2f} — Acceptable, some ups and downs"
    elif s >= 0.0: return f"Sharpe {s:.2f} — Weak"
    else:          return f"Sharpe {s:.2f} — Negative (losing on risk-adjusted basis)"


def plain_drawdown(mdd_pct):
    p = abs(mdd_pct)
    if p < 5:    return f"Worst decline: {mdd_pct:.1f}% — Very low risk"
    elif p < 15: return f"Worst decline: {mdd_pct:.1f}% — Moderate risk"
    elif p < 30: return f"Worst decline: {mdd_pct:.1f}% — High risk"
    else:        return f"Worst decline: {mdd_pct:.1f}% — Very high risk"


def plain_return(ret_pct):
    if ret_pct > 100:  return "More than doubled", "Great long-term growth"
    elif ret_pct > 50: return f"+{ret_pct:.1f}%", "Strong gains"
    elif ret_pct > 20: return f"+{ret_pct:.1f}%", "Good gains"
    elif ret_pct > 0:  return f"+{ret_pct:.1f}%", "Small gain"
    elif ret_pct > -20:return f"{ret_pct:.1f}%", "Small loss"
    else:               return f"{ret_pct:.1f}%", "Significant loss"


# ── Metric helpers ──
def sharpe_ratio(pv, rfr=0.05):
    s  = pd.Series(pv)
    dr = s.pct_change().dropna()
    ex = dr - rfr/252
    return round((ex.mean()/ex.std())*np.sqrt(252), 3) if ex.std() > 0 else 0.0


def max_drawdown(pv):
    s = pd.Series(pv)
    return ((s - s.cummax()) / s.cummax()).min()


def cagr(pv, start, end):
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    return (pv[-1]/pv[0])**(1/years) - 1 if years > 0 else 0.0


def win_rate(pv):
    return (pd.Series(pv).pct_change().dropna() > 0).mean()


def load_paper_account():
    p = "data/paper_account.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"cash": 100000, "positions": {}, "orders": []}


# ======================================================================
# SECTION 5: CACHED DATA + BACKTEST
# ======================================================================
@st.cache_data
def run_sma_backtest(ticker, start, end, sma_window=20):
    data = get_price_data(ticker, start, end)
    if data is None or data.empty:
        return None, None, None
    data = add_sma(data, window=sma_window)
    data = add_atr(data)
    data["signal"] = (data["Close"] > data["sma"]).shift(1)
    cash = 10000; shares = 0; portfolio = []
    for i in range(len(data)):
        pt = data["Close"].iloc[i]
        py = data["Close"].iloc[i-1] if i > 0 else pt
        sig = data["signal"].iloc[i]
        atr = data["atr"].iloc[i]
        if sig and shares == 0:
            sh = calculate_shares(cash, py, atr)
            if sh > 0:
                cash -= sh * py * 1.001; shares = sh
        elif not sig and shares > 0:
            cash += shares * py * 0.999; shares = 0
        portfolio.append(cash + shares * pt)
    bh = (10000 // data["Close"].iloc[0] * data["Close"]).tolist()
    return data, portfolio, bh


@st.cache_data
def run_multi_ticker(tickers, start, end, sma_window):
    results = []
    for t in tickers:
        data, portfolio, bh = run_sma_backtest(t, start, end, sma_window)
        if portfolio is None:
            continue
        results.append({
            "Ticker":     t,
            "Return":     (portfolio[-1]/10000) - 1,
            "CAGR":       cagr(portfolio, start, end),
            "Sharpe":     sharpe_ratio(portfolio),
            "Max DD":     max_drawdown(portfolio),
            "Win Rate":   win_rate(portfolio),
            "B&H Return": (bh[-1]/10000) - 1,
            "Beat B&H":   "✓" if portfolio[-1] > bh[-1] else "✗",
            "portfolio":  portfolio,
        })
    return results


@st.cache_data
def run_ml_backtest(ticker, start, end):
    from sklearn.ensemble import RandomForestClassifier
    data = get_price_data(ticker, start, end)
    if data is None or data.empty:
        return None, None
    data = add_atr(data)
    df   = build_features(data)
    cols = feature_cols()
    df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna(subset=cols + ["label"])
    if len(df) < 600:
        return None, None
    signals = []
    for i in range(500, len(df)-1):
        tr = df.iloc[:i]
        m  = RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=20, random_state=42)
        m.fit(tr[cols], tr["label"])
        p = m.predict_proba(df[cols].iloc[[i]])[0][1]
        signals.append({"date": df.index[i], "signal": 1 if p >= 0.55 else 0})
    sdf = pd.DataFrame(signals).set_index("date")
    btd = df.loc[sdf.index]; data2 = add_atr(data).loc[sdf.index]
    cash = 10000; sh = 0; pv = []
    for i in range(len(btd)):
        pt = btd["Close"].iloc[i]; py = btd["Close"].iloc[i-1] if i > 0 else pt
        atr = data2["atr"].iloc[i]; sig = sdf["signal"].iloc[i]
        if sig and sh == 0:
            s = calculate_shares(cash, py, atr)
            if s > 0: cash -= s*py*1.001; sh = s
        elif not sig and sh > 0:
            cash += sh*py*0.999; sh = 0
        pv.append(cash + sh*pt)
    return btd.index, pv


# ======================================================================
# SECTION 6: SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 24px 0;'>
        <div style='font-family:Space Grotesk,sans-serif; font-size:1.25rem;
                    font-weight:700; color:#e8f0fe; letter-spacing:-0.5px;'>
            ⬡ Aureline Labs
        </div>
        <div style='font-family:JetBrains Mono,monospace; font-size:0.58rem;
                    color:#00d4aa; letter-spacing:0.2em; margin-top:5px;'>
            QUANT RESEARCH PLATFORM v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mode toggle — first class
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.6rem;
                color:#7b9bc0; letter-spacing:0.15em; margin-bottom:6px;'>
        INTERFACE MODE
    </div>
    """, unsafe_allow_html=True)
    mode        = st.radio("Mode", ["Professional", "Beginner"],
                           horizontal=True, label_visibility="collapsed")
    is_beginner = (mode == "Beginner")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.6rem;
                color:#7b9bc0; letter-spacing:0.15em; margin-bottom:8px;'>
        STRATEGY CONTROLS
    </div>""", unsafe_allow_html=True)

    ticker     = st.text_input("Ticker", value="AAPL").upper()
    start_date = st.date_input("Start", value=pd.Timestamp("2021-01-01")).strftime("%Y-%m-%d")
    end_date   = st.date_input("End",   value=pd.Timestamp("2026-06-01")).strftime("%Y-%m-%d")
    sma_window = st.slider("SMA Window", 5, 100, 20)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.6rem;
                color:#7b9bc0; letter-spacing:0.15em; margin-bottom:8px;'>
        WATCHLIST
    </div>""", unsafe_allow_html=True)
    ticker_input = st.text_area("Tickers (comma-separated)",
                                 value="AAPL,MSFT,NVDA,JPM,XOM,TSLA,SPY",
                                 height=68)
    watchlist = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.62rem;
                color:#1a3357; line-height:2;'>
        <span style='color:#00d4aa;'>RESEARCHER</span><br>
        Louis Andre<br>
        <span style='color:#00d4aa;'>INSTITUTION</span><br>
        Ateneo de Manila University<br>
        <span style='color:#00d4aa;'>MODE</span><br>
        {'Beginner — plain English' if is_beginner else 'Professional — full quant metrics'}
    </div>""", unsafe_allow_html=True)


# ======================================================================
# SECTION 7: HEADER
# ======================================================================
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown(f"""
    <div style='padding-top:8px;'>
        <div style='font-family:Space Grotesk,sans-serif; font-size:2.2rem;
                    font-weight:700; color:#e8f0fe; letter-spacing:-1.5px;
                    line-height:1;'>Aureline Labs</div>
        <div style='font-family:JetBrains Mono,monospace; font-size:0.72rem;
                    color:#7b9bc0; margin-top:5px;'>
            {'Financial Intelligence — Simplified · For the everyday investor' if is_beginner
             else 'Quantitative Research & Intelligence Platform · Applied Mathematics'}
        </div>
    </div>""", unsafe_allow_html=True)
with hcol2:
    st.markdown(f"""
    <div style='display:flex; gap:6px; justify-content:flex-end;
                padding-top:14px; flex-wrap:wrap;'>
        <span style='font-family:JetBrains Mono,monospace; font-size:0.58rem;
                     font-weight:700; color:#060d1f; background:#00d4aa;
                     padding:3px 8px; border-radius:3px; letter-spacing:0.1em;'>
            PAPER TRADING
        </span>
        <span style='font-family:JetBrains Mono,monospace; font-size:0.58rem;
                     color:#7b9bc0; border:1px solid #1a3357;
                     padding:3px 8px; border-radius:3px; letter-spacing:0.1em;'>
            v2.0
        </span>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div style='font-family:JetBrains Mono,monospace; font-size:0.72rem;
            color:#1a3357; margin:8px 0 0 0; padding-bottom:16px;
            border-bottom:1px solid #1a3357;'>
    <span style='color:#7b9bc0;'>{ticker}</span>
    &nbsp;·&nbsp; {start_date} → {end_date}
    &nbsp;·&nbsp; SMA({sma_window})
    &nbsp;·&nbsp; <span style='color:#00d4aa;'>{mode} Mode</span>
</div>""", unsafe_allow_html=True)
st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)


# ======================================================================
# SECTION 8: TABS
# ======================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "OVERVIEW",
    "RESEARCH",
    "ML vs SMA",
    "🧠 INTELLIGENCE",
    "🏢 COMPANY",
    "💬 COPILOT",
    "📊 PORTFOLIO"
])

# ======================================================================
# TAB 1: OVERVIEW (replaces "Single Ticker")
# ======================================================================
with tab1:
    with st.spinner(f"Loading {ticker}..."):
        data, portfolio, bh = run_sma_backtest(ticker, start_date, end_date, sma_window)

    if data is None:
        st.error(f"No data found for **{ticker}**. Check the ticker symbol.")
    else:
        latest  = data["Close"].iloc[-1]
        final   = portfolio[-1]
        ret     = (final/10000) - 1
        sh      = sharpe_ratio(portfolio)
        mdd     = max_drawdown(portfolio)
        bh_ret  = (bh[-1]/10000) - 1
        _cagr   = cagr(portfolio, start_date, end_date)
        mdd_pct = mdd * 100

        # ── Metrics row ──
        if is_beginner:
            ret_val, ret_desc = plain_return(ret * 100)
            risk_label = ("Low" if abs(mdd_pct) < 15
                          else "Medium" if abs(mdd_pct) < 30 else "High")
            risk_color = ("#00d4aa" if risk_label == "Low"
                          else "#ffd166" if risk_label == "Medium" else "#ff4d6a")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""
            <div class='bcard'>
                <div class='bcard-label'>Current Price</div>
                <div class='bcard-value'>${latest:.2f}</div>
                <div class='bcard-desc'>Latest closing price of {ticker}</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""
            <div class='bcard'>
                <div class='bcard-label'>Your Strategy Result</div>
                <div class='bcard-value' style='color:{"#00d4aa" if ret > 0 else "#ff4d6a"};'>{ret_val}</div>
                <div class='bcard-desc'>{ret_desc} on ₱10,000 invested → ₱{final:,.0f}</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""
            <div class='bcard'>
                <div class='bcard-label'>Risk Level</div>
                <div class='bcard-value' style='color:{risk_color};'>{risk_label}</div>
                <div class='bcard-desc'>{plain_drawdown(mdd_pct)}</div>
            </div>""", unsafe_allow_html=True)

            # Plain English summary box
            st.markdown(f"""
            <div style='background:#0f2040; border:1px solid #1a3357;
                        border-left:3px solid #00d4aa; border-radius:6px;
                        padding:16px 20px; margin:12px 0;
                        font-family:DM Sans,sans-serif;'>
                <div style='font-size:0.68rem; color:#00d4aa; font-weight:600;
                            letter-spacing:0.1em; margin-bottom:8px;'>
                    PLAIN ENGLISH SUMMARY
                </div>
                <div style='font-size:0.88rem; color:#e8f0fe; line-height:1.7;'>
                    If you had put <strong>₱10,000</strong> into <strong>{ticker}</strong>
                    at the start of this period using this moving-average strategy,
                    you would now have <strong style='color:#00d4aa;'>₱{final:,.0f}</strong>.
                    Just buying and holding the stock the whole time would have given you
                    <strong>₱{bh[-1]:,.0f}</strong>.
                </div>
                <div style='font-size:0.78rem; color:#7b9bc0; margin-top:8px; line-height:1.6;'>
                    {plain_sharpe(sh)} · {plain_drawdown(mdd_pct)}
                </div>
            </div>""", unsafe_allow_html=True)

        else:
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Latest Price",  f"${latest:.2f}")
            c2.metric("Final Value",   f"${final:,.2f}")
            c3.metric("Total Return",  f"{ret:+.2%}", delta=f"{(ret-bh_ret):+.2%} vs B&H")
            c4.metric("CAGR",          f"{_cagr:+.2%}")
            c5.metric("Sharpe Ratio",  f"{sh:.3f}")
            c6.metric("Max Drawdown",  f"{mdd:.2%}")

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

        # ── Price chart ──
        section("PRICE HISTORY & SMA SIGNAL")
        fig1, ax1 = plt.subplots(figsize=(13, 3.5))
        aureline_chart_style(ax1, fig1)
        ax1.plot(data.index, data["Close"], color="#00d4aa", linewidth=1.3,
                 label="Close Price")
        ax1.plot(data.index, data["sma"], color="#1a6eff", linewidth=1,
                 linestyle="--", alpha=0.7, label=f"SMA({sma_window})")
        signal = data["signal"].fillna(False).astype(bool)
        ax1.fill_between(data.index, data["Close"].min(), data["Close"].max(),
                         where=signal, alpha=0.05, color="#00d4aa",
                         label="In Market")
        ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        ax1.legend(fontsize=8, facecolor="#0f2040", edgecolor="#1a3357",
                   labelcolor="#7b9bc0")
        plt.tight_layout()
        st.pyplot(fig1); plt.close()

        # ── Performance chart ──
        section("STRATEGY vs BENCHMARK")
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})
        aureline_chart_style(ax2, fig2); aureline_chart_style(ax3, fig2)
        ax2.plot(data.index, portfolio, color="#00d4aa", linewidth=1.5,
                 label="SMA Strategy")
        ax2.plot(data.index, bh, color="#1a6eff", linewidth=1.2,
                 linestyle="--", alpha=0.6, label="Buy & Hold")
        ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        ax2.set_ylabel("Portfolio Value", fontsize=8, color="#7b9bc0")
        ax2.legend(fontsize=8, facecolor="#0f2040", edgecolor="#1a3357",
                   labelcolor="#7b9bc0")
        ps = pd.Series(portfolio); bs = pd.Series(bh)
        ax3.fill_between(data.index, ((ps-ps.cummax())/ps.cummax())*100, 0,
                         alpha=0.7, color="#00d4aa", label="Strategy DD")
        ax3.fill_between(data.index, ((bs-bs.cummax())/bs.cummax())*100, 0,
                         alpha=0.3, color="#1a6eff", label="B&H DD")
        ax3.set_ylabel("Drawdown %", fontsize=8, color="#7b9bc0")
        ax3.legend(fontsize=8, facecolor="#0f2040", edgecolor="#1a3357",
                   labelcolor="#7b9bc0")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

        # ── Paper account ──
        st.markdown("<hr>", unsafe_allow_html=True)
        section("PAPER TRADING ACCOUNT")
        acct = load_paper_account()
        a1, a2, a3 = st.columns(3)
        a1.metric("Cash Available",  f"${acct['cash']:,.2f}")
        a2.metric("Open Positions",   len(acct["positions"]))
        a3.metric("Orders Placed",    len(acct["orders"]))
        if acct["positions"]:
            pos_df = pd.DataFrame([
                {"Ticker": k, "Shares": v["qty"],
                 "Avg Entry": f"${v['avg_price']:.2f}"}
                for k, v in acct["positions"].items()
            ])
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        if st.checkbox("Show raw OHLCV data"):
            st.dataframe(data[["Open","High","Low","Close","Volume","sma","atr"]]
                         .tail(20).round(2), use_container_width=True)


# ======================================================================
# TAB 2: RESEARCH (replaces "Multi-Ticker")
# ======================================================================
with tab2:
    section("WATCHLIST STRATEGY COMPARISON")
    with st.spinner("Running multi-ticker backtest..."):
        multi = run_multi_ticker(tuple(watchlist), start_date, end_date, sma_window)

    if not multi:
        st.error("No results. Check watchlist tickers.")
    else:
        if is_beginner:
            st.markdown("""
            <div style='background:#0f2040; border:1px solid #1a3357;
                        border-left:3px solid #00d4aa; border-radius:6px;
                        padding:14px 18px; margin-bottom:16px;
                        font-family:DM Sans,sans-serif; font-size:0.85rem;
                        color:#7b9bc0; line-height:1.6;'>
                This table shows how a simple moving-average trading strategy
                performed on each stock. <strong style='color:#e8f0fe;'>Return</strong>
                is how much money you'd have made. <strong style='color:#e8f0fe;'>
                Sharpe</strong> tells you how smooth the returns were
                (higher = better). <strong style='color:#e8f0fe;'>Max Drawdown</strong>
                is the biggest loss from peak to trough.
            </div>""", unsafe_allow_html=True)

        disp_df = pd.DataFrame([{
            "Ticker":    r["Ticker"],
            "Return":    f"{r['Return']:+.2%}",
            "CAGR":      f"{r['CAGR']:+.2%}",
            "Sharpe":    f"{r['Sharpe']:.3f}",
            "Max DD":    f"{r['Max DD']:.2%}",
            "Win Rate":  f"{r['Win Rate']:.1%}",
            "B&H Return":f"{r['B&H Return']:+.2%}",
            "Beat B&H":  r["Beat B&H"]
        } for r in multi])
        st.dataframe(disp_df, use_container_width=True, hide_index=True)

        beat       = sum(1 for r in multi if r["Return"] > r["B&H Return"])
        avg_sharpe = np.mean([r["Sharpe"] for r in multi])
        m1, m2, m3 = st.columns(3)
        m1.metric("Beat Buy & Hold",  f"{beat}/{len(multi)}")
        m2.metric("Avg Sharpe Ratio", f"{avg_sharpe:.3f}")
        m3.metric("Tickers Analyzed", len(multi))

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        section("NORMALIZED PORTFOLIO CURVES")
        colors_palette = ["#00d4aa","#1a6eff","#ffd166","#ff4d6a",
                          "#7b61ff","#06d6a0","#118ab2","#8ed4b4"]
        fig3, ax3 = plt.subplots(figsize=(13, 5))
        aureline_chart_style(ax3, fig3)
        for idx, r in enumerate(multi):
            norm = [v/10000 for v in r["portfolio"]]
            ax3.plot(norm, color=colors_palette[idx % len(colors_palette)],
                     linewidth=1.5, label=r["Ticker"], alpha=0.9)
        ax3.axhline(y=1.0, color="#1a3357", linestyle="--", linewidth=1)
        ax3.set_ylabel("Return Multiple (×)", fontsize=8, color="#7b9bc0")
        ax3.set_xlabel("Trading Days", fontsize=8, color="#7b9bc0")
        ax3.legend(fontsize=8, facecolor="#0f2040", edgecolor="#1a3357",
                   labelcolor="#7b9bc0", ncol=2)
        plt.tight_layout(); st.pyplot(fig3); plt.close()


# ======================================================================
# TAB 3: ML vs SMA
# ======================================================================
with tab3:
    section(f"WALK-FORWARD ML MODEL vs SMA({sma_window}) — {ticker}")

    if is_beginner:
        st.markdown("""
        <div style='background:#0f2040; border:1px solid #1a3357;
                    border-left:2px solid #00d4aa; border-radius:4px;
                    padding:14px 18px; margin-bottom:16px;
                    font-family:DM Sans,sans-serif; font-size:0.85rem;
                    color:#7b9bc0; line-height:1.6;'>
            This test compares a machine learning model against a simple
            moving-average rule. The ML model studies past patterns to
            decide when to buy and sell — it's like having a research
            assistant that reads the history books before making a call.
            <strong style='color:#e8f0fe;'>A higher Sharpe ratio means
            smoother, more consistent returns.</strong>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Walk-forward training fits a new model at every time step "
                "using only past data — ~800 iterations. "
                "Results are cached after the first run.")

    if st.button("▶  RUN ML BACKTEST"):
        with st.spinner("Running walk-forward ML training..."):
            ml_dates, ml_pv = run_ml_backtest(ticker, start_date, end_date)

        if ml_pv is None:
            st.error("Not enough data for ML backtest. Try a longer date range.")
        else:
            _, sma_pv, _ = run_sma_backtest(ticker, start_date, end_date, sma_window)
            sma_aligned  = sma_pv[-len(ml_pv):]

            ml_ret  = (ml_pv[-1]/10000)  - 1
            sma_ret = (sma_aligned[-1]/10000) - 1
            ml_sh   = sharpe_ratio(ml_pv)
            sma_sh  = sharpe_ratio(sma_aligned)
            ml_mdd  = max_drawdown(ml_pv)
            sma_mdd = max_drawdown(sma_aligned)

            section("HEAD-TO-HEAD METRICS")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""<div style='font-family:JetBrains Mono,monospace;
                    font-size:0.65rem; color:#00d4aa; letter-spacing:0.15em;
                    margin-bottom:8px;'>ML MODEL</div>""",
                    unsafe_allow_html=True)
                st.metric("Return",       f"{ml_ret:+.2%}")
                st.metric("Sharpe Ratio", f"{ml_sh:.3f}")
                st.metric("Max Drawdown", f"{ml_mdd:.2%}")
                if is_beginner:
                    st.markdown(f"""
                    <div class='bcard' style='margin-top:10px;'>
                        <div class='bcard-label'>What this means</div>
                        <div class='bcard-desc'>{plain_sharpe(ml_sh)}</div>
                    </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div style='font-family:JetBrains Mono,monospace;
                    font-size:0.65rem; color:#1a6eff; letter-spacing:0.15em;
                    margin-bottom:8px;'>SMA({sma_window}) BASELINE</div>""",
                    unsafe_allow_html=True)
                st.metric("Return",       f"{sma_ret:+.2%}",
                          delta=f"{ml_ret-sma_ret:+.2%} ML edge")
                st.metric("Sharpe Ratio", f"{sma_sh:.3f}",
                          delta=f"{ml_sh-sma_sh:+.3f} ML edge")
                st.metric("Max Drawdown", f"{sma_mdd:.2%}",
                          delta=f"{ml_mdd-sma_mdd:+.2%} ML edge")

            st.markdown("<div style='margin-top:20px;'></div>",
                        unsafe_allow_html=True)
            section("EQUITY CURVES")
            fig4, ax4 = plt.subplots(figsize=(13, 4))
            aureline_chart_style(ax4, fig4)
            ax4.plot(ml_pv,      color="#00d4aa", linewidth=1.5, label="ML Strategy")
            ax4.plot(sma_aligned, color="#1a6eff", linewidth=1.2,
                     linestyle="--", alpha=0.7, label=f"SMA({sma_window})")
            ax4.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
            ax4.set_ylabel("Portfolio Value", fontsize=8, color="#7b9bc0")
            ax4.legend(fontsize=8, facecolor="#0f2040", edgecolor="#1a3357",
                       labelcolor="#7b9bc0")
            plt.tight_layout(); st.pyplot(fig4); plt.close()
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px;
                    font-family:JetBrains Mono,monospace; font-size:0.78rem;
                    color:#1a3357; background:#0f2040; border:1px solid #1a3357;
                    border-radius:6px;'>
            Press RUN ML BACKTEST to begin walk-forward analysis.
        </div>""", unsafe_allow_html=True)


# ======================================================================
# TAB 4: INTELLIGENCE HUB
# ======================================================================
with tab4:

    # ── Morning Brief ──
    section("MORNING INTELLIGENCE BRIEF")
    brief_files = sorted(glob.glob("experiments/reports/MORNING_BRIEF_*.md"),
                          reverse=True)
    daily_files = sorted(glob.glob("experiments/reports/DAILY_BRIEF_*.md"),
                          reverse=True)
    all_briefs  = brief_files + daily_files

    if all_briefs:
        latest_brief = all_briefs[0]
        brief_date   = Path(latest_brief).stem.replace(
            "MORNING_BRIEF_", "").replace("DAILY_BRIEF_", "")
        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace; font-size:0.72rem;
                    color:#7b9bc0; margin-bottom:12px;'>
            Latest: {brief_date} · {len(all_briefs)} briefs in archive
        </div>""", unsafe_allow_html=True)
        with st.expander("📋 View Full Morning Brief", expanded=True):
            with open(latest_brief) as f:
                st.markdown(f.read())
    else:
        st.info("No morning brief found. Run milestone_38.py to generate one.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── News Intelligence ──
    section("NEWS INTELLIGENCE" if not is_beginner
            else "WHAT'S HAPPENING IN THE MARKET?")

    try:
        from src.database import Database as _DB
        _db = _DB()
        recent_news = _db.get_recent_news(limit=30)

        if recent_news:
            scores    = [n.get("impact_score", 0) for n in recent_news]
            avg_score = sum(scores)/len(scores) if scores else 0.0
            pos_count = sum(1 for s in scores if s > 0.2)
            neg_count = sum(1 for s in scores if s < -0.2)
            mood      = ("RISK-ON"  if avg_score > 0.1
                         else "RISK-OFF" if avg_score < -0.1
                         else "NEUTRAL")
            mood_color= ("#00d4aa" if mood == "RISK-ON"
                         else "#ff4d6a" if mood == "RISK-OFF"
                         else "#ffd166")

            st.markdown(f"""
            <div style='background:#0f2040; border:1px solid #1a3357;
                        border-left:3px solid {mood_color}; border-radius:6px;
                        padding:14px 18px; margin-bottom:16px;'>
                <span style='font-family:JetBrains Mono,monospace;
                             font-size:1rem; font-weight:700;
                             color:{mood_color};'>{mood}</span>
                <span style='font-family:JetBrains Mono,monospace;
                             font-size:0.72rem; color:#7b9bc0;
                             margin-left:12px;'>
                    Score: {avg_score:+.3f} ·
                    {pos_count} positive ·
                    {neg_count} negative ·
                    {len(recent_news)} articles
                </span>
                {f"<div style='font-family:DM Sans,sans-serif; font-size:0.83rem; color:#7b9bc0; margin-top:8px; line-height:1.5;'>{'More positive stories than negative — generally a good sign for markets.' if mood == 'RISK-ON' else 'More negative stories than positive — investors may be cautious.' if mood == 'RISK-OFF' else 'About equal positive and negative news — markets are uncertain.'}</div>" if is_beginner else ""}
            </div>""", unsafe_allow_html=True)

            # Ticker sentiment pills
            watch_tickers = ["AAPL","MSFT","NVDA","TSLA","SPY","PSEI.PS"]
            cols = st.columns(len(watch_tickers))
            for col, t in zip(cols, watch_tickers):
                t_news = _db.get_recent_news(ticker=t, limit=30)
                if t_news:
                    t_sc  = [n.get("impact_score",0) for n in t_news]
                    t_avg = sum(t_sc)/len(t_sc)
                    t_col = ("#00d4aa" if t_avg > 0.1
                             else "#ff4d6a" if t_avg < -0.1
                             else "#7b9bc0")
                    col.markdown(f"""
                    <div style='text-align:center; padding:10px 6px;
                                background:#0f2040; border:1px solid #1a3357;
                                border-radius:6px;'>
                        <div style='font-family:JetBrains Mono; font-size:0.6rem;
                                    color:#7b9bc0;'>{t}</div>
                        <div style='font-family:JetBrains Mono; font-size:0.88rem;
                                    font-weight:700; color:{t_col};'>
                            {t_avg:+.2f}</div>
                        <div style='font-family:JetBrains Mono; font-size:0.58rem;
                                    color:#1a3357;'>{len(t_news)} art.</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    col.markdown(f"""
                    <div style='text-align:center; padding:10px 6px;
                                background:#0f2040; border:1px solid #1a3357;
                                border-radius:6px;'>
                        <div style='font-family:JetBrains Mono; font-size:0.6rem;
                                    color:#7b9bc0;'>{t}</div>
                        <div style='font-family:JetBrains Mono; font-size:0.88rem;
                                    color:#1a3357;'>─</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:16px;'></div>",
                        unsafe_allow_html=True)
            section("RECENT HEADLINES")

            for news in recent_news[:15]:
                impact = news.get("market_impact", "neutral")
                score  = news.get("impact_score", 0)
                color  = ("#00d4aa" if impact == "positive"
                          else "#ff4d6a" if impact == "negative"
                          else "#7b9bc0")
                icon   = ("▲" if impact == "positive"
                          else "▼" if impact == "negative" else "─")
                source = news.get("source", "")[:14]
                try:
                    tickers_raw  = news.get("tickers_mentioned", "[]")
                    tickers_list = (json.loads(tickers_raw)
                                    if isinstance(tickers_raw, str)
                                    else tickers_raw)
                    tickers_str  = ", ".join(tickers_list[:2]) if tickers_list else ""
                except Exception:
                    tickers_str = ""
                headline = news.get("headline", "")[:75]

                st.markdown(f"""
                <div style='display:flex; align-items:center; gap:10px;
                            padding:8px 12px; background:#0f2040;
                            border:1px solid #1a3357;
                            border-left:2px solid {color};
                            border-radius:4px; margin-bottom:4px;'>
                    <span style='color:{color}; font-size:0.85rem;
                                 min-width:10px;'>{icon}</span>
                    <span style='font-family:DM Sans,sans-serif;
                                 font-size:0.82rem; color:#e8f0fe;
                                 flex:1;'>{headline}</span>
                    <span style='font-family:JetBrains Mono; font-size:0.6rem;
                                 color:#1a3357; white-space:nowrap;'>
                        {tickers_str}</span>
                    <span style='font-family:JetBrains Mono; font-size:0.6rem;
                                 color:#1a3357; white-space:nowrap;'>
                        {source}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No news data. Run milestone_37.py to populate.")

        _db.close()
    except Exception as e:
        st.error(f"Could not load news: {e}")
        st.info("Run milestone_38.py first.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Philippine Market ──
    section("PHILIPPINE MARKET" if not is_beginner
            else "PHILIPPINE STOCKS & MARKET")

    try:
        from src.philippine_market import (
            load_ph_universe, analyze_ph_ticker, PH_MACRO
        )
        _ph_tickers = ["PSEI.PS","PHI","BDOUY","BPHLY","JBFCF","AYAAF"]
        _today      = datetime.now().strftime("%Y-%m-%d")
        _ph_data    = load_ph_universe(_ph_tickers, "2023-01-01", _today)
        _ph_analyses= [analyze_ph_ticker(t, d)
                       for t, d in _ph_data.items()
                       if analyze_ph_ticker(t, d)]

        if _ph_analyses:
            # Macro row
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("BSP Rate",    f"{PH_MACRO['bsp_rate']}%",
                      delta="Key interest rate")
            m2.metric("Inflation",   f"{PH_MACRO['inflation']}%",
                      delta="Consumer price index")
            m3.metric("GDP Growth",  f"{PH_MACRO['gdp_growth']}%")
            m4.metric("USD/PHP",     f"₱{PH_MACRO['usd_php']:.2f}")

            if is_beginner:
                real_rate = PH_MACRO['bsp_rate'] - PH_MACRO['inflation']
                st.markdown(f"""
                <div style='background:#0f2040; border:1px solid #1a3357;
                            border-left:3px solid #00d4aa; border-radius:6px;
                            padding:14px 18px; margin:12px 0;
                            font-family:DM Sans,sans-serif; font-size:0.85rem;
                            color:#7b9bc0; line-height:1.6;'>
                    <strong style='color:#e8f0fe;'>What this means for you:</strong>
                    The BSP (Bangko Sentral ng Pilipinas) rate is {PH_MACRO['bsp_rate']}%
                    while inflation is {PH_MACRO['inflation']}%, giving a real interest
                    rate of {real_rate:.1f}%. This means your money in savings earns more
                    than inflation loses — which is generally good. GDP growth of
                    {PH_MACRO['gdp_growth']}% means the economy is still expanding.
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:14px;'></div>",
                        unsafe_allow_html=True)

            # Philippine ticker cards
            ph_cols = st.columns(3)
            for idx, a in enumerate(sorted(_ph_analyses,
                                           key=lambda x: x["ticker"])):
                col   = ph_cols[idx % 3]
                sig   = a["signal"]
                clr   = ("#00d4aa" if sig == "BUY"
                         else "#ff4d6a" if sig == "AVOID"
                         else "#7b9bc0")
                sym   = a["symbol"]
                label = plain_signal(sig) if is_beginner else sig
                col.markdown(f"""
                <div style='background:#0f2040; border:1px solid #1a3357;
                            border-top:2px solid {clr}; border-radius:6px;
                            padding:14px; margin-bottom:10px;'>
                    <div style='font-family:JetBrains Mono; font-size:0.62rem;
                                color:#7b9bc0; letter-spacing:0.1em;'>
                        {a['ticker']}</div>
                    <div style='font-family:Space Grotesk; font-size:0.78rem;
                                color:#e8f0fe; font-weight:600; margin:4px 0;'>
                        {a['name'][:24]}</div>
                    <div style='font-family:JetBrains Mono; font-size:1.05rem;
                                font-weight:700; color:{clr};'>
                        {sym}{a['price']:.2f}</div>
                    <div style='font-family:JetBrains Mono; font-size:0.7rem;
                                color:#7b9bc0; margin-top:6px;'>
                        20d: {a['ret_20d']:+.2f}% · RSI {a['rsi']:.1f}</div>
                    <div style='font-family:JetBrains Mono; font-size:0.62rem;
                                font-weight:700; color:{clr}; margin-top:6px;
                                letter-spacing:0.08em;'>
                        {label}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Philippine market data unavailable. "
                    "Run milestone_36.py first.")
    except Exception as e:
        st.error(f"Philippine market error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Experiment Registry ──
    section("EXPERIMENT REGISTRY")
    try:
        from src.experiment_tracker import ExperimentTracker as _ET
        _tracker = _ET()
        if _tracker.registry:
            _df = _tracker.get_all()
            if not _df.empty:
                cols_to_show = [c for c in ["ID","Date","Ticker","Strategy"]
                                if c in _df.columns]
                st.dataframe(_df[cols_to_show].head(10),
                             use_container_width=True, hide_index=True)
                st.markdown(f"""
                <div style='font-family:JetBrains Mono; font-size:0.65rem;
                            color:#1a3357; margin-top:6px;'>
                    {len(_tracker.registry)} total experiments ·
                    Registry: experiments/registry.json
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No experiments logged yet.")
    except Exception as e:
        st.error(f"Registry error: {e}")


# ======================================================================
# TAB 5: COMPANY INTELLIGENCE PAGE
# ======================================================================
with tab5:

    # ── Company search ──
    section("COMPANY INTELLIGENCE")

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        company_ticker = st.text_input(
            "Enter ticker symbol",
            value="AAPL",
            placeholder="e.g. AAPL, NVDA, PHI, BDO...",
            label_visibility="collapsed"
        ).upper()
    with col_btn:
        run_company = st.button("▶  ANALYZE", key="company_btn")

    if is_beginner:
        st.markdown("""
        <div style='font-family:DM Sans,sans-serif; font-size:0.83rem;
                    color:#7b9bc0; margin-bottom:16px; line-height:1.5;'>
            Type any stock ticker above and press Analyze.
            We'll look up what the company does, how healthy its finances are,
            and whether our system thinks it's worth watching right now.
        </div>""", unsafe_allow_html=True)

    if run_company or "company_report" not in st.session_state:
        if run_company or st.session_state.get(
                "company_ticker_last") != company_ticker:
            with st.spinner(f"Generating intelligence report "
                           f"for {company_ticker}..."):
                try:
                    from src.company_intelligence import (
                        generate_company_report
                    )
                    report = generate_company_report(
                        company_ticker, is_beginner)
                    st.session_state["company_report"] = report
                    st.session_state["company_ticker_last"] = \
                        company_ticker
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")
                    report = None
        else:
            report = st.session_state.get("company_report")
    else:
        report = st.session_state.get("company_report")

    if not report:
        st.info("Enter a ticker symbol and press Analyze "
                "to generate a company intelligence report.")
    else:
        f = report.get("fundamentals") or {}
        t = report.get("technical")    or {}
        bull = report.get("bull_case", [])
        bear = report.get("bear_case", [])

        # ── CONVICTION BANNER — the signature element ──
        sig   = t.get("signal", "HOLD")
        rsi   = t.get("rsi", 50)
        reg   = t.get("regime", "UNKNOWN")
        price = t.get("price", 0)
        rat   = f.get("analyst_rating", "N/A")

        if sig == "BUY" and "BUY" in rat:
            conviction       = "STRONG OPPORTUNITY"
            conv_color       = "#00d4aa"
            conv_sub         = ("Technical signals and analyst consensus "
                               "both point positive.")
        elif sig == "BUY":
            conviction       = "WORTH WATCHING"
            conv_color       = "#00d4aa"
            conv_sub         = ("Price signals are constructive. "
                               "Mixed analyst view.")
        elif sig == "AVOID" and "SELL" in rat:
            conviction       = "HIGH RISK — AVOID"
            conv_color       = "#ff4d6a"
            conv_sub         = ("Both technical signals and analysts "
                               "are cautious right now.")
        elif sig == "AVOID":
            conviction       = "CAUTION — RISKY ENVIRONMENT"
            conv_color       = "#ff4d6a"
            conv_sub         = (f"Currently in {reg} regime. "
                               "Wait for better conditions.")
        else:
            conviction       = "NEUTRAL — MONITOR"
            conv_color       = "#ffd166"
            conv_sub         = ("No strong signal in either direction. "
                               "Watch for a clearer setup.")

        beginner_verdict = {
            "STRONG OPPORTUNITY": "This looks like a potentially good time to research buying.",
            "WORTH WATCHING":     "Worth keeping on your watchlist.",
            "HIGH RISK — AVOID":  "Our system says to stay away for now.",
            "CAUTION — RISKY ENVIRONMENT": "The market for this stock is turbulent — be careful.",
            "NEUTRAL — MONITOR":  "No strong signal either way. Keep watching.",
        }.get(conviction, "")

        st.markdown(f"""
        <div style='background:#0f2040;
                    border:1px solid #1a3357;
                    border-left:4px solid {conv_color};
                    border-radius:8px; padding:22px 24px;
                    margin-bottom:20px;'>
            <div style='font-family:JetBrains Mono,monospace;
                        font-size:0.6rem; color:#7b9bc0;
                        letter-spacing:0.2em; margin-bottom:8px;'>
                AURELINE LABS CONVICTION
            </div>
            <div style='font-family:Space Grotesk,sans-serif;
                        font-size:1.6rem; font-weight:700;
                        color:{conv_color}; letter-spacing:-0.5px;
                        line-height:1;'>
                {conviction}
            </div>
            <div style='font-family:DM Sans,sans-serif;
                        font-size:0.85rem; color:#7b9bc0;
                        margin-top:8px;'>
                {beginner_verdict if is_beginner else conv_sub}
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Company header ──
        name   = f.get("name", company_ticker)
        sector = f.get("sector", "")
        indust = f.get("industry", "")
        mktcap = f.get("market_cap", "N/A")
        country= f.get("country", "")

        st.markdown(f"""
        <div style='margin-bottom:20px;'>
            <div style='font-family:Space Grotesk,sans-serif;
                        font-size:1.4rem; font-weight:700;
                        color:#e8f0fe; letter-spacing:-0.5px;'>
                {name}
                <span style='font-family:JetBrains Mono,monospace;
                             font-size:0.75rem; color:#00d4aa;
                             margin-left:10px; font-weight:400;'>
                    {company_ticker}
                </span>
            </div>
            <div style='font-family:JetBrains Mono,monospace;
                        font-size:0.68rem; color:#7b9bc0;
                        margin-top:4px; letter-spacing:0.05em;'>
                {sector}
                {'  ·  ' + indust if indust else ''}
                {'  ·  ' + country if country else ''}
                {'  ·  Market Cap: ' + mktcap if mktcap != 'N/A' else ''}
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Description ──
        desc = f.get("description", "")
        if desc:
            section("BUSINESS OVERVIEW")
            if is_beginner:
                # Show first 300 chars for beginners
                st.markdown(f"""
                <div style='font-family:DM Sans,sans-serif;
                            font-size:0.88rem; color:#7b9bc0;
                            line-height:1.7; background:#0f2040;
                            border:1px solid #1a3357; border-radius:6px;
                            padding:16px 18px;'>
                    {desc[:350]}{'...' if len(desc) > 350 else ''}
                </div>""", unsafe_allow_html=True)
            else:
                with st.expander("Read full description", expanded=True):
                    st.markdown(f"""
                    <div style='font-family:DM Sans,sans-serif;
                                font-size:0.85rem; color:#7b9bc0;
                                line-height:1.7;'>
                        {desc[:800]}{'...' if len(desc) > 800 else ''}
                    </div>""", unsafe_allow_html=True)

        # ── Two-column layout: Fundamentals + Technicals ──
        st.markdown("<div style='margin-top:20px;'></div>",
                    unsafe_allow_html=True)
        fcol, tcol = st.columns(2)

        with fcol:
            section("FUNDAMENTAL METRICS")

            def metric_row(label, value, note="",
                           beginner_label=None):
                display_label = (beginner_label
                                 if is_beginner and beginner_label
                                 else label)
                color = "#e8f0fe" if value != "N/A" else "#1a3357"
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between;
                            align-items:center; padding:7px 0;
                            border-bottom:1px solid #0d1b35;'>
                    <span style='font-family:JetBrains Mono,monospace;
                                 font-size:0.68rem; color:#7b9bc0;'>
                        {display_label}</span>
                    <span style='font-family:JetBrains Mono,monospace;
                                 font-size:0.78rem; font-weight:600;
                                 color:{color};'>{value}</span>
                </div>""", unsafe_allow_html=True)

            metric_row("P/E Ratio",
                       str(round(float(f.get("pe_ratio",0)),1))
                       if f.get("pe_ratio") not in ["N/A",None] else "N/A",
                       beginner_label="Price vs Earnings (P/E)")
            metric_row("Forward P/E",
                       str(round(float(f.get("forward_pe",0)),1))
                       if f.get("forward_pe") not in ["N/A",None] else "N/A",
                       beginner_label="Expected P/E (next year)")
            metric_row("Revenue",
                       f.get("revenue","N/A"),
                       beginner_label="Annual Revenue")
            metric_row("Revenue Growth",
                       f.get("revenue_growth","N/A"),
                       beginner_label="How fast revenue is growing")
            metric_row("Profit Margin",
                       f.get("profit_margin","N/A"),
                       beginner_label="Profit kept per ₱100 of sales")
            metric_row("Gross Margin",
                       f.get("gross_margin","N/A"),
                       beginner_label="Gross profit margin")
            metric_row("ROE",
                       f.get("roe","N/A"),
                       beginner_label="Return on shareholders' money")
            metric_row("Debt/Equity",
                       str(f.get("debt_to_equity","N/A")),
                       beginner_label="Debt vs company value")
            metric_row("Free Cash Flow",
                       f.get("free_cashflow","N/A"),
                       beginner_label="Cash generated after expenses")
            metric_row("Dividend Yield",
                       f.get("dividend_yield","N/A"),
                       beginner_label="Annual dividend payment")

        with tcol:
            section("TECHNICAL SNAPSHOT")

            def tech_row(label, value, color="#e8f0fe",
                         beginner_label=None):
                display_label = (beginner_label
                                 if is_beginner and beginner_label
                                 else label)
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between;
                            align-items:center; padding:7px 0;
                            border-bottom:1px solid #0d1b35;'>
                    <span style='font-family:JetBrains Mono,monospace;
                                 font-size:0.68rem; color:#7b9bc0;'>
                        {display_label}</span>
                    <span style='font-family:JetBrains Mono,monospace;
                                 font-size:0.78rem; font-weight:600;
                                 color:{color};'>{value}</span>
                </div>""", unsafe_allow_html=True)

            price_color = "#00d4aa"
            ret20_color = ("#00d4aa" if t.get("ret_20d",0) > 0
                           else "#ff4d6a")
            rsi_color   = ("#ffd166" if rsi > 65 or rsi < 35
                           else "#e8f0fe")
            reg_color   = ("#00d4aa" if reg == "BULL_TRENDING"
                           else "#ff4d6a"
                           if reg in ["BEAR_TRENDING","HIGH_VOLATILITY"]
                           else "#ffd166")
            sig_color   = ("#00d4aa" if sig == "BUY"
                           else "#ff4d6a" if sig == "AVOID"
                           else "#ffd166")

            tech_row("Price",
                     f"${t.get('price',0):.2f}",
                     price_color)
            tech_row("1-Day Return",
                     f"{t.get('ret_1d',0):+.2f}%",
                     "#00d4aa" if t.get("ret_1d",0) > 0 else "#ff4d6a")
            tech_row("20-Day Return",
                     f"{t.get('ret_20d',0):+.2f}%",
                     ret20_color,
                     beginner_label="Price change (last month)")
            tech_row("RSI (14)",
                     f"{rsi:.1f}",
                     rsi_color,
                     beginner_label="Momentum indicator (0-100)")
            tech_row("Volatility (20d)",
                     f"{t.get('vol_20d',0):.1f}%",
                     beginner_label="How much price swings")
            tech_row("Market Regime",
                     reg, reg_color,
                     beginner_label="Current market environment")
            tech_row("52w High",
                     f"${t.get('high_52w',0):.2f}")
            tech_row("52w Low",
                     f"${t.get('low_52w',0):.2f}")
            tech_row("From 52w High",
                     f"{t.get('pct_from_high',0):+.1f}%",
                     "#ff4d6a" if t.get("pct_from_high",0) < -20
                     else "#e8f0fe",
                     beginner_label="Distance from yearly peak")
            tech_row("3m ATM Call",
                     f"${t.get('call_3m','N/A')}",
                     beginner_label="3-month call option price")
            tech_row("Signal",
                     sig, sig_color,
                     beginner_label="Our system's recommendation")

            if is_beginner and rsi < 35:
                st.markdown(f"""
                <div style='background:#0f2040; border:1px solid #1a3357;
                            border-left:2px solid #ffd166; border-radius:4px;
                            padding:10px 12px; margin-top:10px;
                            font-family:DM Sans,sans-serif; font-size:0.78rem;
                            color:#7b9bc0;'>
                    💡 RSI below 35 means the stock may have been
                    falling too fast and could bounce back — but
                    this isn't guaranteed.
                </div>""", unsafe_allow_html=True)
            elif is_beginner and rsi > 70:
                st.markdown(f"""
                <div style='background:#0f2040; border:1px solid #1a3357;
                            border-left:2px solid #ffd166; border-radius:4px;
                            padding:10px 12px; margin-top:10px;
                            font-family:DM Sans,sans-serif; font-size:0.78rem;
                            color:#7b9bc0;'>
                    💡 RSI above 70 means the stock may have risen
                    too quickly and a pullback is possible.
                </div>""", unsafe_allow_html=True)

        # ── Analyst View ──
        st.markdown("<div style='margin-top:20px;'></div>",
                    unsafe_allow_html=True)
        section("ANALYST CONSENSUS")

        rat_color = ("#00d4aa"
                     if any(x in rat for x in ["BUY","OUTPERFORM"])
                     else "#ff4d6a"
                     if any(x in rat for x in ["SELL","UNDERPERFORM"])
                     else "#ffd166")

        tgt_mean = f.get("target_mean", "N/A")
        tgt_high = f.get("target_high", "N/A")
        tgt_low  = f.get("target_low",  "N/A")

        upside_str = ""
        if tgt_mean not in ["N/A", None] and price > 0:
            try:
                upside = (float(tgt_mean)/price - 1) * 100
                upside_str = (f"+{upside:.1f}% upside"
                              if upside > 0
                              else f"{upside:.1f}% downside")
            except Exception:
                pass

        a1, a2, a3, a4 = st.columns(4)
        a1.markdown(f"""
        <div style='background:#0f2040; border:1px solid #1a3357;
                    border-top:2px solid {rat_color}; border-radius:6px;
                    padding:14px; text-align:center;'>
            <div style='font-family:JetBrains Mono; font-size:0.6rem;
                        color:#7b9bc0;'>CONSENSUS</div>
            <div style='font-family:Space Grotesk; font-size:1rem;
                        font-weight:700; color:{rat_color};
                        margin-top:6px;'>{rat}</div>
        </div>""", unsafe_allow_html=True)
        a2.markdown(f"""
        <div style='background:#0f2040; border:1px solid #1a3357;
                    border-top:2px solid #1a6eff; border-radius:6px;
                    padding:14px; text-align:center;'>
            <div style='font-family:JetBrains Mono; font-size:0.6rem;
                        color:#7b9bc0;'>PRICE TARGET</div>
            <div style='font-family:Space Grotesk; font-size:1rem;
                        font-weight:700; color:#1a6eff; margin-top:6px;'>
                ${tgt_mean if tgt_mean != 'N/A' else '—'}
            </div>
        </div>""", unsafe_allow_html=True)
        a3.markdown(f"""
        <div style='background:#0f2040; border:1px solid #1a3357;
                    border-top:2px solid #00d4aa; border-radius:6px;
                    padding:14px; text-align:center;'>
            <div style='font-family:JetBrains Mono; font-size:0.6rem;
                        color:#7b9bc0;'>UPSIDE / DOWNSIDE</div>
            <div style='font-family:Space Grotesk; font-size:1rem;
                        font-weight:700;
                        color:{"#00d4aa" if "upside" in upside_str else "#ff4d6a"};
                        margin-top:6px;'>
                {upside_str or "—"}
            </div>
        </div>""", unsafe_allow_html=True)
        a4.markdown(f"""
        <div style='background:#0f2040; border:1px solid #1a3357;
                    border-top:2px solid #7b9bc0; border-radius:6px;
                    padding:14px; text-align:center;'>
            <div style='font-family:JetBrains Mono; font-size:0.6rem;
                        color:#7b9bc0;'>ANALYSTS COVERING</div>
            <div style='font-family:Space Grotesk; font-size:1rem;
                        font-weight:700; color:#7b9bc0; margin-top:6px;'>
                {f.get("num_analysts", "—")}
            </div>
        </div>""", unsafe_allow_html=True)

        if is_beginner and tgt_mean not in ["N/A", None]:
            st.markdown(f"""
            <div style='font-family:DM Sans,sans-serif; font-size:0.82rem;
                        color:#7b9bc0; background:#0f2040;
                        border:1px solid #1a3357; border-radius:6px;
                        padding:12px 16px; margin-top:10px; line-height:1.6;'>
                💡 <strong style='color:#e8f0fe;'>What this means:</strong>
                {f.get('num_analysts', 'Several')} professional analysts
                who research this company for a living are saying
                <strong style='color:{rat_color};'>{rat}</strong>,
                with an average price target of
                <strong style='color:#1a6eff;'>
                    ${tgt_mean}
                </strong>
                (range ${tgt_low}–${tgt_high}).
                {f"This implies {upside_str} from the current price." if upside_str else ""}
            </div>""", unsafe_allow_html=True)

        # ── Bull / Bear Cases ──
        st.markdown("<div style='margin-top:20px;'></div>",
                    unsafe_allow_html=True)
        section("BULL vs BEAR ANALYSIS" if not is_beginner
                else "WHY IT COULD GO UP OR DOWN")

        if is_beginner:
            st.markdown("""
            <div style='font-family:DM Sans,sans-serif;
                        font-size:0.82rem; color:#7b9bc0;
                        margin-bottom:12px;'>
                Here are the main reasons the stock might
                go up (bull case) or down (bear case).
                These are based on real financial data,
                not opinions.
            </div>""", unsafe_allow_html=True)

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            st.markdown("""
            <div style='font-family:JetBrains Mono,monospace;
                        font-size:0.65rem; color:#00d4aa;
                        letter-spacing:0.15em; margin-bottom:10px;
                        padding-bottom:4px;
                        border-bottom:1px solid #00d4aa;'>
                ▲ BULL CASE
            </div>""", unsafe_allow_html=True)
            for b in bull:
                st.markdown(f"""
                <div style='display:flex; gap:8px; padding:8px 0;
                            border-bottom:1px solid #0d1b35;'>
                    <span style='color:#00d4aa; font-size:0.8rem;
                                 min-width:12px;'>▲</span>
                    <span style='font-family:DM Sans,sans-serif;
                                 font-size:0.82rem; color:#e8f0fe;
                                 line-height:1.5;'>{b}</span>
                </div>""", unsafe_allow_html=True)

        with bcol2:
            st.markdown("""
            <div style='font-family:JetBrains Mono,monospace;
                        font-size:0.65rem; color:#ff4d6a;
                        letter-spacing:0.15em; margin-bottom:10px;
                        padding-bottom:4px;
                        border-bottom:1px solid #ff4d6a;'>
                ▼ BEAR CASE
            </div>""", unsafe_allow_html=True)
            for b in bear:
                st.markdown(f"""
                <div style='display:flex; gap:8px; padding:8px 0;
                            border-bottom:1px solid #0d1b35;'>
                    <span style='color:#ff4d6a; font-size:0.8rem;
                                 min-width:12px;'>▼</span>
                    <span style='font-family:DM Sans,sans-serif;
                                 font-size:0.82rem; color:#e8f0fe;
                                 line-height:1.5;'>{b}</span>
                </div>""", unsafe_allow_html=True)

        # ── Disclaimer ──
        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace;
                    font-size:0.6rem; color:#1a3357; margin-top:24px;
                    padding-top:12px; border-top:1px solid #0d1b35;
                    line-height:1.8;'>
            Report generated: {report.get('generated_at','')[:19]}
            &nbsp;·&nbsp; Data: Yahoo Finance
            &nbsp;·&nbsp; FOR RESEARCH PURPOSES ONLY
            &nbsp;·&nbsp; NOT FINANCIAL ADVICE
        </div>""", unsafe_allow_html=True)

# ======================================================================
# TAB 6: FINANCIAL COPILOT
# ======================================================================
with tab6:
    section("FINANCIAL COPILOT")

    if is_beginner:
        st.markdown("""
        <div style='background:#0f2040; border:1px solid #1a3357;
                    border-left:3px solid #00d4aa; border-radius:6px;
                    padding:14px 18px; margin-bottom:16px;
                    font-family:DM Sans,sans-serif; font-size:0.85rem;
                    color:#7b9bc0; line-height:1.6;'>
            👋 Ask me anything about investing, stocks, or finance.
            I'll explain it in plain language. You can also ask
            about specific stocks like "What is AAPL doing?"
            or "Is NVDA a good buy right now?"
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#0f2040; border:1px solid #1a3357;
                    border-left:3px solid #00d4aa; border-radius:6px;
                    padding:12px 18px; margin-bottom:14px;
                    font-family:JetBrains Mono,monospace;
                    font-size:0.75rem; color:#7b9bc0; line-height:1.7;'>
            Query topics: inflation · interest_rates · pe_ratio ·
            sharpe_ratio · rsi · diversification · market_regime ·
            options · aureline_performance · any ticker (AAPL, NVDA, PHI...)
        </div>""", unsafe_allow_html=True)

    # ── Chat history ──
    if "copilot_history" not in st.session_state:
        st.session_state["copilot_history"] = []

    # ── Chat input ──
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What is the P/E ratio? / What is AAPL doing?",
        label_visibility="collapsed",
        key="copilot_input"
    )
    col_ask, col_clear = st.columns([5, 1])
    with col_ask:
        ask_btn = st.button("▶  ASK", key="copilot_ask")
    with col_clear:
        if st.button("Clear", key="copilot_clear"):
            st.session_state["copilot_history"] = []
            st.rerun()

    if ask_btn and question.strip():
        try:
            from src.financial_copilot import FinancialCopilot
            from src.database import Database as _DB2
            _db2    = _DB2()
            copilot = FinancialCopilot(db=_db2,
                                        is_beginner=is_beginner)
            response, followups, sources = copilot.answer(question)
            _db2.close()

            st.session_state["copilot_history"].append({
                "question":  question,
                "response":  response,
                "followups": followups,
                "sources":   sources,
                "time":      datetime.now().strftime("%H:%M"),
            })
        except Exception as e:
            st.error(f"Copilot error: {e}")

    # ── Display history ──
    history = st.session_state.get("copilot_history", [])

    if not history:
        # Suggested questions
        section("TRY ASKING")
        suggestions = [
            "What is inflation and how does it affect stocks?",
            "What is the current signal for AAPL?",
            "Explain the Sharpe ratio",
            "What is market regime detection?",
            "How has Aureline Labs performed?",
            "What is a P/E ratio?",
        ] if is_beginner else [
            "Explain the RSI indicator",
            "What is the Aureline Labs regime detector?",
            "Black-Scholes options pricing",
            "Curse of dimensionality in ML",
            "What is risk parity?",
            "NVDA fundamentals",
        ]

        scols = st.columns(2)
        for i, sug in enumerate(suggestions):
            col = scols[i % 2]
            col.markdown(f"""
            <div style='background:#0f2040;
                        border:1px solid #1a3357;
                        border-radius:4px; padding:10px 14px;
                        margin-bottom:6px; cursor:pointer;
                        font-family:DM Sans,sans-serif;
                        font-size:0.82rem; color:#7b9bc0;'>
                💬 {sug}
            </div>""", unsafe_allow_html=True)
    else:
        # Show conversation newest first
        for item in reversed(history):
            # Question bubble
            st.markdown(f"""
            <div style='background:#1a3357; border-radius:8px;
                        padding:12px 16px; margin:8px 0 4px 40px;
                        font-family:DM Sans,sans-serif;
                        font-size:0.85rem; color:#e8f0fe;'>
                <span style='font-family:JetBrains Mono;
                             font-size:0.6rem; color:#7b9bc0;
                             display:block; margin-bottom:4px;'>
                    YOU · {item['time']}</span>
                {item['question']}
            </div>""", unsafe_allow_html=True)

            # Response bubble
            st.markdown(f"""
            <div style='background:#0f2040;
                        border:1px solid #1a3357;
                        border-left:3px solid #00d4aa;
                        border-radius:8px; padding:14px 18px;
                        margin:4px 40px 4px 0;'>
                <span style='font-family:JetBrains Mono;
                             font-size:0.6rem; color:#00d4aa;
                             display:block; margin-bottom:8px;'>
                    ⬡ AURELINE COPILOT</span>
                <div style='font-family:DM Sans,sans-serif;
                            font-size:0.85rem; color:#e8f0fe;
                            line-height:1.7;'>""",
                unsafe_allow_html=True)
            st.markdown(item["response"])
            st.markdown("</div></div>", unsafe_allow_html=True)

            # Follow-ups
            if item.get("followups"):
                fu_cols = st.columns(len(item["followups"][:3]))
                for fi, fu in enumerate(item["followups"][:3]):
                    fu_cols[fi].markdown(f"""
                    <div style='background:#060d1f;
                                border:1px solid #1a3357;
                                border-radius:4px; padding:8px 10px;
                                font-family:JetBrains Mono,monospace;
                                font-size:0.65rem; color:#7b9bc0;
                                cursor:pointer;'>
                        ↪ {fu}
                    </div>""", unsafe_allow_html=True)

# ======================================================================
# TAB 7: PORTFOLIO INTELLIGENCE
# ======================================================================
with tab7:
    section("PORTFOLIO INTELLIGENCE")

    if is_beginner:
        st.markdown("""
        <div style='background:#0f2040; border:1px solid #1a3357;
                    border-left:3px solid #00d4aa; border-radius:6px;
                    padding:14px 18px; margin-bottom:16px;
                    font-family:DM Sans,sans-serif; font-size:0.85rem;
                    color:#7b9bc0; line-height:1.6;'>
            Build your own portfolio here. Select the stocks you want
            to invest in, choose how to split your money, and see what
            would have happened historically — including the best and
            worst case scenarios from 5,000 simulations.
        </div>""", unsafe_allow_html=True)

    # ── Portfolio controls ──
    pcol1, pcol2 = st.columns([3, 2])

    with pcol1:
        st.markdown("""
        <div style='font-family:JetBrains Mono,monospace;
                    font-size:0.6rem; color:#7b9bc0;
                    letter-spacing:0.15em; margin-bottom:6px;'>
            SELECT ASSETS
        </div>""", unsafe_allow_html=True)

        available_assets = ["AAPL", "MSFT", "NVDA", "JPM",
                            "XOM", "JNJ", "TSLA", "SPY"]
        selected_assets  = st.multiselect(
            "Assets",
            options=available_assets,
            default=["AAPL", "MSFT", "NVDA", "JPM",
                     "XOM", "JNJ"],
            label_visibility="collapsed"
        )

    with pcol2:
        st.markdown("""
        <div style='font-family:JetBrains Mono,monospace;
                    font-size:0.6rem; color:#7b9bc0;
                    letter-spacing:0.15em; margin-bottom:6px;'>
            ALLOCATION STRATEGY
        </div>""", unsafe_allow_html=True)
        strategy_choice = st.selectbox(
            "Strategy",
            options=["Equal Weight (1/N)",
                     "Risk Parity",
                     "Momentum Weighted"],
            label_visibility="collapsed"
        )

    port_start = st.date_input(
        "Portfolio Start Date",
        value=pd.Timestamp("2021-01-01"),
        label_visibility="visible"
    ).strftime("%Y-%m-%d")

    run_portfolio = st.button("▶  BUILD PORTFOLIO", key="port_btn")

    if not selected_assets:
        st.info("Select at least 2 assets to build a portfolio.")
    elif run_portfolio or "portfolio_result" in st.session_state:

        if run_portfolio:
            with st.spinner("Loading price data and running "
                            "portfolio analysis..."):
                try:
                    from src.portfolio import Portfolio

                    # Load prices
                    prices = {}
                    for t in selected_assets:
                        data = get_price_data(
                            t, port_start, end_date)
                        if data is not None and len(data) > 60:
                            prices[t] = data["Close"]

                    if len(prices) < 2:
                        st.error("Need at least 2 assets with "
                                 "valid price data.")
                    else:
                        port = Portfolio(prices,
                                         starting_cash=100000)

                        # Select weights
                        if strategy_choice == "Equal Weight (1/N)":
                            weights = port.equal_weight()
                        elif strategy_choice == "Risk Parity":
                            weights = port.risk_parity()
                        else:
                            weights = port.momentum_weighted()

                        # Simulate
                        pv   = port.simulate(
                            weights,
                            rebalance_frequency="monthly")
                        corr = port.correlation_matrix()
                        stats = port.asset_statistics()
                        dr   = port.diversification_ratio(weights)
                        pv_vol = port.portfolio_volatility(weights)

                        # Metrics
                        ret  = (pv[-1]/100000 - 1) * 100
                        ps   = pd.Series(pv)
                        dret = ps.pct_change().dropna()
                        ex   = dret - 0.05/252
                        sh   = round(
                            (ex.mean()/ex.std())*np.sqrt(252), 3
                        ) if ex.std() > 0 else 0.0
                        peak = ps.cummax()
                        mdd_p= round(
                            ((ps-peak)/peak).min()*100, 2)

                        st.session_state["portfolio_result"] = {
                            "pv": pv, "corr": corr,
                            "stats": stats, "weights": weights,
                            "dr": dr, "pv_vol": pv_vol,
                            "ret": ret, "sharpe": sh,
                            "mdd": mdd_p, "prices": prices,
                            "port": port,
                            "strategy": strategy_choice,
                        }
                except Exception as e:
                    st.error(f"Portfolio error: {e}")

        # ── Display results ──
        result = st.session_state.get("portfolio_result")
        if result:
            pv       = result["pv"]
            corr     = result["corr"]
            weights  = result["weights"]
            ret      = result["ret"]
            sh       = result["sharpe"]
            mdd_p    = result["mdd"]
            dr       = result["dr"]
            pv_vol   = result["pv_vol"]

            # ── Top metrics ──
            st.markdown("<div style='margin-top:16px;'></div>",
                        unsafe_allow_html=True)

            if is_beginner:
                m1, m2, m3 = st.columns(3)
                ret_val, ret_desc = plain_return(ret)
                m1.markdown(f"""
                <div class='bcard'>
                    <div class='bcard-label'>Total Return</div>
                    <div class='bcard-value'
                         style='color:{"#00d4aa" if ret > 0 else "#ff4d6a"};'>
                        {ret_val}
                    </div>
                    <div class='bcard-desc'>
                        ₱100,000 grew to
                        ₱{pv[-1]:,.0f}
                    </div>
                </div>""", unsafe_allow_html=True)
                m2.markdown(f"""
                <div class='bcard'>
                    <div class='bcard-label'>Consistency</div>
                    <div class='bcard-value'>{plain_sharpe(sh).split("—")[0].strip()}</div>
                    <div class='bcard-desc'>
                        {plain_sharpe(sh).split("—")[1].strip()
                         if "—" in plain_sharpe(sh) else ""}
                    </div>
                </div>""", unsafe_allow_html=True)
                m3.markdown(f"""
                <div class='bcard'>
                    <div class='bcard-label'>Worst Period</div>
                    <div class='bcard-value'
                         style='color:{"#ffd166" if abs(mdd_p) < 20 else "#ff4d6a"};'>
                        {mdd_p:.1f}%
                    </div>
                    <div class='bcard-desc'>
                        {plain_drawdown(mdd_p)}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric("Total Return",    f"{ret:+.2f}%")
                m2.metric("Sharpe Ratio",    f"{sh:.3f}")
                m3.metric("Max Drawdown",    f"{mdd_p:.2f}%")
                m4.metric("Portfolio Vol",   f"{pv_vol*100:.1f}%")
                m5.metric("Diversif. Ratio", f"{dr:.3f}")

            # ── Allocation weights ──
            st.markdown("<div style='margin-top:20px;'></div>",
                        unsafe_allow_html=True)
            section("ALLOCATION WEIGHTS")

            w_cols = st.columns(len(weights))
            colors_w = ["#00d4aa","#1a6eff","#ffd166",
                        "#ff4d6a","#7b61ff","#06d6a0",
                        "#118ab2","#8ed4b4"]
            for idx, (t, w) in enumerate(
                    sorted(weights.items(),
                           key=lambda x: x[1], reverse=True)):
                col = w_cols[idx % len(w_cols)]
                col.markdown(f"""
                <div style='background:#0f2040;
                            border:1px solid #1a3357;
                            border-top:2px solid
                            {colors_w[idx % len(colors_w)]};
                            border-radius:6px; padding:12px;
                            text-align:center;'>
                    <div style='font-family:JetBrains Mono;
                                font-size:0.6rem; color:#7b9bc0;'>
                        {t}</div>
                    <div style='font-family:Space Grotesk;
                                font-size:1.1rem; font-weight:700;
                                color:{colors_w[idx%len(colors_w)]};
                                margin-top:4px;'>
                        {w:.0%}</div>
                </div>""", unsafe_allow_html=True)

            # ── Portfolio equity curve ──
            st.markdown("<div style='margin-top:20px;'></div>",
                        unsafe_allow_html=True)
            section("PORTFOLIO PERFORMANCE")

            fig_p, ax_p = plt.subplots(figsize=(13, 4))
            aureline_chart_style(ax_p, fig_p)
            ax_p.plot(pv, color="#00d4aa",
                      linewidth=1.8, label=result["strategy"])
            ax_p.axhline(y=100000, color="#1a3357",
                         linestyle="--", linewidth=1)
            ax_p.yaxis.set_major_formatter(
                mticker.StrMethodFormatter("${x:,.0f}"))
            ax_p.set_ylabel("Portfolio Value ($)",
                            fontsize=8, color="#7b9bc0")
            ax_p.legend(fontsize=8, facecolor="#0f2040",
                        edgecolor="#1a3357",
                        labelcolor="#7b9bc0")

            if is_beginner:
                ax_p.set_title(
                    f"₱100,000 invested → ₱{pv[-1]:,.0f}",
                    color="#7b9bc0", fontsize=9,
                    fontfamily="monospace")

            plt.tight_layout()
            st.pyplot(fig_p); plt.close()

            # ── Correlation matrix ──
            st.markdown("<div style='margin-top:20px;'></div>",
                        unsafe_allow_html=True)
            section("CORRELATION MATRIX" if not is_beginner
                    else "HOW MUCH DO THESE STOCKS MOVE TOGETHER?")

            if is_beginner:
                st.markdown("""
                <div style='font-family:DM Sans,sans-serif;
                            font-size:0.82rem; color:#7b9bc0;
                            margin-bottom:10px; line-height:1.5;'>
                    Numbers close to <strong style='color:#00d4aa;'>
                    +1.0</strong> mean the stocks move together
                    (less diversification). Numbers close to
                    <strong style='color:#ff4d6a;'>0.0 or negative
                    </strong> mean they move independently
                    (better diversification).
                </div>""", unsafe_allow_html=True)

            tickers_list = list(corr.columns)
            fig_c, ax_c  = plt.subplots(
                figsize=(max(6, len(tickers_list)*1.2),
                         max(4, len(tickers_list)*0.9)))
            aureline_chart_style(ax_c, fig_c)
            im = ax_c.imshow(corr.values, cmap="RdYlGn",
                              vmin=-1, vmax=1, aspect="auto")
            ax_c.set_xticks(range(len(tickers_list)))
            ax_c.set_yticks(range(len(tickers_list)))
            ax_c.set_xticklabels(tickers_list,
                                  fontsize=8, color="#7b9bc0")
            ax_c.set_yticklabels(tickers_list,
                                  fontsize=8, color="#7b9bc0")
            for i in range(len(tickers_list)):
                for j in range(len(tickers_list)):
                    ax_c.text(j, i,
                              f"{corr.values[i,j]:.2f}",
                              ha="center", va="center",
                              fontsize=7, color="#060d1f",
                              fontweight="bold")
            ax_c.set_title(
                "Correlation Matrix — 1.0 = move together, "
                "0.0 = independent",
                color="#7b9bc0", fontsize=8,
                fontfamily="monospace")
            plt.tight_layout()
            st.pyplot(fig_c); plt.close()

            # ── High / Low correlation pairs ──
            try:
                port_obj = result["port"]
                high_pairs = port_obj.find_high_correlation_pairs(0.6)
                low_pairs  = port_obj.find_low_correlation_pairs(0.3)

                cp1, cp2 = st.columns(2)
                with cp1:
                    if high_pairs:
                        st.markdown(f"""
                        <div style='background:#0f2040;
                                    border:1px solid #1a3357;
                                    border-left:2px solid #ff4d6a;
                                    border-radius:4px; padding:12px 14px;'>
                            <div style='font-family:JetBrains Mono;
                                        font-size:0.6rem; color:#ff4d6a;
                                        margin-bottom:6px;'>
                                HIGH CORRELATION — REDUNDANT RISK</div>
                            {''.join([
                                f"<div style='font-family:JetBrains Mono;"
                                f"font-size:0.72rem; color:#7b9bc0;"
                                f"padding:3px 0;'>{t1} ↔ {t2}: {c:+.3f}</div>"
                                for t1, t2, c in high_pairs[:4]
                            ])}
                        </div>""", unsafe_allow_html=True)
                with cp2:
                    if low_pairs:
                        st.markdown(f"""
                        <div style='background:#0f2040;
                                    border:1px solid #1a3357;
                                    border-left:2px solid #00d4aa;
                                    border-radius:4px; padding:12px 14px;'>
                            <div style='font-family:JetBrains Mono;
                                        font-size:0.6rem; color:#00d4aa;
                                        margin-bottom:6px;'>
                                LOW CORRELATION — GENUINE DIVERSIFICATION</div>
                            {''.join([
                                f"<div style='font-family:JetBrains Mono;"
                                f"font-size:0.72rem; color:#7b9bc0;"
                                f"padding:3px 0;'>{t1} ↔ {t2}: {c:+.3f}</div>"
                                for t1, t2, c in low_pairs[:4]
                            ])}
                        </div>""", unsafe_allow_html=True)
            except Exception:
                pass

            # ── Monte Carlo ──
            st.markdown("<div style='margin-top:20px;'></div>",
                        unsafe_allow_html=True)
            section("MONTE CARLO RISK SIMULATION" if not is_beginner
                    else "WHAT COULD HAPPEN IN THE FUTURE?")

            if is_beginner:
                st.markdown("""
                <div style='font-family:DM Sans,sans-serif;
                            font-size:0.82rem; color:#7b9bc0;
                            margin-bottom:10px; line-height:1.5;'>
                    We ran <strong>1,000 simulations</strong> of what
                    could happen to your portfolio based on historical
                    patterns. The shaded area shows the range of
                    likely outcomes.
                </div>""", unsafe_allow_html=True)

            try:
                from src.monte_carlo import MonteCarloSimulator
                with st.spinner("Running Monte Carlo simulation..."):
                    n_sims = 1000 if is_beginner else 3000
                    mc     = MonteCarloSimulator(
                        pv, n_simulations=n_sims,
                        block_size=20)
                    paths   = mc.generate_paths()
                    mc_res  = mc.analyze(paths)

                # Monte Carlo chart
                fig_mc, ax_mc = plt.subplots(figsize=(13, 4))
                aureline_chart_style(ax_mc, fig_mc)

                # Draw sample paths
                sample_idx = np.random.choice(
                    n_sims, min(150, n_sims), replace=False)
                for idx in sample_idx:
                    ax_mc.plot(paths[idx], color="#00d4aa",
                               alpha=0.02, linewidth=0.5)

                # Percentile bands
                p5  = np.percentile(paths, 5,  axis=0)
                p25 = np.percentile(paths, 25, axis=0)
                p75 = np.percentile(paths, 75, axis=0)
                p95 = np.percentile(paths, 95, axis=0)
                med = np.median(paths, axis=0)

                ax_mc.fill_between(range(len(p5)),
                                    p5, p95, alpha=0.12,
                                    color="#00d4aa")
                ax_mc.fill_between(range(len(p25)),
                                    p25, p75, alpha=0.22,
                                    color="#00d4aa")
                ax_mc.plot(med, color="#00d4aa",
                           linewidth=1.8, label="Median path")
                ax_mc.plot(pv, color="#ffffff",
                           linewidth=1.2, linestyle="--",
                           alpha=0.7, label="Historical")
                ax_mc.axhline(y=100000, color="#1a3357",
                              linestyle="--", linewidth=0.8)
                ax_mc.yaxis.set_major_formatter(
                    mticker.StrMethodFormatter("${x:,.0f}"))
                ax_mc.set_ylabel("Portfolio Value ($)",
                                  fontsize=8, color="#7b9bc0")
                ax_mc.legend(fontsize=8, facecolor="#0f2040",
                             edgecolor="#1a3357",
                             labelcolor="#7b9bc0")
                plt.tight_layout()
                st.pyplot(fig_mc); plt.close()

                # MC metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                p_loss   = mc_res["prob_loss"]
                med_ret  = mc_res["median_return"]
                p5_ret   = mc_res["p5_return"]
                p95_ret  = mc_res["p95_return"]
                worst_dd = mc_res["worst_max_dd"]

                if is_beginner:
                    mc1.markdown(f"""
                    <div class='bcard'>
                        <div class='bcard-label'>Chance of Loss</div>
                        <div class='bcard-value'
                             style='color:{"#00d4aa" if p_loss < 5 else "#ffd166"};'>
                            {p_loss:.1f}%</div>
                        <div class='bcard-desc'>
                            Probability of losing money
                            over the full period</div>
                    </div>""", unsafe_allow_html=True)
                    mc2.markdown(f"""
                    <div class='bcard'>
                        <div class='bcard-label'>Typical Outcome</div>
                        <div class='bcard-value'
                             style='color:#00d4aa;'>
                            +{med_ret:.0f}%</div>
                        <div class='bcard-desc'>
                            Median return across
                            all simulations</div>
                    </div>""", unsafe_allow_html=True)
                    mc3.markdown(f"""
                    <div class='bcard'>
                        <div class='bcard-label'>Bad Case (5%)</div>
                        <div class='bcard-value'
                             style='color:#ffd166;'>
                            {p5_ret:+.0f}%</div>
                        <div class='bcard-desc'>
                            The worst 5% of scenarios
                            still produced this return</div>
                    </div>""", unsafe_allow_html=True)
                    mc4.markdown(f"""
                    <div class='bcard'>
                        <div class='bcard-label'>Worst Drawdown</div>
                        <div class='bcard-value'
                             style='color:#ff4d6a;'>
                            {worst_dd:.1f}%</div>
                        <div class='bcard-desc'>
                            Biggest simulated loss
                            from peak to trough</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    mc1.metric("P(Loss)",
                               f"{p_loss:.1f}%")
                    mc2.metric("Median Return",
                               f"{med_ret:+.1f}%")
                    mc3.metric("5th Pct Return",
                               f"{p5_ret:+.1f}%")
                    mc4.metric("Worst Simulated DD",
                               f"{worst_dd:.1f}%")

                # Historical vs percentile context
                hist_ret = ret
                pct_rank = float(
                    np.mean(
                        mc_res["_final_returns"] < hist_ret/100
                    ) * 100
                )
                st.markdown(f"""
                <div style='background:#0f2040;
                            border:1px solid #1a3357;
                            border-left:3px solid #ffd166;
                            border-radius:6px; padding:12px 18px;
                            margin-top:12px;
                            font-family:{'DM Sans' if is_beginner
                                         else 'JetBrains Mono'},
                                         {'sans-serif' if is_beginner
                                          else 'monospace'};
                            font-size:0.82rem; color:#7b9bc0;
                            line-height:1.6;'>
                    {'💡' if is_beginner else ''}
                    The historical return of
                    <strong style='color:#e8f0fe;'>
                        {hist_ret:+.1f}%</strong>
                    sits at the
                    <strong style='color:#ffd166;'>
                        {pct_rank:.1f}th percentile</strong>
                    of {n_sims:,} simulated paths —
                    {'meaning history gave us a slightly above-average outcome, but not an unusually lucky one.' if is_beginner
                     else f'confirming the backtest result is not an outlier — {pct_rank:.1f}% of paths underperformed the historical path.'}
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"Monte Carlo unavailable: {e}")

            # ── Asset statistics table ──
            if not is_beginner:
                st.markdown("<div style='margin-top:20px;'></div>",
                            unsafe_allow_html=True)
                section("INDIVIDUAL ASSET STATISTICS")
                try:
                    stats_df = result["stats"].round(3)
                    st.dataframe(stats_df,
                                 use_container_width=True)
                except Exception:
                    pass

# ======================================================================
# SECTION 9: FOOTER
# ======================================================================
st.markdown(f"""
<div style='text-align:center; font-family:JetBrains Mono,monospace;
            font-size:0.6rem; color:#1a3357; margin-top:40px;
            padding-top:16px; border-top:1px solid #1a3357;'>
    AURELINE LABS v2.0
    &nbsp;·&nbsp; FOR RESEARCH PURPOSES ONLY
    &nbsp;·&nbsp; NOT FINANCIAL ADVICE
    &nbsp;·&nbsp; ATENEO DE MANILA UNIVERSITY
    &nbsp;·&nbsp; {'BEGINNER MODE' if is_beginner else 'PROFESSIONAL MODE'}
</div>""", unsafe_allow_html=True)