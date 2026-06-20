# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import os
import sys
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))

from src.data_handler import get_price_data
from src.indicators import add_sma, add_atr
from src.risk import calculate_shares
from src.features import build_features, feature_cols


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
# SECTION 3: AURELINE LABS VISUAL IDENTITY — CSS
# ======================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
.stApp {
    background-color: #060d1f;
    font-family: 'DM Sans', sans-serif;
    color: #e8f0fe;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d1b35 !important;
    border-right: 1px solid #1a3357;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    background-color: #060d1f !important;
    border: 1px solid #1a3357 !important;
    color: #e8f0fe !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 4px;
}
[data-testid="stSidebar"] .stTextInput input:focus,
[data-testid="stSidebar"] .stTextArea textarea:focus {
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 1px #00d4aa22 !important;
}

/* ── Typography ── */
h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
    color: #e8f0fe !important;
    letter-spacing: -1px !important;
    line-height: 1.1 !important;
}
h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #e8f0fe !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color: #0f2040;
    border: 1px solid #1a3357;
    border-top: 2px solid #00d4aa;
    border-radius: 6px;
    padding: 18px 20px 14px 20px;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    color: #7b9bc0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #00d4aa !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* ── Section labels — the signature element ── */
.al-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    color: #00d4aa;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    padding-bottom: 6px;
    border-bottom: 1px solid #00d4aa;
    margin-bottom: 12px;
    display: block;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #1a3357 !important;
    margin: 16px 0 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #0d1b35;
    border: 1px solid #1a3357;
    border-radius: 6px;
    padding: 4px;
    gap: 2px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    color: #7b9bc0;
    border-radius: 4px;
    letter-spacing: 0.05em;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background-color: #00d4aa !important;
    color: #060d1f !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a3357 !important;
    border-radius: 6px;
}

/* ── Buttons ── */
.stButton button {
    background-color: #00d4aa !important;
    color: #060d1f !important;
    border: none !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 4px !important;
    padding: 10px 24px !important;
}
.stButton button:hover {
    background-color: #00f0c4 !important;
}

/* ── Info/alert boxes ── */
[data-testid="stAlert"] {
    background-color: #0f2040 !important;
    border: 1px solid #1a3357 !important;
    border-left: 3px solid #00d4aa !important;
    border-radius: 4px !important;
    color: #7b9bc0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Slider ── */
[data-baseweb="slider"] [data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #00d4aa !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label p {
    font-family: 'DM Sans', sans-serif !important;
    color: #7b9bc0 !important;
    font-size: 0.85rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    font-family: 'JetBrains Mono', monospace !important;
    color: #7b9bc0 !important;
    font-size: 0.78rem !important;
}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# SECTION 4: HELPER FUNCTIONS
# ======================================================================
def sharpe_ratio(portfolio_values, risk_free_rate=0.05):
    values = pd.Series(portfolio_values)
    daily_returns = values.pct_change().dropna()
    daily_rf = risk_free_rate / 252
    excess = daily_returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(252)


def max_drawdown(portfolio_values):
    values = pd.Series(portfolio_values)
    rolling_peak = values.cummax()
    return ((values - rolling_peak) / rolling_peak).min()


def cagr(portfolio_values, start, end):
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    if years <= 0:
        return 0.0
    return (portfolio_values[-1] / portfolio_values[0]) ** (1 / years) - 1


def win_rate(portfolio_values):
    values = pd.Series(portfolio_values)
    daily_returns = values.pct_change().dropna()
    return (daily_returns > 0).mean()


def aureline_style(ax, fig):
    """Apply Aureline Labs dark chart style."""
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


def load_paper_account():
    path = "data/paper_account.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"cash": 100000, "positions": {}, "orders": []}


def section(label):
    st.markdown(f'<span class="al-section">{label}</span>',
                unsafe_allow_html=True)


# ======================================================================
# SECTION 5: BACKTEST FUNCTIONS (CACHED)
# ======================================================================
@st.cache_data
def run_sma_backtest(ticker, start, end, sma_window=20):
    data = get_price_data(ticker, start, end)
    if data is None or data.empty:
        return None, None, None
    data = add_sma(data, window=sma_window)
    data = add_atr(data)
    data["signal"] = (data["Close"] > data["sma"]).shift(1)

    cash = 10000
    shares_held = 0
    portfolio = []

    for i in range(len(data)):
        price_today     = data["Close"].iloc[i]
        price_yesterday = data["Close"].iloc[i-1] if i > 0 else price_today
        signal_today    = data["signal"].iloc[i]
        atr_today       = data["atr"].iloc[i]

        if signal_today == True and shares_held == 0:
            shares = calculate_shares(cash, price_yesterday, atr_today)
            if shares > 0:
                cash -= shares * price_yesterday * 1.001
                shares_held = shares
        elif signal_today == False and shares_held > 0:
            cash += shares_held * price_yesterday * 0.999
            shares_held = 0

        portfolio.append(cash + shares_held * price_today)

    bh_shares = 10000 // data["Close"].iloc[0]
    bh = (bh_shares * data["Close"]).tolist()
    return data, portfolio, bh


@st.cache_data
def run_ml_backtest(ticker, start, end):
    data = get_price_data(ticker, start, end)
    if data is None or data.empty:
        return None, None
    data = add_atr(data)
    df   = build_features(data)
    df   = df.dropna(subset=feature_cols() + ["label"])
    if len(df) < 600:
        return None, None

    min_train = 500
    signals   = []

    for i in range(min_train, len(df) - 1):
        train = df.iloc[:i]
        model = RandomForestClassifier(
            n_estimators=50, max_depth=4,
            min_samples_leaf=20, random_state=42
        )
        model.fit(train[feature_cols()], train["label"])
        prob = model.predict_proba(df[feature_cols()].iloc[[i]])[0][1]
        signals.append({
            "date":   df.index[i],
            "signal": 1 if prob >= 0.55 else 0
        })

    signals_df = pd.DataFrame(signals).set_index("date")
    bt_data    = df.loc[signals_df.index]

    cash = 10000
    shares_held = 0
    portfolio   = []

    for i in range(len(bt_data)):
        price_today     = bt_data["Close"].iloc[i]
        price_yesterday = bt_data["Close"].iloc[i-1] if i > 0 else price_today
        atr_today       = bt_data["atr"].iloc[i]
        signal_today    = signals_df["signal"].iloc[i]

        if signal_today == 1 and shares_held == 0:
            shares = calculate_shares(cash, price_yesterday, atr_today)
            if shares > 0:
                cash -= shares * price_yesterday * 1.001
                shares_held = shares
        elif signal_today == 0 and shares_held > 0:
            cash += shares_held * price_yesterday * 0.999
            shares_held = 0

        portfolio.append(cash + shares_held * price_today)

    return bt_data.index, portfolio


@st.cache_data
def run_multi_ticker(tickers, start, end, sma_window):
    results = []
    for t in tickers:
        data, portfolio, bh = run_sma_backtest(t, start, end, sma_window)
        if portfolio is None:
            continue
        results.append({
            "Ticker":     t,
            "Return":     (portfolio[-1] / 10000) - 1,
            "CAGR":       cagr(portfolio, start, end),
            "Sharpe":     sharpe_ratio(portfolio),
            "Max DD":     max_drawdown(portfolio),
            "Win Rate":   win_rate(portfolio),
            "B&H Return": (bh[-1] / 10000) - 1,
            "Beat B&H":   "✓" if portfolio[-1] > bh[-1] else "✗",
            "portfolio":  portfolio,
        })
    return results


# ======================================================================
# SECTION 6: SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 24px 0;'>
        <div style='font-family: Space Grotesk, sans-serif;
                    font-size: 1.3rem; font-weight: 700;
                    color: #e8f0fe; letter-spacing: -0.5px;
                    line-height: 1;'>
            ⬡ &nbsp;Aureline Labs
        </div>
        <div style='font-family: JetBrains Mono, monospace;
                    font-size: 0.6rem; color: #00d4aa;
                    letter-spacing: 0.2em; margin-top: 6px;'>
            QUANT RESEARCH PLATFORM v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family: JetBrains Mono, monospace;
                font-size: 0.6rem; color: #7b9bc0;
                letter-spacing: 0.15em; margin-bottom: 8px;'>
        STRATEGY CONTROLS
    </div>
    """, unsafe_allow_html=True)

    ticker     = st.text_input("Ticker Symbol", value="AAPL").upper()
    start_date = st.date_input("Start Date",
                  value=pd.Timestamp("2021-01-01")).strftime("%Y-%m-%d")
    end_date   = st.date_input("End Date",
                  value=pd.Timestamp("2026-06-01")).strftime("%Y-%m-%d")
    sma_window = st.slider("SMA Window", 5, 100, 20)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family: JetBrains Mono, monospace;
                font-size: 0.6rem; color: #7b9bc0;
                letter-spacing: 0.15em; margin-bottom: 8px;'>
        WATCHLIST
    </div>
    """, unsafe_allow_html=True)

    ticker_input = st.text_area("Tickers (comma-separated)",
                                 value="AAPL,MSFT,NVDA,JPM,XOM,TSLA,SPY",
                                 height=70)
    watchlist = [t.strip().upper() for t in ticker_input.split(",")
                 if t.strip()]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family: JetBrains Mono, monospace;
                font-size: 0.62rem; color: #1a3357;
                line-height: 2; margin-top: 4px;'>
        <span style='color:#00d4aa;'>RESEARCHER</span><br>
        Louis Andre<br>
        <span style='color:#00d4aa;'>INSTITUTION</span><br>
        Ateneo de Manila University<br>
        <span style='color:#00d4aa;'>PROGRAM</span><br>
        BS Applied Mathematics<br>
        Mathematical Finance
    </div>
    """, unsafe_allow_html=True)


# ======================================================================
# SECTION 7: HEADER
# ======================================================================
col_title, col_badges = st.columns([3, 1])

with col_title:
    st.markdown("""
    <div style='padding-top: 8px;'>
        <div style='font-family: Space Grotesk, sans-serif;
                    font-size: 2.2rem; font-weight: 700;
                    color: #e8f0fe; letter-spacing: -1.5px;
                    line-height: 1;'>
            Aureline Labs
        </div>
        <div style='font-family: JetBrains Mono, monospace;
                    font-size: 0.72rem; color: #7b9bc0;
                    margin-top: 6px; letter-spacing: 0.05em;'>
            Quantitative Research &amp; Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_badges:
    st.markdown("""
    <div style='display: flex; gap: 6px; justify-content: flex-end;
                padding-top: 12px; flex-wrap: wrap;'>
        <span style='font-family: JetBrains Mono, monospace;
                     font-size: 0.6rem; font-weight: 700;
                     color: #060d1f; background: #00d4aa;
                     padding: 3px 8px; border-radius: 3px;
                     letter-spacing: 0.1em;'>
            PAPER TRADING
        </span>
        <span style='font-family: JetBrains Mono, monospace;
                     font-size: 0.6rem; color: #7b9bc0;
                     border: 1px solid #1a3357;
                     padding: 3px 8px; border-radius: 3px;
                     letter-spacing: 0.1em;'>
            v1.0
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div style='font-family: JetBrains Mono, monospace;
            font-size: 0.72rem; color: #1a3357;
            margin: 8px 0 0 0; padding-bottom: 16px;
            border-bottom: 1px solid #1a3357;'>
    <span style='color:#7b9bc0;'>{ticker}</span>
    &nbsp;·&nbsp;
    {start_date} → {end_date}
    &nbsp;·&nbsp;
    SMA({sma_window})
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)


# ======================================================================
# SECTION 8: TABS
# ======================================================================
tab1, tab2, tab3 = st.tabs([
    "SINGLE TICKER",
    "MULTI-TICKER",
    "ML vs SMA"
])


# ======================================================================
# TAB 1: SINGLE TICKER
# ======================================================================
with tab1:
    with st.spinner(f"Loading {ticker}..."):
        data, portfolio, bh = run_sma_backtest(
            ticker, start_date, end_date, sma_window)

    if data is None:
        st.error(f"No data found for **{ticker}**. "
                 f"Verify the ticker symbol.")
    else:
        latest_price  = data["Close"].iloc[-1]
        final_value   = portfolio[-1]
        total_return  = (final_value / 10000) - 1
        sharpe        = sharpe_ratio(portfolio)
        mdd           = max_drawdown(portfolio)
        bh_return     = (bh[-1] / 10000) - 1
        strategy_cagr = cagr(portfolio, start_date, end_date)

        section("Performance Overview")

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Latest Price",    f"${latest_price:.2f}")
        c2.metric("Final Value",     f"${final_value:,.2f}")
        c3.metric("Total Return",    f"{total_return:+.2%}")
        c4.metric("CAGR",            f"{strategy_cagr:+.2%}")
        c5.metric("Sharpe Ratio",    f"{sharpe:.3f}")
        c6.metric("Max Drawdown",    f"{mdd:.2%}")

        st.markdown("<div style='margin-top:24px'></div>",
                    unsafe_allow_html=True)
        section("Price History & SMA Signal")

        fig1, ax1 = plt.subplots(figsize=(13, 3.5))
        aureline_style(ax1, fig1)
        ax1.plot(data.index, data["Close"],
                 color="#00d4aa", linewidth=1.3, label="Close")
        ax1.plot(data.index, data["sma"],
                 color="#1a6eff", linewidth=1,
                 linestyle="--", alpha=0.7,
                 label=f"SMA({sma_window})")
        signal = data["signal"].fillna(False).astype(bool)
        ax1.fill_between(data.index,
                         data["Close"].min(),
                         data["Close"].max(),
                         where=signal,
                         alpha=0.05, color="#00d4aa",
                         label="In Market")
        ax1.yaxis.set_major_formatter(
            mticker.StrMethodFormatter("${x:,.0f}"))
        legend = ax1.legend(
            fontsize=8, facecolor="#0f2040",
            edgecolor="#1a3357", labelcolor="#7b9bc0")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        st.markdown("<div style='margin-top:24px'></div>",
                    unsafe_allow_html=True)
        section("Strategy vs Benchmark")

        fig2, (ax2, ax3) = plt.subplots(
            2, 1, figsize=(13, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]})
        aureline_style(ax2, fig2)
        aureline_style(ax3, fig2)

        ax2.plot(data.index, portfolio,
                 color="#00d4aa", linewidth=1.5,
                 label="Aureline SMA Strategy")
        ax2.plot(data.index, bh,
                 color="#1a6eff", linewidth=1.2,
                 linestyle="--", alpha=0.6,
                 label="Buy & Hold")
        ax2.yaxis.set_major_formatter(
            mticker.StrMethodFormatter("${x:,.0f}"))
        ax2.set_ylabel("Portfolio Value", fontsize=8,
                       color="#7b9bc0")
        ax2.legend(fontsize=8, facecolor="#0f2040",
                   edgecolor="#1a3357", labelcolor="#7b9bc0")

        ps = pd.Series(portfolio)
        bs = pd.Series(bh)
        ax3.fill_between(data.index,
                         ((ps-ps.cummax())/ps.cummax())*100, 0,
                         alpha=0.7, color="#00d4aa",
                         label="Strategy DD")
        ax3.fill_between(data.index,
                         ((bs-bs.cummax())/bs.cummax())*100, 0,
                         alpha=0.3, color="#1a6eff",
                         label="B&H DD")
        ax3.set_ylabel("Drawdown %", fontsize=8,
                       color="#7b9bc0")
        ax3.legend(fontsize=8, facecolor="#0f2040",
                   edgecolor="#1a3357", labelcolor="#7b9bc0")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown("<hr>", unsafe_allow_html=True)
        section("Paper Account")
        account = load_paper_account()

        a1, a2, a3 = st.columns(3)
        a1.metric("Cash",           f"${account['cash']:,.2f}")
        a2.metric("Open Positions",  len(account["positions"]))
        a3.metric("Orders Placed",   len(account["orders"]))

        if account["positions"]:
            pos_df = pd.DataFrame([
                {"Ticker": k,
                 "Shares": v["qty"],
                 "Avg Entry": f"${v['avg_price']:.2f}"}
                for k, v in account["positions"].items()
            ])
            st.dataframe(pos_df, use_container_width=True,
                         hide_index=True)

        if st.checkbox("Show raw OHLCV data (last 20 sessions)"):
            st.dataframe(
                data[["Open","High","Low","Close",
                       "Volume","sma","atr"]]
                .tail(20).round(2),
                use_container_width=True)


# ======================================================================
# TAB 2: MULTI-TICKER
# ======================================================================
with tab2:
    section("Watchlist Backtest — Strategy Comparison")

    with st.spinner("Running multi-ticker backtest..."):
        multi_results = run_multi_ticker(
            tuple(watchlist), start_date, end_date, sma_window)

    if not multi_results:
        st.error("No results returned. Check watchlist tickers.")
    else:
        display_df = pd.DataFrame([{
            "Ticker":     r["Ticker"],
            "Return":     f"{r['Return']:+.2%}",
            "CAGR":       f"{r['CAGR']:+.2%}",
            "Sharpe":     f"{r['Sharpe']:.3f}",
            "Max DD":     f"{r['Max DD']:.2%}",
            "Win Rate":   f"{r['Win Rate']:.1%}",
            "B&H Return": f"{r['B&H Return']:+.2%}",
            "Beat B&H":   r["Beat B&H"]
        } for r in multi_results])

        st.dataframe(display_df, use_container_width=True,
                     hide_index=True)

        beat      = sum(1 for r in multi_results
                        if r["Return"] > r["B&H Return"])
        avg_sharpe = np.mean([r["Sharpe"] for r in multi_results])

        m1, m2, m3 = st.columns(3)
        m1.metric("Beat Buy & Hold",   f"{beat}/{len(multi_results)}")
        m2.metric("Avg Sharpe Ratio",  f"{avg_sharpe:.3f}")
        m3.metric("Tickers Analyzed",  len(multi_results))

        st.markdown("<div style='margin-top:24px'></div>",
                    unsafe_allow_html=True)
        section("Normalized Portfolio Curves")

        teals = ["#00d4aa","#1a6eff","#00a8ff","#7b61ff",
                 "#ff6b6b","#ffd166","#06d6a0","#118ab2"]

        fig3, ax3 = plt.subplots(figsize=(13, 5))
        aureline_style(ax3, fig3)

        for idx, r in enumerate(multi_results):
            norm = [v/10000 for v in r["portfolio"]]
            ax3.plot(norm,
                     color=teals[idx % len(teals)],
                     linewidth=1.5,
                     label=r["Ticker"],
                     alpha=0.9)

        ax3.axhline(y=1.0, color="#1a3357",
                    linestyle="--", linewidth=1)
        ax3.set_ylabel("Return Multiple (×)", fontsize=8,
                       color="#7b9bc0")
        ax3.set_xlabel("Trading Days", fontsize=8,
                       color="#7b9bc0")
        ax3.legend(fontsize=8, facecolor="#0f2040",
                   edgecolor="#1a3357",
                   labelcolor="#7b9bc0", ncol=2)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()


# ======================================================================
# TAB 3: ML vs SMA
# ======================================================================
with tab3:
    section(f"Walk-Forward ML Model vs SMA({sma_window}) — {ticker}")

    st.markdown("""
    <div style='font-family: JetBrains Mono, monospace;
                font-size: 0.72rem; color: #7b9bc0;
                background: #0f2040; border: 1px solid #1a3357;
                border-left: 2px solid #00d4aa;
                padding: 12px 16px; border-radius: 4px;
                margin-bottom: 20px; line-height: 1.7;'>
        Walk-forward training fits a new model at every time step using
        only past data — approximately 800 iterations. Results are
        cached after the first run on each ticker.
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶  RUN ML BACKTEST"):
        with st.spinner("Running walk-forward ML training..."):
            ml_dates, ml_portfolio = run_ml_backtest(
                ticker, start_date, end_date)

        if ml_portfolio is None:
            st.error("Insufficient data for ML backtest. "
                     "Try a longer date range.")
        else:
            _, sma_portfolio, _ = run_sma_backtest(
                ticker, start_date, end_date, sma_window)
            sma_aligned = sma_portfolio[-len(ml_portfolio):]

            ml_ret  = (ml_portfolio[-1]  / 10000) - 1
            sma_ret = (sma_aligned[-1]   / 10000) - 1
            ml_sr   = sharpe_ratio(ml_portfolio)
            sma_sr  = sharpe_ratio(sma_aligned)
            ml_mdd  = max_drawdown(ml_portfolio)
            sma_mdd = max_drawdown(sma_aligned)

            section("Head-to-Head Metrics")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div style='font-family:JetBrains Mono,monospace;
                            font-size:0.65rem; color:#00d4aa;
                            letter-spacing:0.15em; margin-bottom:8px;'>
                    ML MODEL (RANDOM FOREST)
                </div>""", unsafe_allow_html=True)
                st.metric("Return",       f"{ml_ret:+.2%}")
                st.metric("Sharpe Ratio", f"{ml_sr:.3f}")
                st.metric("Max Drawdown", f"{ml_mdd:.2%}")

            with c2:
                st.markdown(f"""
                <div style='font-family:JetBrains Mono,monospace;
                            font-size:0.65rem; color:#1a6eff;
                            letter-spacing:0.15em; margin-bottom:8px;'>
                    SMA({sma_window}) BASELINE
                </div>""", unsafe_allow_html=True)
                st.metric("Return",       f"{sma_ret:+.2%}",
                          delta=f"{ml_ret-sma_ret:+.2%} ML edge")
                st.metric("Sharpe Ratio", f"{sma_sr:.3f}",
                          delta=f"{ml_sr-sma_sr:+.3f} ML edge")
                st.metric("Max Drawdown", f"{sma_mdd:.2%}",
                          delta=f"{ml_mdd-sma_mdd:+.2%} ML edge")

            st.markdown("<div style='margin-top:24px'></div>",
                        unsafe_allow_html=True)
            section("Equity Curves")

            fig4, ax4 = plt.subplots(figsize=(13, 4))
            aureline_style(ax4, fig4)
            ax4.plot(ml_portfolio,
                     color="#00d4aa", linewidth=1.5,
                     label="ML Strategy")
            ax4.plot(sma_aligned,
                     color="#1a6eff", linewidth=1.2,
                     linestyle="--", alpha=0.7,
                     label=f"SMA({sma_window})")
            ax4.yaxis.set_major_formatter(
                mticker.StrMethodFormatter("${x:,.0f}"))
            ax4.set_ylabel("Portfolio Value", fontsize=8,
                           color="#7b9bc0")
            ax4.legend(fontsize=8, facecolor="#0f2040",
                       edgecolor="#1a3357",
                       labelcolor="#7b9bc0")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px;
                    font-family:JetBrains Mono,monospace;
                    font-size:0.78rem; color:#1a3357;
                    background:#0f2040;
                    border:1px solid #1a3357;
                    border-radius:6px;'>
            Press RUN ML BACKTEST to begin walk-forward analysis.
        </div>
        """, unsafe_allow_html=True)


# ======================================================================
# SECTION 9: FOOTER
# ======================================================================
st.markdown("""
<div style='text-align:center;
            font-family:JetBrains Mono,monospace;
            font-size:0.6rem; color:#1a3357;
            margin-top:40px; padding-top:16px;
            border-top:1px solid #1a3357;'>
    AURELINE LABS v1.0
    &nbsp;·&nbsp;
    FOR RESEARCH PURPOSES ONLY
    &nbsp;·&nbsp;
    NOT FINANCIAL ADVICE
    &nbsp;·&nbsp;
    ATENEO DE MANILA UNIVERSITY
</div>
""", unsafe_allow_html=True)