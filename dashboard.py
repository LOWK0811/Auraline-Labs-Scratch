# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.data_handler import get_price_data
from src.features import build_features, feature_cols
from src.indicators import add_sma, add_atr
from src.risk import calculate_shares


# ======================================================================
# SECTION 2: PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Epoch Quant — Trading Platform",
    page_icon="📈",
    layout="wide"
)


# ======================================================================
# SECTION 3: HELPER FUNCTIONS
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
    drawdown = (values - rolling_peak) / rolling_peak
    return drawdown.min()


def load_paper_account():
    path = "data/paper_account.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"cash": 100000, "positions": {}, "orders": []}


# ======================================================================
# SECTION 4: LOAD DATA (CACHED SO IT DOESN'T RELOAD ON EVERY CLICK)
# ======================================================================
@st.cache_data
def load_and_prepare_data(ticker, start, end, sma_window=20):
    data = get_price_data(ticker, start, end)
    data = add_sma(data, window=sma_window)
    data = add_atr(data)
    data["signal"] = (data["Close"] > data["sma"]).shift(1)
    return data


@st.cache_data
def run_sma_backtest(ticker, start, end):
    data = load_and_prepare_data(ticker, start, end)
    cost_per_trade = 0.001
    starting_cash = 10000
    cash = starting_cash
    shares_held = 0
    portfolio = []

    for i in range(len(data)):
        price_today = data["Close"].iloc[i]
        price_yesterday = data["Close"].iloc[i-1] if i > 0 else price_today
        signal_today = data["signal"].iloc[i]
        atr_today = data["atr"].iloc[i]

        if signal_today == True and shares_held == 0:
            shares = calculate_shares(cash, price_yesterday, atr_today)
            if shares > 0:
                cash -= shares * price_yesterday * (1 + cost_per_trade)
                shares_held = shares
        elif signal_today == False and shares_held > 0:
            cash += shares_held * price_yesterday * (1 - cost_per_trade)
            shares_held = 0

        portfolio.append(cash + shares_held * price_today)

    bh_shares = starting_cash // data["Close"].iloc[0]
    bh = (bh_shares * data["Close"]).tolist()
    return data, portfolio, bh


# ======================================================================
# SECTION 5: SIDEBAR — CONTROLS
# ======================================================================
st.sidebar.title("⚙️ Platform Controls")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
start  = st.sidebar.date_input("Start Date",
         value=pd.Timestamp("2021-01-01")).strftime("%Y-%m-%d")
end    = st.sidebar.date_input("End Date",
         value=pd.Timestamp("2026-06-01")).strftime("%Y-%m-%d")
sma_window = st.sidebar.slider("SMA Window", 5, 100, 20)

st.sidebar.markdown("---")
st.sidebar.markdown("**Platform:** Epoch Quant v0.1")
st.sidebar.markdown("**Built by:** Louis Andre")


# ======================================================================
# SECTION 6: HEADER
# ======================================================================
st.title("📈 Epoch Quant — Trading Platform")
st.markdown(f"Analyzing **{ticker}** from `{start}` to `{end}`")
st.markdown("---")


# ======================================================================
# SECTION 7: LOAD DATA AND RUN BACKTEST
# ======================================================================
with st.spinner("Loading data and running backtest..."):
    try:
        data, portfolio, bh = run_sma_backtest(ticker, start, end)
    except Exception as e:
        st.error(f"Failed to load data for {ticker}: {e}")
        st.stop()


# ======================================================================
# SECTION 8: TOP METRICS ROW
# ======================================================================
final_value  = portfolio[-1]
total_return = (final_value / 10000) - 1
sharpe       = sharpe_ratio(portfolio)
mdd          = max_drawdown(portfolio)
latest_price = data["Close"].iloc[-1]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Latest Price",    f"${latest_price:.2f}")
col2.metric("Final Value",     f"${final_value:,.2f}")
col3.metric("Total Return",    f"{total_return:.2%}")
col4.metric("Sharpe Ratio",    f"{sharpe:.3f}")
col5.metric("Max Drawdown",    f"{mdd:.2%}")


# ======================================================================
# SECTION 9: PRICE CHART WITH SMA
# ======================================================================
st.subheader("📊 Price History & SMA Signal")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(data.index, data["Close"], label="Close Price", linewidth=1)
ax1.plot(data.index, data["sma"],   label=f"SMA({sma_window})",
         linestyle="--", linewidth=1)
ax1.set_ylabel("Price (USD)")
ax1.legend()
st.pyplot(fig1)
plt.close()


# ======================================================================
# SECTION 10: PORTFOLIO PERFORMANCE CHART
# ======================================================================
st.subheader("💰 Strategy vs Buy & Hold")
fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax2.plot(data.index, portfolio, label="SMA Strategy", linewidth=1.5)
ax2.plot(data.index, bh,        label="Buy & Hold",   linewidth=1.5)
ax2.set_ylabel("Portfolio Value (USD)")
ax2.legend()

# Drawdown
port_series  = pd.Series(portfolio)
bh_series    = pd.Series(bh)
port_dd = ((port_series - port_series.cummax()) / port_series.cummax()) * 100
bh_dd   = ((bh_series   - bh_series.cummax())   / bh_series.cummax())   * 100

ax3.fill_between(data.index, port_dd, 0,
                 alpha=0.5, color="blue",   label="Strategy DD")
ax3.fill_between(data.index, bh_dd,   0,
                 alpha=0.5, color="orange", label="B&H DD")
ax3.set_ylabel("Drawdown (%)")
ax3.legend()
plt.tight_layout()
st.pyplot(fig2)
plt.close()


# ======================================================================
# SECTION 11: PAPER ACCOUNT STATUS
# ======================================================================
st.subheader("🏦 Paper Trading Account")
account = load_paper_account()

acol1, acol2, acol3 = st.columns(3)
acol1.metric("Cash",         f"${account['cash']:,.2f}")
acol2.metric("Open Positions", len(account["positions"]))
acol3.metric("Total Orders",   len(account["orders"]))

if account["positions"]:
    st.markdown("**Open Positions:**")
    pos_df = pd.DataFrame([
        {"Ticker": k, "Shares": v["qty"], "Avg Entry": f"${v['avg_price']:.2f}"}
        for k, v in account["positions"].items()
    ])
    st.dataframe(pos_df, use_container_width=True)

if account["orders"]:
    st.markdown("**Recent Orders:**")
    orders_df = pd.DataFrame(account["orders"]).tail(10)
    st.dataframe(orders_df, use_container_width=True)


# ======================================================================
# SECTION 12: RAW DATA EXPLORER
# ======================================================================
st.subheader("🔍 Raw Data Explorer")
if st.checkbox("Show raw OHLCV data"):
    st.dataframe(data[["Open","High","Low","Close","Volume","sma","atr"]]\
                 .tail(20).round(2), use_container_width=True)