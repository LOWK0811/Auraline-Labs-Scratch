# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.data_handler import get_price_data
from src.regime_detector import RegimeDetector, REGIME_COLORS
from src.indicators import add_sma, add_atr
from src.risk import calculate_shares


# ======================================================================
# SECTION 2: LOGGING SETUP
# ======================================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 3: DETECT REGIMES ON AAPL
# ======================================================================
data     = get_price_data("AAPL", "2021-01-01", "2026-06-01")
detector = RegimeDetector()
df       = detector.detect(data)

stats = detector.summarize(df)


# ======================================================================
# SECTION 4: REGIME-AWARE BACKTEST
# ======================================================================
def run_regime_aware_backtest(df, active_regimes,
                               sma_window=20,
                               cost_per_trade=0.001,
                               starting_cash=10000):
    """
    Only trades when the current regime is in active_regimes.
    Sits in cash during regimes not on the approved list.
    """
    df = df.copy()
    df = add_sma(df, window=sma_window)
    df = add_atr(df)
    df["signal"] = (df["Close"] > df["sma"]).shift(1)

    cash        = starting_cash
    shares_held = 0
    portfolio   = []
    num_trades  = 0
    days_active = 0

    for i in range(len(df)):
        price_today     = df["Close"].iloc[i]
        price_yesterday = df["Close"].iloc[i-1] if i > 0 else price_today
        signal_today    = df["signal"].iloc[i]
        atr_today       = df["atr"].iloc[i]
        regime_today    = df["regime"].iloc[i]

        # Only trade in approved regimes
        in_approved_regime = regime_today in active_regimes

        if in_approved_regime:
            days_active += 1
            if signal_today == True and shares_held == 0:
                shares = calculate_shares(
                    cash, price_yesterday, atr_today)
                if shares > 0:
                    cash -= shares * price_yesterday * \
                            (1 + cost_per_trade)
                    shares_held = shares
                    num_trades += 1
        else:
            # Force exit when entering a non-approved regime
            if shares_held > 0:
                cash += shares_held * price_yesterday * \
                        (1 - cost_per_trade)
                shares_held = 0
                num_trades += 1

        # Normal sell signal in approved regime
        if in_approved_regime and signal_today == False \
           and shares_held > 0:
            cash += shares_held * price_yesterday * \
                    (1 - cost_per_trade)
            shares_held = 0
            num_trades += 1

        portfolio.append(cash + shares_held * price_today)

    return portfolio, num_trades, days_active


# ======================================================================
# SECTION 5: COMPARE STRATEGIES
# ======================================================================
# Baseline: trade in all regimes (same as before)
portfolio_all, trades_all, days_all = run_regime_aware_backtest(
    df,
    active_regimes={"BULL_TRENDING", "BEAR_TRENDING",
                    "HIGH_VOLATILITY", "SIDEWAYS"}
)

# Regime-filtered: only trade in bull trending
portfolio_bull, trades_bull, days_bull = run_regime_aware_backtest(
    df,
    active_regimes={"BULL_TRENDING"}
)

# Conservative: trade in bull and sideways, avoid bear and high vol
portfolio_safe, trades_safe, days_safe = run_regime_aware_backtest(
    df,
    active_regimes={"BULL_TRENDING", "SIDEWAYS"}
)

# Buy and hold benchmark
bh_shares   = 10000 // df["Close"].iloc[0]
bh_portfolio = (bh_shares * df["Close"]).tolist()


def metrics(portfolio, label, trades, days_active):
    ret    = (portfolio[-1] / 10000 - 1) * 100
    values = pd.Series(portfolio)
    dr     = values.pct_change().dropna()
    excess = dr - 0.05/252
    sharpe = round((excess.mean()/excess.std())*np.sqrt(252), 3) \
             if excess.std() > 0 else 0.0
    peak   = values.cummax()
    mdd    = round(((values-peak)/peak).min()*100, 2)
    print(f"  {label:<30} {ret:>+8.2f}%  "
          f"{sharpe:>8.3f}  {mdd:>9.2f}%  "
          f"{trades:>8}  {days_active:>10}")


print(f"\n{'='*80}")
print(f"  REGIME-AWARE STRATEGY COMPARISON — AAPL 2021–2026")
print(f"{'='*80}")
print(f"  {'Strategy':<30} {'Return':>9} {'Sharpe':>9} "
      f"{'Max DD':>10} {'Trades':>9} {'Days Active':>11}")
print(f"  {'-'*75}")
metrics(portfolio_all,  "SMA All Regimes",      trades_all,  days_all)
metrics(portfolio_bull, "SMA Bull Only",         trades_bull, days_bull)
metrics(portfolio_safe, "SMA Bull + Sideways",   trades_safe, days_safe)
metrics(bh_portfolio,   "Buy & Hold",
        1, len(df))


# ======================================================================
# SECTION 6: VISUALIZE REGIMES ON PRICE CHART
# ======================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)
fig.patch.set_facecolor("#060d1f")

for ax in [ax1, ax2]:
    ax.set_facecolor("#060d1f")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1a3357")
    ax.spines["bottom"].set_color("#1a3357")
    ax.tick_params(colors="#7b9bc0", labelsize=8)
    ax.grid(True, color="#0d1b35", linewidth=0.8, linestyle="--")

# Shade background by regime
prev_regime = None
start_idx   = 0

for i, (date, row) in enumerate(df.iterrows()):
    regime = row["regime"]
    if regime != prev_regime or i == len(df) - 1:
        if prev_regime is not None:
            color = REGIME_COLORS.get(prev_regime, "#1a3357")
            ax1.axvspan(df.index[start_idx], date,
                        alpha=0.15, color=color, linewidth=0)
            ax2.axvspan(df.index[start_idx], date,
                        alpha=0.25, color=color, linewidth=0)
        start_idx  = i
        prev_regime = regime

# Price line
ax1.plot(df.index, df["Close"],
         color="#e8f0fe", linewidth=1.2,
         label="AAPL Close")
ax1.set_ylabel("Price (USD)", color="#7b9bc0", fontsize=9)
ax1.yaxis.set_major_formatter(
    plt.matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))

# Portfolio curves
ax2.plot(portfolio_all,  color="#7b9bc0", linewidth=1,
         linestyle="--", label="All Regimes")
ax2.plot(portfolio_safe, color="#00d4aa", linewidth=1.5,
         label="Bull + Sideways")
ax2.plot(portfolio_bull, color="#ffd166", linewidth=1,
         label="Bull Only")
ax2.set_ylabel("Portfolio ($)", color="#7b9bc0", fontsize=9)
ax2.yaxis.set_major_formatter(
    plt.matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))

# Legend for regimes
patches = [
    mpatches.Patch(color=REGIME_COLORS["BULL_TRENDING"],
                   label="Bull Trending", alpha=0.6),
    mpatches.Patch(color=REGIME_COLORS["BEAR_TRENDING"],
                   label="Bear Trending", alpha=0.6),
    mpatches.Patch(color=REGIME_COLORS["HIGH_VOLATILITY"],
                   label="High Volatility", alpha=0.6),
    mpatches.Patch(color=REGIME_COLORS["SIDEWAYS"],
                   label="Sideways", alpha=0.6),
]
ax1.legend(handles=patches, loc="upper left",
           fontsize=8, facecolor="#0f2040",
           edgecolor="#1a3357", labelcolor="#e8f0fe",
           framealpha=0.9)
ax2.legend(loc="upper left", fontsize=8,
           facecolor="#0f2040", edgecolor="#1a3357",
           labelcolor="#e8f0fe")

fig.suptitle(
    "Aureline Labs — Market Regime Detection\n"
    "AAPL 2021–2026 · Rule-Based Classification",
    color="#e8f0fe", fontsize=11,
    fontfamily="monospace", y=1.01
)
plt.tight_layout()
plt.savefig("data/regime_analysis.png",
            dpi=150, bbox_inches="tight",
            facecolor="#060d1f")
plt.show()
logger.info("Chart saved to data/regime_analysis.png")