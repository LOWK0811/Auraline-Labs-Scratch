# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_auc_score
from src.regime_ml import RegimeConditionalML
from src.features import all_feature_cols
from src.data_handler import get_price_data
from src.indicators import add_atr
from src.features import build_all_features
from src.regime_detector import RegimeDetector
from src.risk import calculate_shares
from sklearn.ensemble import RandomForestClassifier


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
# SECTION 3: CONFIG
# ======================================================================
TICKER     = "AAPL"
START      = "2021-01-01"
END        = "2026-06-01"
SPLIT_DATE = "2024-01-01"
STARTING_CASH = 10000


# ======================================================================
# SECTION 4: BUILD REGIME-CONDITIONAL ML STRATEGY
# ======================================================================
engine = RegimeConditionalML()
df, raw_data = engine.prepare_data(TICKER, START, END)

engine.train(df, SPLIT_DATE)

signals_df = engine.generate_signals(df, SPLIT_DATE)
portfolio_regime_ml, trades_regime, regime_days = \
    engine.backtest(raw_data, signals_df)


# ======================================================================
# SECTION 5: BUILD GLOBAL ML BASELINE FOR COMPARISON
# ======================================================================
logger.info("Training global ML baseline...")

train_global = df[df.index < SPLIT_DATE]
test_global  = df[df.index >= SPLIT_DATE]
feature_cols = all_feature_cols()

global_model = RandomForestClassifier(
    n_estimators=100, max_depth=4,
    min_samples_leaf=20, random_state=42
)
global_model.fit(train_global[feature_cols], train_global["label"])

global_proba   = global_model.predict_proba(test_global[feature_cols])[:,1]
global_signals = pd.DataFrame(
    {"signal": (global_proba >= 0.55).astype(int)},
    index=test_global.index
)

raw_atr = add_atr(raw_data)
bt_global = raw_atr.loc[test_global.index]

cash = STARTING_CASH
shares_held = 0
portfolio_global = []

for i in range(len(bt_global)):
    price_today     = bt_global["Close"].iloc[i]
    price_yesterday = bt_global["Close"].iloc[i-1] \
                      if i > 0 else price_today
    atr_today       = bt_global["atr"].iloc[i]
    signal_today    = global_signals["signal"].iloc[i]

    if signal_today == 1 and shares_held == 0:
        shares = calculate_shares(cash, price_yesterday, atr_today)
        if shares > 0:
            cash -= shares * price_yesterday * 1.001
            shares_held = shares
    elif signal_today == 0 and shares_held > 0:
        cash += shares_held * price_yesterday * 0.999
        shares_held = 0

    portfolio_global.append(cash + shares_held * price_today)

# ROC-AUC for both
regime_preds  = []
regime_labels = []
for i in range(len(test_global)):
    regime  = signals_df["regime"].iloc[i]
    if regime in engine.models:
        X = test_global[feature_cols].iloc[[i]]
        prob = engine.models[regime].predict_proba(X)[0][1]
    else:
        prob = 0.5
    regime_preds.append(prob)
    regime_labels.append(test_global["label"].iloc[i])

regime_auc = roc_auc_score(regime_labels, regime_preds)
global_auc = roc_auc_score(
    test_global["label"], global_proba)


# ======================================================================
# SECTION 6: BUY AND HOLD BASELINE
# ======================================================================
bh_data    = raw_atr.loc[test_global.index]
bh_shares  = STARTING_CASH // bh_data["Close"].iloc[0]
bh_portfolio = (bh_shares * bh_data["Close"]).tolist()


# ======================================================================
# SECTION 7: METRICS
# ======================================================================
def compute_metrics(portfolio, label):
    s  = pd.Series(portfolio)
    dr = s.pct_change().dropna()
    ex = dr - 0.05/252
    sh = round((ex.mean()/ex.std())*np.sqrt(252),3) \
         if ex.std()>0 else 0.0
    peak = s.cummax()
    mdd  = round(((s-peak)/peak).min()*100,2)
    ret  = round((s.iloc[-1]/STARTING_CASH-1)*100,2)
    return {"label":label, "return":ret,
            "sharpe":sh, "mdd":mdd}

results = [
    compute_metrics(portfolio_regime_ml, "Regime-Conditional ML"),
    compute_metrics(portfolio_global,    "Global ML"),
    compute_metrics(bh_portfolio,        "Buy & Hold"),
]


# ======================================================================
# SECTION 8: PRINT RESULTS
# ======================================================================
print(f"\n{'='*65}")
print(f"  AURELINE LABS — REGIME-CONDITIONAL ML STRATEGY")
print(f"{'='*65}")
print(f"  Asset: {TICKER} | Test Period: {SPLIT_DATE} → {END}")

print(f"\n  REGIME MODEL TRAINING STATS")
print(f"  {'-'*55}")
for regime, stats in engine.regime_stats.items():
    print(f"  {regime:<20} "
          f"Train rows: {stats['train_rows']:>4} | "
          f"Train AUC: {stats['train_auc']:.4f} | "
          f"Up days: {stats['up_day_pct']}%")

print(f"\n  TEST PERIOD ROC-AUC")
print(f"  Regime-Conditional: {regime_auc:.4f}")
print(f"  Global ML:          {global_auc:.4f}")

print(f"\n  REGIME DISTRIBUTION (Test Period)")
total_test = sum(regime_days.values())
for regime, count in regime_days.items():
    pct = count/total_test*100 if total_test > 0 else 0
    active = "ACTIVE" if regime in engine.models else "SIT OUT"
    print(f"  {regime:<20} {count:>4} days "
          f"({pct:.1f}%) — {active}")

print(f"\n  STRATEGY COMPARISON")
print(f"  {'-'*55}")
print(f"  {'Strategy':<25} {'Return':>9} "
      f"{'Sharpe':>9} {'Max DD':>10}")
print(f"  {'-'*55}")
for r in results:
    print(f"  {r['label']:<25} {r['return']:>+8.2f}%  "
          f"{r['sharpe']:>8.3f}  {r['mdd']:>9.2f}%")
print(f"{'='*65}")


# ======================================================================
# SECTION 9: VISUALIZE
# ======================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=False,
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
    ax.grid(True, color="#0d1b35",
            linewidth=0.8, linestyle="--")

ax1.plot(portfolio_regime_ml, color="#00d4aa",
         linewidth=1.8, label="Regime-Conditional ML")
ax1.plot(portfolio_global,    color="#1a6eff",
         linewidth=1.2, linestyle="--",
         alpha=0.7,     label="Global ML")
ax1.plot(bh_portfolio,        color="#7b9bc0",
         linewidth=1.0, linestyle=":",
         alpha=0.6,     label="Buy & Hold")

ax1.yaxis.set_major_formatter(
    mticker.StrMethodFormatter("${x:,.0f}"))
ax1.set_ylabel("Portfolio Value", color="#7b9bc0", fontsize=9)
ax1.legend(fontsize=8, facecolor="#0f2040",
           edgecolor="#1a3357", labelcolor="#e8f0fe")

# Regime shading on bottom panel
from src.regime_detector import REGIME_COLORS
regime_seq = signals_df["regime"].values
prev = None
start_i = 0
for i, r in enumerate(regime_seq):
    if r != prev or i == len(regime_seq)-1:
        if prev is not None:
            color = REGIME_COLORS.get(prev, "#1a3357")
            ax2.axvspan(start_i, i, alpha=0.4,
                        color=color, linewidth=0)
        start_i = i
        prev = r

ax2.set_ylabel("Regime", color="#7b9bc0", fontsize=9)
ax2.set_yticks([])

import matplotlib.patches as mpatches
patches = [
    mpatches.Patch(color=REGIME_COLORS["BULL_TRENDING"],
                   label="Bull (Active)", alpha=0.6),
    mpatches.Patch(color=REGIME_COLORS["SIDEWAYS"],
                   label="Sideways (Active)", alpha=0.6),
    mpatches.Patch(color=REGIME_COLORS["HIGH_VOLATILITY"],
                   label="High Vol (Sit Out)", alpha=0.6),
    mpatches.Patch(color=REGIME_COLORS["BEAR_TRENDING"],
                   label="Bear (Sit Out)", alpha=0.6),
]
ax2.legend(handles=patches, loc="upper left",
           fontsize=7, facecolor="#0f2040",
           edgecolor="#1a3357", labelcolor="#e8f0fe",
           ncol=2)

fig.suptitle(
    f"Aureline Labs — Regime-Conditional ML Strategy\n"
    f"{TICKER} · Test Period: {SPLIT_DATE} → {END}",
    color="#e8f0fe", fontsize=11,
    fontfamily="monospace", y=1.01
)
plt.tight_layout()
plt.savefig("data/regime_ml_analysis.png",
            dpi=150, bbox_inches="tight",
            facecolor="#060d1f")
plt.show()
logger.info("Chart saved to data/regime_ml_analysis.png")