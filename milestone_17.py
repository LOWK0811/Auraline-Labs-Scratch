# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from src.data_handler import get_price_data
from src.features import build_features, feature_cols
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
# SECTION 3: BUILD FEATURES
# ======================================================================
data = get_price_data("AAPL", "2021-01-01", "2026-06-01")
data = add_atr(data)
df   = build_features(data)
df   = df.dropna(subset=feature_cols() + ["label"])

logger.info(f"Total clean rows: {len(df)}")


# ======================================================================
# SECTION 4: WALK-FORWARD SIGNAL GENERATION
# ======================================================================
# We start generating signals after at least 500 training days.
# For each test day, we train on all prior days, predict the next day.

min_train_size = 500
signals = []

logger.info("Running walk-forward training...")

for i in range(min_train_size, len(df) - 1):
    train_slice = df.iloc[:i]
    X_train = train_slice[feature_cols()]
    y_train = train_slice["label"]

    model = RandomForestClassifier(
        n_estimators=50,     # fewer trees for speed — still robust
        max_depth=4,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_train, y_train)

    X_today = df[feature_cols()].iloc[[i]]
    prob_up  = model.predict_proba(X_today)[0][1]
    signals.append({
        "date":    df.index[i],
        "prob_up": prob_up,
        "signal":  1 if prob_up >= 0.55 else 0
    })

    if i % 100 == 0:
        logger.info(f"Walk-forward progress: {i}/{len(df)} rows processed")

signals_df = pd.DataFrame(signals).set_index("date")
logger.info(f"Signals generated: {len(signals_df)} days")
logger.info(f"Buy signals: {signals_df['signal'].sum()} | "
            f"Stay out: {(signals_df['signal']==0).sum()}")


# ======================================================================
# SECTION 5: BACKTEST ON SIGNAL PERIOD
# ======================================================================
cost_per_trade = 0.001
starting_cash  = 10000
cash           = starting_cash
shares_held    = 0
portfolio_history = []
num_buys  = 0
num_sells = 0

# Align backtest data to signal period
bt_data = df.loc[signals_df.index]

for i in range(len(bt_data)):
    date          = bt_data.index[i]
    price_today   = bt_data["Close"].iloc[i]
    price_yesterday = bt_data["Close"].iloc[i-1] if i > 0 else price_today
    atr_today     = bt_data["atr"].iloc[i]
    signal_today  = signals_df["signal"].iloc[i]

    if signal_today == 1 and shares_held == 0:
        shares = calculate_shares(cash, price_yesterday, atr_today, risk_pct=0.01)
        if shares > 0:
            cash -= shares * price_yesterday * (1 + cost_per_trade)
            shares_held = shares
            num_buys += 1
            logger.debug(f"{date.date()} | BUY {shares} @ ${price_yesterday:.2f}")

    elif signal_today == 0 and shares_held > 0:
        cash += shares_held * price_yesterday * (1 - cost_per_trade)
        shares_held = 0
        num_sells += 1
        logger.debug(f"{date.date()} | SELL @ ${price_yesterday:.2f}")

    portfolio_history.append(cash + shares_held * price_today)


# ======================================================================
# SECTION 6: RESULTS
# ======================================================================
final_value  = portfolio_history[-1]
total_return = (final_value / starting_cash) - 1

logger.info(f"Buys: {num_buys} | Sells: {num_sells}")
logger.info(f"Final value: ${final_value:,.2f} | Return: {total_return:.2%}")

# Buy and hold over the same period for fair comparison
bh_start_price = bt_data["Close"].iloc[0]
bh_shares      = starting_cash // bh_start_price
bh_values      = bh_shares * bt_data["Close"]

plt.plot(bt_data.index, portfolio_history, label="ML Strategy")
plt.plot(bt_data.index, bh_values,         label="Buy & Hold")
plt.title("ML Model vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.show()