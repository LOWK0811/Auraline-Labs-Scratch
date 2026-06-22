# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data_handler import get_price_data
from src.features import build_all_features, all_feature_cols
from src.indicators import add_atr
from src.regime_detector import RegimeDetector
from src.risk import calculate_shares

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: REGIME-CONDITIONAL ML ENGINE
# ======================================================================
class RegimeConditionalML:
    """
    Trains and deploys separate ML models for each market regime.

    Philosophy:
    A single global model averages across very different market
    environments, diluting regime-specific signals. A model trained
    only on Bull Trending days learns what predicts returns when
    markets are calm and trending — a fundamentally different
    problem from predicting returns during High Volatility chaos.

    Regimes:
    - BULL_TRENDING:   Deploy a momentum-focused model
    - SIDEWAYS:        Deploy a mean-reversion focused model
    - HIGH_VOLATILITY: Sit out — models trained on other regimes
                       don't generalize to volatility spikes
    - BEAR_TRENDING:   Sit out — preserve capital
    """

    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or all_feature_cols()
        self.models       = {}
        self.regime_stats = {}
        self.detector     = RegimeDetector()


    # ======================================================================
    # SECTION 3: BUILD REGIME-LABELED DATASET
    # ======================================================================
    def prepare_data(self, ticker, start, end):
        """
        Loads data, builds features, detects regimes,
        and attaches labels for next-day direction.
        """
        raw  = get_price_data(ticker, start, end)
        raw  = add_atr(raw)
        df   = build_all_features(raw)
        df   = self.detector.detect(df)

        df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna(subset=self.feature_cols + ["label", "regime"])

        logger.info(f"Dataset prepared: {len(df)} rows")
        for regime in ["BULL_TRENDING","SIDEWAYS",
                       "HIGH_VOLATILITY","BEAR_TRENDING"]:
            count = (df["regime"] == regime).sum()
            logger.info(f"  {regime}: {count} days "
                       f"({count/len(df):.1%})")
        return df, raw


    # ======================================================================
    # SECTION 4: TRAIN REGIME-SPECIFIC MODELS
    # ======================================================================
    def train(self, df, split_date):
        """
        Trains one model per regime on the training period.
        Only trains on data from the matching regime.
        """
        train = df[df.index < split_date]
        logger.info(f"Training on {len(train)} rows "
                   f"(before {split_date})")

        for regime in ["BULL_TRENDING", "SIDEWAYS"]:
            regime_train = train[train["regime"] == regime]

            if len(regime_train) < 50:
                logger.warning(f"{regime}: only "
                               f"{len(regime_train)} training rows "
                               f"— skipping")
                continue

            X = regime_train[self.feature_cols]
            y = regime_train["label"]

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=max(10, len(regime_train)//20),
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            self.models[regime] = model

            train_auc = roc_auc_score(y, model.predict_proba(X)[:,1])
            self.regime_stats[regime] = {
                "train_rows":  len(regime_train),
                "train_auc":   round(train_auc, 4),
                "up_day_pct":  round(y.mean() * 100, 1)
            }
            logger.info(f"{regime} model trained: "
                       f"{len(regime_train)} rows, "
                       f"train AUC={train_auc:.4f}")

        logger.info(f"Trained {len(self.models)} regime models: "
                   f"{list(self.models.keys())}")


    # ======================================================================
    # SECTION 5: GENERATE REGIME-CONDITIONAL SIGNALS
    # ======================================================================
    def generate_signals(self, df, split_date,
                          prob_threshold=0.55):
        """
        For each day in the test period:
        - Detect the current regime
        - If regime has a trained model, use it to predict
        - If regime is HIGH_VOLATILITY or BEAR_TRENDING, signal=0
        """
        test = df[df.index >= split_date].copy()
        signals = []

        for i in range(len(test)):
            regime  = test["regime"].iloc[i]
            X_today = test[self.feature_cols].iloc[[i]]

            if regime in self.models:
                prob = self.models[regime].predict_proba(X_today)[0][1]
                signal = 1 if prob >= prob_threshold else 0
            else:
                # Sit out in High Volatility and Bear Trending
                signal = 0

            signals.append({
                "date":   test.index[i],
                "signal": signal,
                "regime": regime
            })

        return pd.DataFrame(signals).set_index("date")


    # ======================================================================
    # SECTION 6: BACKTEST REGIME-CONDITIONAL STRATEGY
    # ======================================================================
    def backtest(self, raw_data, signals_df,
                  cost_per_trade=0.001, starting_cash=10000):
        """
        Runs the backtest using regime-conditional signals.
        """
        raw_data = add_atr(raw_data)
        bt_data  = raw_data.loc[signals_df.index]

        cash        = starting_cash
        shares_held = 0
        portfolio   = []
        num_trades  = 0
        regime_days = {r: 0 for r in
                       ["BULL_TRENDING","SIDEWAYS",
                        "HIGH_VOLATILITY","BEAR_TRENDING"]}

        for i in range(len(bt_data)):
            price_today     = bt_data["Close"].iloc[i]
            price_yesterday = bt_data["Close"].iloc[i-1] \
                              if i > 0 else price_today
            atr_today       = bt_data["atr"].iloc[i]
            signal_today    = signals_df["signal"].iloc[i]
            regime_today    = signals_df["regime"].iloc[i]

            regime_days[regime_today] = \
                regime_days.get(regime_today, 0) + 1

            if signal_today == 1 and shares_held == 0:
                shares = calculate_shares(
                    cash, price_yesterday, atr_today)
                if shares > 0:
                    cash -= shares * price_yesterday * \
                            (1 + cost_per_trade)
                    shares_held = shares
                    num_trades += 1

            elif signal_today == 0 and shares_held > 0:
                cash += shares_held * price_yesterday * \
                        (1 - cost_per_trade)
                shares_held = 0
                num_trades += 1

            portfolio.append(cash + shares_held * price_today)

        return portfolio, num_trades, regime_days