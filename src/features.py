# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: FEATURE ENGINEERING
# ======================================================================
def build_features(data):
    """
    Takes raw OHLCV data and returns a DataFrame of engineered features.
    Every feature is derived only from information available at close today
    — no look-ahead bias. The label (next-day direction) uses shift(-1),
    which is only used for training, never for live signals.
    """
    df = data.copy()

    # --- Returns over different lookback windows ---
    # These capture momentum: is the stock trending up or down?
    df["return_1d"]  = df["Close"].pct_change(1)
    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["return_20d"] = df["Close"].pct_change(20)

    # --- Z-score of 1-day return ---
    # How unusual is today's move relative to recent history?
    rolling_mean = df["return_1d"].rolling(20).mean()
    rolling_std  = df["return_1d"].rolling(20).std()
    df["zscore_1d"] = (df["return_1d"] - rolling_mean) / rolling_std

    # --- Distance from moving averages ---
    # Is the price stretched above or below its trend?
    df["dist_sma20"]  = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"]
    df["dist_sma50"]  = (df["Close"] - df["Close"].rolling(50).mean()) / df["Close"]

    # --- Volatility ---
    # How much has the stock been moving lately?
    df["volatility_10d"] = df["return_1d"].rolling(10).std()

    # --- Volume signal ---
    # Is today's volume unusually high or low vs recent average?
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # --- Label: did price go UP the next day? ---
    # 1 = yes (buy signal), 0 = no (stay out)
    # shift(-1) looks forward — ONLY valid for training, not live use
    df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    logger.info(f"Features built: {len(df)} rows, "
                f"{df[feature_cols()].notna().all(axis=1).sum()} complete rows")
    return df


# ======================================================================
# SECTION 3: FEATURE COLUMN NAMES (SINGLE SOURCE OF TRUTH)
# ======================================================================
def feature_cols():
    """Returns the list of feature column names used by the model."""
    return [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "zscore_1d", "dist_sma20", "dist_sma50",
        "volatility_10d", "volume_ratio"
    ]