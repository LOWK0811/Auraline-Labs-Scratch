# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: REGIME LABELS
# ======================================================================
REGIMES = {
    0: "BULL_TRENDING",
    1: "BEAR_TRENDING",
    2: "HIGH_VOLATILITY",
    3: "SIDEWAYS"
}

REGIME_COLORS = {
    "BULL_TRENDING":   "#00d4aa",
    "BEAR_TRENDING":   "#ff4d6a",
    "HIGH_VOLATILITY": "#ffd166",
    "SIDEWAYS":        "#7b9bc0"
}


# ======================================================================
# SECTION 3: REGIME DETECTOR
# ======================================================================
class RegimeDetector:
    """
    Classifies the market into one of four regimes on each day
    using a rule-based system derived from price, volatility,
    and trend signals.

    Rule-based rather than ML-based by design: regime labels
    need to be interpretable and stable. A model that silently
    changes what it means by "bull market" is dangerous in
    production.
    """

    def __init__(self,
                 sma_window=200,
                 vol_window=20,
                 trend_window=50,
                 vol_lookback=252):
        self.sma_window   = sma_window
        self.vol_window   = vol_window
        self.trend_window = trend_window
        self.vol_lookback = vol_lookback


    # ======================================================================
    # SECTION 4: COMPUTE REGIME SIGNALS
    # ======================================================================
    def _compute_signals(self, data):
        """Computes the raw signals used for regime classification."""
        df = data.copy()

        # ── Trend direction: price vs SMA(200) ──
        df["_sma200"] = df["Close"].rolling(self.sma_window).mean()
        df["_above_sma200"] = df["Close"] > df["_sma200"]

        # ── Realized volatility (annualized) ──
        daily_return = df["Close"].pct_change()
        df["_realized_vol"] = (
            daily_return.rolling(self.vol_window).std() * np.sqrt(252)
        )

        # ── Volatility regime: above or below historical median ──
        df["_vol_median"] = (
            df["_realized_vol"].rolling(self.vol_lookback).median()
        )
        df["_high_vol"] = df["_realized_vol"] > df["_vol_median"]

        # ── Trend strength: consistency of directional moves ──
        up_moves   = (df["High"] - df["High"].shift(1)).clip(lower=0)
        down_moves = (df["Low"].shift(1) - df["Low"]).clip(lower=0)
        diff       = (up_moves - down_moves).abs()
        total      = up_moves + down_moves + 1e-10
        df["_trend_strength"] = (diff / total).rolling(14).mean()

        # ── Price momentum: is the trend accelerating? ──
        df["_momentum"] = df["Close"].pct_change(self.trend_window)

        return df


    # ======================================================================
    # SECTION 5: CLASSIFY REGIME PER DAY
    # ======================================================================
    def detect(self, data):
        """
        Classifies each trading day into one of four regimes.
        Returns the DataFrame with a 'regime' column added.

        Classification logic (in priority order):
        1. HIGH_VOLATILITY if realized vol > historical median
           AND vol is in the top quartile of the past year
        2. BEAR_TRENDING if price below SMA200 AND momentum negative
        3. BULL_TRENDING if price above SMA200 AND trend strong
           AND momentum positive
        4. SIDEWAYS otherwise
        """
        df = self._compute_signals(data)
        regimes = []

        # Precompute vol percentile for stricter high-vol detection
        vol_75pct = df["_realized_vol"].rolling(252).quantile(0.75)

        for i in range(len(df)):
            above_sma  = df["_above_sma200"].iloc[i]
            high_vol   = df["_high_vol"].iloc[i]
            vol_severe = df["_realized_vol"].iloc[i] > \
                         (vol_75pct.iloc[i]
                          if not pd.isna(vol_75pct.iloc[i])
                          else float("inf"))
            momentum   = df["_momentum"].iloc[i]
            trend_str  = df["_trend_strength"].iloc[i]

            # Handle NaN values during warmup period
            if pd.isna(above_sma) or pd.isna(high_vol) \
               or pd.isna(momentum) or pd.isna(trend_str):
                regimes.append("SIDEWAYS")
                continue

            # Priority 1: High volatility override
            if high_vol and vol_severe:
                regimes.append("HIGH_VOLATILITY")

            # Priority 2: Bear trending
            elif not above_sma and momentum < -0.02:
                regimes.append("BEAR_TRENDING")

            # Priority 3: Bull trending
            elif above_sma and momentum > 0.02 and trend_str > 0.5:
                regimes.append("BULL_TRENDING")

            # Priority 4: Sideways/Choppy
            else:
                regimes.append("SIDEWAYS")

        df["regime"] = regimes
        logger.info(f"Regime detection complete: "
                   f"{len(df)} days classified")
        return df


    # ======================================================================
    # SECTION 6: REGIME STATISTICS
    # ======================================================================
    def summarize(self, data_with_regimes):
        """
        Prints a summary of how much time was spent in each regime
        and what returns looked like in each one.
        """
        df = data_with_regimes.copy()
        df["daily_return"] = df["Close"].pct_change()

        print(f"\n{'='*65}")
        print(f"  AURELINE LABS — MARKET REGIME ANALYSIS")
        print(f"{'='*65}")
        print(f"  {'Regime':<20} {'Days':>6} {'%Time':>7} "
              f"{'Avg Return':>12} {'Volatility':>12}")
        print(f"  {'-'*60}")

        total_days = len(df.dropna(subset=["regime"]))

        for regime_name in REGIMES.values():
            mask  = df["regime"] == regime_name
            subset = df[mask]["daily_return"].dropna()

            if len(subset) == 0:
                continue

            days    = len(subset)
            pct     = days / total_days * 100
            avg_ret = subset.mean() * 100
            vol     = subset.std() * np.sqrt(252) * 100

            print(f"  {regime_name:<20} {days:>6} {pct:>6.1f}% "
                  f"{avg_ret:>+11.3f}% {vol:>11.1f}%")

        print(f"{'='*65}")
        return df.groupby("regime")["daily_return"].agg(
            ["mean", "std", "count"]
        )