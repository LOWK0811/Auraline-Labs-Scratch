# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import logging
import os
from src.data_handler import get_price_data
from src.features import build_features, feature_cols

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

# ======================================================================
# SECTION 3: BUILD AND INSPECT FEATURES
# ======================================================================
data = get_price_data("AAPL", "2021-01-01", "2026-06-01")
df = build_features(data)

print("\nFeature columns:")
print(df[feature_cols()].tail(5).to_string())

print(f"\nLabel distribution (0=down, 1=up):")
print(df["label"].value_counts())

print(f"\nComplete rows (no NaN): {df[feature_cols()].dropna().shape[0]}")
print(f"Total rows: {len(df)}")
