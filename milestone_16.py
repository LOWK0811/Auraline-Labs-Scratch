# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay)
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
logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 3: BUILD FEATURES AND DROP INCOMPLETE ROWS
# ======================================================================
data = get_price_data("AAPL", "2021-01-01", "2026-06-01")
df = build_features(data)
df = df.dropna(subset=feature_cols() + ["label"])

logger.info(f"Clean rows after dropna: {len(df)}")


# ======================================================================
# SECTION 4: TIME-BASED TRAIN/TEST SPLIT
# ======================================================================
split_date = "2024-06-01"
train = df[df.index < split_date]
test  = df[df.index >= split_date]

X_train = train[feature_cols()]
y_train = train["label"]
X_test  = test[feature_cols()]
y_test  = test["label"]

logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows")
logger.info(f"Train label split: {y_train.mean():.1%} up-days")
logger.info(f"Test label split:  {y_test.mean():.1%} up-days")


# ======================================================================
# SECTION 5: TRAIN THE MODEL
# ======================================================================
model = RandomForestClassifier(
    n_estimators=100,    # number of trees in the forest
    max_depth=4,         # how deep each tree can grow — keeps it from memorizing
    min_samples_leaf=20, # each leaf must have at least 20 samples — prevents overfit
    random_state=42      # makes results reproducible
)

model.fit(X_train, y_train)
logger.info("Model trained successfully")


# ======================================================================
# SECTION 6: EVALUATE ON THE HELD-OUT TEST SET
# ======================================================================
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability of "up"

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {auc:.4f}")


# ======================================================================
# SECTION 7: CONFUSION MATRIX
# ======================================================================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
disp.plot()
plt.title("Confusion Matrix — Test Set")
plt.show()


# ======================================================================
# SECTION 8: FEATURE IMPORTANCES
# ======================================================================
importances = pd.Series(model.feature_importances_, index=feature_cols())
importances = importances.sort_values(ascending=True)

plt.figure()
importances.plot(kind="barh")
plt.title("Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()