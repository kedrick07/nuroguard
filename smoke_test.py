import pickle
import pandas as pd
import lightgbm as lgb
from src.services.explanation_service import ExplanationService
from src.services.decision_engine import DecisionEngine

# --- Load and unpack the model dict ---
with open("models/lgbm_baseline_ieee.pkl", "rb") as f:
    saved = pickle.load(f)

booster       = saved["model"]        # lgb.Booster
feature_cols  = saved["feature_cols"] # exact training column order
metrics       = saved["metrics"]      # Day 5 metrics — print for reference

print(f"Model loaded ✅  |  Features: {len(feature_cols)}  |  Day 5 ROC-AUC: {metrics.get('roc_auc', 'N/A')}")

# --- Load validation set ---
val     = pd.read_parquet("data/processed/features_val.parquet")
X_val   = val[feature_cols]           # keep only training features, in exact order
y_val   = val["isFraud"]

# --- Score every row ---
scores = booster.predict(X_val)
print(f"Scores computed ✅  |  Val rows: {len(scores)}  |  Score range: {scores.min():.4f} – {scores.max():.4f}")

# --- Pick one sample per tier using locked Day 6 thresholds ---
APPROVE_THRESH = 0.1888
BLOCK_THRESH   = 0.6973

approve_idx = next(i for i, s in enumerate(scores) if s < APPROVE_THRESH)
flag_idx    = next(i for i, s in enumerate(scores) if APPROVE_THRESH <= s < BLOCK_THRESH)
block_idx   = next(i for i, s in enumerate(scores) if s >= BLOCK_THRESH)

print(f"\nSample indices  →  APPROVE: {approve_idx}  |  FLAG: {flag_idx}  |  BLOCK: {block_idx}")

# --- Initialise ExplanationService ---
svc = ExplanationService("models/lgbm_baseline_ieee.pkl", top_n=3)

# --- Run explanations ---
for label, idx in [("APPROVE", approve_idx), ("FLAG", flag_idx), ("BLOCK", block_idx)]:
    row    = X_val.iloc[idx]
    score  = scores[idx]
    actual = int(y_val.iloc[idx])
    reasons = svc.explain(row)

    # Annotate whether this is a true positive/negative or false positive/negative
    if label == "APPROVE" and actual == 0:   verdict = "TRUE NEGATIVE  ✅ (legit, correctly approved)"
    elif label == "APPROVE" and actual == 1: verdict = "FALSE NEGATIVE ⚠️  (fraud slipped through)"
    elif label == "FLAG"    and actual == 1: verdict = "TRUE POSITIVE  ✅ (fraud, correctly flagged)"
    elif label == "FLAG"    and actual == 0: verdict = "FALSE POSITIVE ⚠️  (legit, incorrectly flagged)"
    elif label == "BLOCK"   and actual == 1: verdict = "TRUE POSITIVE  ✅ (fraud, correctly blocked)"
    else:                                    verdict = "FALSE POSITIVE ⚠️  (legit, blocked — within FPR budget)"

    print(f"\n{'='*55}")
    print(f"[{label}]  score={score:.4f}  |  {verdict}")
    for i, r in enumerate(reasons, 1):
        print(f"  Reason {i}: {r}")