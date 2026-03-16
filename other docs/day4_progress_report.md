# NuroGuard Day 4 Progress Report
**Varsity Hackathon 2026 — USM VHack | Case Study 2: Digital Trust — Real-Time Fraud Shield**
**Date:** 11 March 2026
**Status:** ✅ Complete

---

## Summary

Day 4 focused on training the baseline LightGBM model in `notebooks/03model.ipynb`. All 7 cells were completed successfully, producing a trained model with exceptional evaluation metrics on the PaySim test set. The model was saved to `models/lgbm_baseline.pkl` and is ready for use in the pipeline from Day 6 onwards.

> ⚠️ **IMPORTANT — Dataset Switch Pending (Read Before Day 5)**
> At the end of Day 4, Kaggle access to the **IEEE-CIS Fraud Detection dataset** was confirmed. A decision was made to **switch from PaySim to IEEE-CIS** before proceeding to Day 5. See the full context in the [Dataset Switch Decision](#dataset-switch-decision) section below. Day 5 must begin with re-running Days 2 and 3 on IEEE-CIS before any new model work.

---

## Completed Tasks — `notebooks/03model.ipynb` All 7 Cells Complete

| Cell | Task | Status |
|------|------|--------|
| Cell 1 | Imports, path setup (`pathlib`, `lightgbm`, `sklearn.metrics`, `tqdm`, `time`) | ✅ Done |
| Cell 2 | Load `train_features.parquet` and `test_features.parquet` | ✅ Done |
| Cell 3 | Define feature matrix `X_train`, `X_test` and label vectors `y_train`, `y_test` | ✅ Done |
| Cell 4 | Compute `scale_pos_weight` from live class counts | ✅ Done |
| Cell 5 | Train LightGBM with `tqdm` progress bar, early stopping, training time report | ✅ Done |
| Cell 6 | Evaluate — Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, confusion matrix | ✅ Done |
| Cell 7 | Save model to `models/lgbm_baseline.pkl`, reload sanity check | ✅ Done |

---

## Training Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| `objective` | `binary` | Binary fraud vs not-fraud classification |
| `metric` | `auc` | `average_precision` unsupported in installed LightGBM version — switched to `auc`, PR-AUC computed via sklearn instead |
| `scale_pos_weight` | `1100.68` | Derived live from class counts (6,046,035 negatives / 5,493 positives) |
| `learning_rate` | `0.05` | Conservative step size, allows precise convergence |
| `num_leaves` | `63` | Moderate tree complexity, balances expressiveness vs overfitting |
| `min_child_samples` | `50` | Prevents memorising rare fraud transactions |
| `feature_fraction` | `0.8` | Samples 80% of features per tree for robustness |
| `bagging_fraction` | `0.8` | Samples 80% of rows per tree for robustness |
| `early_stopping` | `50 rounds` | Stops if AUC does not improve for 50 consecutive rounds |
| `num_boost_round` | `1000` | Maximum cap — early stopping fired at round 17 |

---

## Class Imbalance — Actual Counts

| Class | Count | Rate |
|-------|-------|------|
| Non-fraud (0) | 6,046,035 | 99.91% |
| Fraud (1) | 5,493 | 0.09% |
| **scale_pos_weight** | **1100.68** | Ratio used in LightGBM |

> Note: Day 3 plan estimated ratio ~1773. Actual ratio was 1100.68 — always derive live from data, never hardcode.

---

## Training Result

| Metric | Value |
|--------|-------|
| Best iteration | 17 / 1000 |
| Training time | 0m 6.3s |

> Early stopping at round 17 is expected behaviour on PaySim. The engineered features from Day 3 (`istransferorcashout`, `iszeronewbalanceOrig`) are extremely strong, clean synthetic signals — the model converges in very few rounds. This will NOT be the case on IEEE-CIS, which has noisier real-world data and will likely train for many more rounds.

---

## Evaluation Results — PaySim Test Set

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Precision | 0.9676 | — | ✅ |
| Recall | 0.9996 | ≥ 0.85 | ✅ Crushed |
| F1 Score | 0.9834 | ≥ 0.70 | ✅ Crushed |
| ROC-AUC | 0.9995 | — | ✅ |
| **PR-AUC** | **0.9703** | **≥ 0.75** | ✅ **Crushed** |
| False Positive Rate | 0.0003 | < 0.02 | ✅ 67× better than target |

### Confusion Matrix

| | Predicted Not Fraud | Predicted Fraud |
|---|---|---|
| **Actual Not Fraud** | TN = 308,281 | FP = 91 |
| **Actual Fraud** | FN = 1 | TP = 2,719 |

> Only 1 fraud missed. 91 false alarms out of 308,372 legitimate transactions.

### Classification Report

```
              precision    recall  f1-score   support

   Not Fraud       1.00      1.00      1.00    308,372
       Fraud       0.97      1.00      0.98      2,720

    accuracy                           1.00    311,092
   macro avg       0.98      1.00      0.99    311,092
weighted avg       1.00      1.00      1.00    311,092
```

---

## Why These Metrics Are NOT Submission-Ready

These results are outstanding but **expected** on PaySim because:
- PaySim is **synthetic** — fraud patterns were deliberately programmed to be clean and detectable
- Features like `istransferorcashout` and `iszeronewbalanceOrig` directly encode how the simulation defined fraud
- Judges familiar with PaySim will not be impressed by 0.97 PR-AUC — they have seen it many times
- A 0.80 PR-AUC on **IEEE-CIS real data** is worth far more to judges than 0.97 on PaySim

---

## Deviations & Issues

| Item | Issue | Fix Applied |
|------|-------|-------------|
| `metric: average_precision` | Not supported in installed LightGBM version — early stopping fired at round 1 | Changed to `metric: auc`; PR-AUC computed via `sklearn.average_precision_score` in Cell 6 |
| RAM tracking via `psutil` | Negative RAM delta due to GC freeing previous run's memory between re-runs | Removed RAM tracking entirely — not essential for hackathon |
| `scale_pos_weight` estimate | Day 3 plan estimated 1773, actual is 1100.68 | Derived live from `y_train` counts — always correct |

---

## Files Created / Modified

| File | Status |
|------|--------|
| `notebooks/03model.ipynb` | New — 7 cells complete |
| `models/lgbm_baseline.pkl` | New — trained LightGBM model, reload verified |
| `data/processed/train_features.parquet` | Unchanged from Day 3 |
| `data/processed/test_features.parquet` | Unchanged from Day 3 |

---

## Dataset Switch Decision

> ⚠️ **This section is critical context for Day 5 planning. Read fully before starting.**

### What Happened
Kaggle access to the **IEEE-CIS Fraud Detection dataset** was confirmed at the end of Day 4. The decision was made to switch from PaySim to IEEE-CIS before continuing with Day 5 (Decision Engine & Privacy).

### Why Switch

| Reason | Detail |
|--------|--------|
| IEEE-CIS is the **primary recommended dataset** in the project plan | PaySim was always intended as optional/fallback |
| IEEE-CIS is **real transaction data** | 590K rows, not synthetic — judges know the difference |
| Real data = feature engineering actually matters | 434 raw features require real decisions, not just encoding a simulation |
| Better alignment with Case Study 2 | Real card fraud context vs simulated mobile money |

### What Needs to Be Redone

| Task | Notebook | Estimated Time |
|------|----------|----------------|
| EDA on IEEE-CIS | Redo `01eda.ipynb` | ~2–3 hrs |
| Feature engineering on IEEE-CIS | Redo `02features.ipynb` — merge `train_transaction.csv` + `train_identity.csv`, handle heavy nulls, engineer features | ~3–4 hrs |
| Retrain model | Re-run `03model.ipynb` Cell 2 onwards | ~30 mins |
| **Total estimated setback** | | **~1 day** |

### What Carries Over (No Changes Needed)

| Asset | Reusable? |
|-------|-----------|
| `03model.ipynb` Cells 1, 3–7 | ✅ Fully reusable — dataset-agnostic |
| LightGBM training code and params | ✅ Same params, just re-derive `scale_pos_weight` |
| `scale_pos_weight` computation logic | ✅ Same formula, new ratio (~28 for IEEE-CIS vs 1100 for PaySim) |
| Evaluation metrics code | ✅ Identical |
| `models/lgbm_baseline.pkl` | ❌ Will be overwritten after IEEE-CIS retraining |

### Key Differences — IEEE-CIS vs PaySim

| Dimension | PaySim (Done) | IEEE-CIS (Next) |
|-----------|--------------|-----------------|
| Data type | Synthetic | Real transactions |
| Size | 6.3M rows | 590K rows |
| Raw features | 10 | 434 |
| Null values | 0 | Heavy — some cols 90%+ null |
| Tables | 1 CSV | 2 CSVs — must merge on `TransactionID` |
| Fraud rate | ~0.09% train | ~3.5% |
| Expected `scale_pos_weight` | 1100 | ~28 |
| Expected PR-AUC | 0.97 (trivial) | 0.75–0.88 (meaningful) |

### Day 5 Starting Instructions

**Do NOT jump straight to Decision Engine.** Day 5 must begin with:

1. Place `train_transaction.csv`, `train_identity.csv` into `data/raw/`
2. Redo `notebooks/01eda.ipynb` — explore fraud rate, null rates per column, feature distributions, merge strategy
3. Redo `notebooks/02features.ipynb` — merge tables with `pd.merge(..., how="left", on="TransactionID")`, handle nulls with median/mode imputation, engineer features from the 434 columns
4. Re-run `notebooks/03model.ipynb` Cell 2 onwards with the new parquet files
5. Only after retraining on IEEE-CIS — proceed to Day 5 proper (Decision Engine)

### IEEE-CIS Null Handling Heads-Up
Many columns in `train_identity.csv` are 90%+ null — this is normal and expected. Strategy:
- Numerical nulls → fill with **median**
- Categorical nulls → fill with **`"unknown"`** then label encode or one-hot
- Columns with >95% nulls → consider dropping entirely
- Never drop rows due to nulls — you'll lose too much data

---

## Day 4 Output vs Target

| Target | Status |
|--------|--------|
| Train baseline LightGBM classifier | ✅ Done |
| Handle class imbalance with `scale_pos_weight` | ✅ Done — ratio 1100.68 |
| Evaluate PR-AUC, F1, Recall, FPR on test set | ✅ Done — all targets exceeded |
| Save trained model to `models/lgbm_baseline.pkl` | ✅ Done — reload verified |

---

*Report generated 11 March 2026 — NuroGuard | USM VHack 2026*
