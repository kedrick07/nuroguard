# 🛡️ NuroGuard — Day 3 Progress Report
> **Varsity Hackathon 2026 | USM VHack | Case Study 2: Digital Trust — Real-Time Fraud Shield**
> Date: 10 March 2026 | Status: ✅ Complete

---

## Summary

Day 3 focused on Feature Engineering inside `notebooks/02_features.ipynb`. All 8 cells were completed successfully, producing 7 engineered features, one-hot encoded transaction types, a clean temporal train/test split, and two saved parquet files ready for model training on Day 4.

---

## ✅ Completed Tasks

### 1. Notebook: `02_features.ipynb` — All 8 Cells Complete

| Cell | Task | Status |
|------|------|--------|
| Cell 1 | Imports & load `paysim.csv` via pathlib | ✅ |
| Cell 2 | Load `eda_summary.json` feature hints | ✅ |
| Cell 3 | Engineer features 1–3 (type flag, zero-balance flag, log amount) | ✅ |
| Cell 4 | Engineer features 4–5 (hour-of-day, late-period flag) | ✅ |
| Cell 5 | Engineer features 6–7 (orig & dest balance diffs) | ✅ |
| Cell 6 | One-hot encode `type` column | ✅ |
| Cell 7 | Temporal train/test split | ✅ |
| Cell 8 | Validate feature set & save to parquet | ✅ |

---

## 📐 Engineered Features

### 7 New Features Created

| Feature | Type | Logic | Signal Found in Day 2 |
|---------|------|-------|----------------------|
| `is_transfer_or_cashout` | Binary flag | `type` ∈ {TRANSFER, CASH_OUT} → 1 else 0 | Only these 2 types contain fraud |
| `is_zero_newbalanceOrig` | Binary flag | `newbalanceOrig == 0` → 1 else 0 | 98.05% of fraud rows drain origin to 0 |
| `amount_log` | Continuous | `np.log1p(amount)` | Fraud amounts ~8x higher, right-skewed |
| `step_mod_24` | Cyclic (0–23) | `step % 24` | Extracts hour-of-day from simulation step |
| `is_late_period` | Binary flag | `step > 494` → 1 else 0 | Late period fraud rate ~10x higher |
| `orig_balance_diff` | Continuous | `(oldbalanceOrg − amount) − newbalanceOrig` | Non-zero in fraud rows due to balance manipulation |
| `dest_balance_diff` | Continuous | `(oldbalanceDest + amount) − newbalanceDest` | Deviates in fraud rows where destination isn't updated correctly |

### One-Hot Encoded Columns (5)

| Column | Represents |
|--------|-----------|
| `type_CASH_IN` | Transaction type = CASH_IN |
| `type_CASH_OUT` | Transaction type = CASH_OUT |
| `type_DEBIT` | Transaction type = DEBIT |
| `type_PAYMENT` | Transaction type = PAYMENT |
| `type_TRANSFER` | Transaction type = TRANSFER |

### Dropped Columns (3)

| Column | Reason |
|--------|--------|
| `nameOrig` | Raw account ID string — not usable as numeric feature, privacy risk |
| `nameDest` | Raw account ID string — not usable as numeric feature, privacy risk |
| `isFlaggedFraud` | Broken rule-based engine — caught only 16/8,213 fraud cases in Day 2 |

---

## 📊 Final Dataset

### Feature Set
- **Total columns**: 19
- **Final column list**: `step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `is_transfer_or_cashout`, `is_zero_newbalanceOrig`, `amount_log`, `step_mod_24`, `is_late_period`, `orig_balance_diff`, `dest_balance_diff`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`
- **Null values**: 0 across all columns in both train and test ✅

### Temporal Train/Test Split

| Split | Step Range | Shape | Fraud Rate | Period |
|-------|-----------|-------|------------|--------|
| Train | 1 – 494 | (6,051,528 × 19) | ~0.09% | Early + Mid |
| Test | 495 – 743 | (311,092 × 19) | ~0.87% | Late only |

- Split boundary at **step 494** — clean temporal separation, zero data leakage
- Test fraud rate is intentionally ~10x higher — evaluating on the hardest period

---

## 📁 New Files Created

```
nuroguard/
├── data/
│   └── processed/
│       ├── eda_summary.json              ← From Day 2 (unchanged)
│       ├── train_features.parquet        ← New (6,051,528 × 19)
│       └── test_features.parquet         ← New (311,092 × 19)
└── notebooks/
    └── 02_features.ipynb                 ← Completed (8 cells)
```

---

## ⚠️ Deviations & Issues

| Item | Issue | Fix Applied |
|------|-------|-------------|
| `to_parquet()` | `ArrowKeyError` — pandas/pyarrow version mismatch | Switched to `engine="fastparquet"` via `pip install fastparquet` |
| `drop(columns=DROP_COLS)` | `KeyError` — Cell 8 re-run after partial execution already dropped columns | Added `errors="ignore"` to both `.drop()` calls for safe re-runnability |

---

## 🎯 Day 3 Output vs Target

| Target | Status |
|--------|--------|
| Engineer 7 features from EDA hints | ✅ Done — all 7 created |
| One-hot encode `type` column | ✅ Done — 5 binary columns |
| Drop unusable columns | ✅ Done — nameOrig, nameDest, isFlaggedFraud removed |
| Temporal train/test split (no leakage) | ✅ Done — step ≤ 494 train / step > 494 test |
| Validate zero nulls | ✅ Done — 0 nulls in both splits |
| Save to `data/processed/` | ✅ Done — train & test parquet files saved |

---

## 🔜 Day 4 Plan

- Open `notebooks/03_model.ipynb`
- Load `train_features.parquet` and `test_features.parquet`
- Define feature matrix `X` and label vector `y` from `isFraud`
- Handle class imbalance using `scale_pos_weight` in LightGBM (ratio ~1:773)
- Train baseline **LightGBM** classifier
- Evaluate with precision, recall, F1, ROC-AUC on test set
- Save trained model to `models/lgbm_baseline.pkl`

---

*Report generated: 10 March 2026 | NuroGuard | USM VHack 2026*
