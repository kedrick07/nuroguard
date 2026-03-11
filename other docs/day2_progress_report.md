# 🛡️ NuroGuard — Day 2 Progress Report
> **Varsity Hackathon 2026 | USM VHack | Case Study 2: Digital Trust — Real-Time Fraud Shield**
> Date: 10 March 2026 | Status: ✅ Complete

---

## Summary

Day 2 focused on Exploratory Data Analysis (EDA) on the PaySim dataset inside `notebooks/01_eda.ipynb`. All 8 parts were completed successfully, producing a clean dataset assessment, key fraud signal discoveries, and a saved EDA summary handoff file at `data/processed/eda_summary.json`. The project is ready to begin Feature Engineering on Day 3.

---

## ✅ Completed Tasks

### 1. Dataset Swap — BAF → PaySim
- Original Day 2 plan specified loading `data/raw/baf/Base.csv` (BAF NeurIPS 2022 dataset)
- Dataset was **changed to PaySim** (`data/raw/paysim.csv`) — a synthetic mobile money transaction dataset from Kaggle
- PaySim has 11 columns vs BAF's 196 — simpler schema, cleaner fraud labels
- File saved at `data/raw/paysim.csv` as a single CSV

### 2. Notebook: `01_eda.ipynb` — All 8 Parts Complete

| Cell | Task | Status |
|------|------|--------|
| Cell 1 | Import libraries & load dataset | ✅ |
| Cell 2 | Basic inspection — shape, dtypes, memory, describe | ✅ |
| Cell 3 | Missing values & duplicate rows check | ✅ |
| Cell 4 | Fraud rate & class imbalance analysis | ✅ |
| Cell 5 | Transaction type distribution | ✅ |
| Cell 6 | Amount & balance distributions | ✅ |
| Cell 7 | Time (step) analysis | ✅ |
| Cell 8 | Save EDA summary to `data/processed/eda_summary.json` | ✅ |

### 3. Key EDA Findings

#### Dataset Health
- **Shape**: 6,362,620 rows × 11 columns
- **Missing values**: 0 across all columns ✅
- **Duplicate rows**: 0 ✅
- **Label sanity**: `isFraud` and `isFlaggedFraud` confirmed binary `[0, 1]` only ✅

#### Class Imbalance
| Class | Count | Percentage |
|-------|-------|------------|
| Legitimate | 6,354,407 | ~99.87% |
| Fraud | 8,213 | ~0.13% |

- Imbalance ratio: **1 fraud per ~773 legitimate transactions**
- Severe imbalance confirmed → SMOTE or class weighting required during model training

#### isFlaggedFraud vs isFraud Gap
- Business rule engine (`isFlaggedFraud`) only caught **16 out of 8,213** fraud cases
- **8,197 real fraud transactions were missed** by the existing rule-based system
- This validates the need for an ML model — the business rules are nearly useless

#### Transaction Type Findings
| Type | Count | Fraud Count | Fraud Rate |
|------|-------|-------------|------------|
| CASH_OUT | 2,237,500 | 4,116 | ~0.18% |
| PAYMENT | 2,151,495 | 0 | 0.00% |
| CASH_IN | 1,399,284 | 0 | 0.00% |
| TRANSFER | 532,909 | 4,097 | ~0.77% |
| DEBIT | 41,432 | 0 | 0.00% |

- Only **TRANSFER** and **CASH_OUT** contain fraud
- CASH_IN, PAYMENT, and DEBIT have **zero fraud** — strong binary filter feature

#### Amount Insights
| Class | Mean Amount |
|-------|-------------|
| Legit | ~178,000 |
| Fraud | ~1,470,000 |

- Fraud mean amount is **~8x higher** than legitimate transactions
- Fraud amount is heavily right-skewed — log-transform will be needed

#### Balance Insights
- **98.05% of fraud transactions** drain the origin account to `newbalanceOrig = 0`
- Only 56.68% of legitimate transactions result in zero balance — a sharp contrast
- Balance difference columns (`orig_balance_diff`, `dest_balance_diff`) show large deviations in fraud rows
- Raw balance columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) are **NOT safe to use directly** as features — they are manipulated in fraud rows, consistent with the dataset's own warning

#### Temporal Insights
| Period | Total Txns | Fraud Count | Fraud Rate |
|--------|-----------|-------------|------------|
| Early (steps 1–247) | 3,193,219 | 2,781 | 0.087% |
| Mid (steps 248–494) | 2,858,309 | 2,712 | 0.095% |
| Late (steps 495–743) | 311,092 | 2,720 | 0.874% |

- Late period fraud rate is **~10x higher** than Early — strong temporal signal
- Steps 734–743 (end of simulation) show 100% fraud rate
- Step 523 burst: 30 out of 30 transactions were fraud — coordinated attack pattern
- Simulation spans steps 1–743 = **31.0 days**

### 4. EDA Summary File Saved
- Saved to: `data/processed/eda_summary.json`
- Contains: dataset shape, class balance, fraud type breakdown, amount insights, balance flags, temporal insights, and 7 feature engineering hints
- Acts as the **handoff document** from Day 2 EDA → Day 3 Feature Engineering

---

## 📁 New Files Created

```
nuroguard/
├── data/
│   ├── raw/
│   │   └── paysim.csv              ← New dataset (replaces BAF)
│   └── processed/
│       └── eda_summary.json        ← EDA findings handoff file (new)
└── notebooks/
    └── 01_eda.ipynb                ← Completed (8 cells)
```

---

## ⚠️ Deviations from Plan

| Item | Planned | Actual | Reason |
|------|---------|--------|--------|
| Dataset | BAF NeurIPS 2022 (`Base.csv`) | PaySim (`paysim.csv`) | Dataset changed by team decision — PaySim is simpler, single-file, and more suitable for the hackathon scope |
| Notebook comment style | `# (number)` inline references | Removed inline number comments | Cleaner code readability; explanations given in separate documentation |
| `time_period` column | Not in original plan | Added during EDA | Emerged naturally from temporal analysis; will be used as a feature in Day 3 |
| `orig_balance_diff` / `dest_balance_diff` | Not in original plan | Computed during EDA | Needed to validate balance manipulation in fraud rows; will become engineered features |

---

## 🎯 Day 2 Output vs Target

| Target | Status |
|--------|--------|
| Load dataset and confirm shape | ✅ Done — 6,362,620 rows × 11 cols |
| Check data quality (nulls, dupes) | ✅ Done — dataset fully clean |
| Identify fraud rate & imbalance | ✅ Done — 0.13%, ratio 1:773 |
| Understand transaction types | ✅ Done — only TRANSFER & CASH_OUT have fraud |
| Explore amount distributions | ✅ Done — fraud 8x higher mean amount |
| Temporal analysis | ✅ Done — late period 10x fraud rate |
| Document feature engineering hints | ✅ Done — 7 hints saved to JSON |

---

## 🔜 Day 3 Plan

- Open `notebooks/02_features.ipynb`
- Load `data/raw/paysim.csv`
- Engineer the 7 features identified in `eda_summary.json`:
  1. `is_transfer_or_cashout` — binary flag for fraud-eligible types
  2. `is_zero_newbalanceOrig` — binary flag for drained origin accounts
  3. `amount_log` — log-transformed transaction amount
  4. `step_mod_24` — hour-of-day extracted from step
  5. `is_late_period` — binary flag for steps > 494
  6. `orig_balance_diff` — deviation from expected origin balance equation
  7. `dest_balance_diff` — deviation from expected destination balance equation
- Encode categorical column `type` using one-hot encoding
- Split dataset temporally (train: Early+Mid / test: Late) to avoid data leakage
- Save processed dataset to `data/processed/features.parquet`

---

*Report generated: 10 March 2026 | NuroGuard | USM VHack 2026*
