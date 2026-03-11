# 🛡️ NuroGuard — Day 1 Progress Report
> **Varsity Hackathon 2026 | USM VHack | Case Study 2: Digital Trust — Real-Time Fraud Shield**
> Date: 8 March 2026 | Status: ✅ Complete

---

## Summary

Day 1 focused on project setup and planning. All structural, environment, and data prerequisites are in place. The project is ready to begin EDA on Day 2.

---

## ✅ Completed Tasks

### 1. Repository & Folder Structure
- Initialized Git repository and linked to GitHub remote (`origin/main`)
- Created full project directory following the implementation plan layout
- Confirmed 65 directories and 134 files in final tree

### 2. File Naming & Conventions
- Renamed all source files from compact camelCase to consistent `snake_case`
  - e.g. `featureengineer.py` → `feature_engineer.py`
  - e.g. `decisionengine.py` → `decision_engine.py`
  - e.g. `scoringresult.py` → `scoring_result.py`
- Moved app entry point from `src/main.py` → `src/api/main.py` to match the API-layer architecture

### 3. `.gitignore` Setup
- Verified `.gitignore` correctly covers:
  - `.env` (secrets)
  - `__pycache__/` and `*.py[cod]` (Python cache)
  - `data/raw/` and `data/processed/` (datasets)
  - `models/*.pkl`, `models/*.pt` (trained artifacts)
  - `.venv/`, `venv/` (virtual environments)
- Confirmed `git status --ignored` shows all expected files properly ignored
- Working tree is clean: `nothing to commit`

### 4. Python Virtual Environment
- Created isolated `.venv` using Python 3.12 (avoiding system Python 3.14.2 for compatibility)
- Installed all project dependencies from the implementation plan:
  - `pandas`, `numpy`, `lightgbm`, `imbalanced-learn`
  - `torch`, `pyod`, `shap`
  - `fastapi`, `uvicorn`, `pydantic`
  - `pytest`, `scipy`, `geopy`, `jupyter`
- Pinned exact versions to `requirements.txt` via `pip freeze`

### 5. Class Skeleton Files
- All source module files created under `src/` with placeholder structure:

| Layer | Files |
|---|---|
| `src/entities/` | `transaction.py`, `user_profile.py`, `scoring_result.py` |
| `src/features/` | `feature_engineer.py`, `geo_velocity.py`, `validators.py` |
| `src/models/` | `base_model.py`, `lightgbm_model.py`, `lstm_model.py`, `fusion_model.py` |
| `src/services/` | `risk_scoring_service.py`, `decision_engine.py`, `explanation_service.py`, `privacy_guard.py`, `data_processor.py` |
| `src/api/` | `main.py`, `routes.py`, `schemas.py` |
| `src/utils/` | `config.py`, `logger.py`, `metrics.py`, `serializer.py` |
| `tests/` | `test_api.py`, `test_decision_engine.py`, `test_feature_engineer.py`, `test_privacy_guard.py` |

### 6. Dataset
- IEEE-CIS Fraud Detection (primary) was unavailable due to Kaggle competition registration rate-limit
- Switched to **Bank Account Fraud (BAF) Dataset Suite — NeurIPS 2022** as the primary dataset
- Downloaded and placed under `data/raw/baf/`:

```
data/raw/baf/
├── Base.csv          ← primary training file
├── Variant I.csv
├── Variant II.csv
├── Variant III.csv
├── Variant IV.csv
└── Variant V.csv
```

- BAF is a 1.36 GB, 6-file synthetic fraud dataset with 196 columns, temporal `month` column, and strong research backing (NeurIPS 2022)

---

## 📁 Final Project Structure

```
nuroguard/
├── data/
│   ├── raw/baf/         ← BAF dataset (gitignored)
│   └── processed/
├── notebooks/           ← 01_eda to 05_evaluation
├── scripts/             ← train.py, evaluate.py, seed_demo_data.py
├── src/
│   ├── api/             ← main.py, routes.py, schemas.py
│   ├── entities/        ← transaction, user_profile, scoring_result
│   ├── features/        ← feature_engineer, geo_velocity, validators
│   ├── models/          ← base, lightgbm, lstm, fusion
│   ├── services/        ← risk_scoring, decision, explanation, privacy, data
│   └── utils/           ← config, logger, metrics, serializer
├── tests/
├── models/              ← trained artifacts (gitignored)
├── .env                 ← gitignored
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚠️ Deviations from Plan

| Item | Planned | Actual | Reason |
|---|---|---|---|
| Primary Dataset | IEEE-CIS Fraud Detection | BAF NeurIPS 2022 | Kaggle rate-limit blocked competition registration |
| Python Version | Not specified | Python 3.12 (not 3.14) | 3.14 has incomplete library support for torch/shap |

---

## 🎯 Day 1 Output vs Target

| Target | Status |
|---|---|
| Repo + folder structure complete | ✅ Done |
| Dependencies installed | ✅ Done |
| Class skeleton files created | ✅ Done |
| Working imports (compile check) | ✅ Done |
| Dataset downloaded to `data/raw/` | ✅ Done |

---

## 🔜 Day 2 Plan

- Open `notebooks/01_eda.ipynb`
- Load `data/raw/baf/Base.csv`
- Explore: fraud rate, shape, column types, missing values
- Identify temporal column (`month`) for time-based splitting
- Document key findings as EDA notes

---

*Report generated: 8 March 2026 | NuroGuard | USM VHack 2026*
