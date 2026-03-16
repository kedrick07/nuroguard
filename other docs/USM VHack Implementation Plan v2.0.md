# 🛡️ Fraud Shield — Implementation Plan v2.0
> **Varsity Hackathon 2026 | Case Study 2: Digital Trust — Real-Time Fraud Shield for the Unbanked**
> Track: Machine Learning (Fraud & Anomaly Detection) | SDG 8: Decent Work and Economic Growth (Target 8.10)
> **Updated: 12 March 2026 | Dataset: IEEE-CIS (Primary) | Status: Day 4 Complete ✅**

---

## ⚠️ What Changed From v1.0

| Item | v1.0 | v2.0 |
|---|---|---|
| Primary dataset | PaySim (synthetic) | IEEE-CIS (real transactions, 590K rows, 434 features) |
| Timeline start | Day 1 from scratch | Day 5 onwards from 12 March 2026 — Days 1–4 complete |
| Feature strategy | Basic velocity + geo features | + UID reconstruction, group aggregations, decimal splits, null-as-signal |
| Null handling | Median imputation | -999 replacement + `_was_null` binary flags |
| Competitive edge | Standard LightGBM pipeline | Kaggle top-solution techniques applied to ASEAN fraud context |

> **v1.0 plan is preserved as-is in `USM VHack Implementation Plan.md`. Do not modify it.**

---

## Table of Contents
1. [Project Summary](#1-project-summary)
2. [Case Study 2 Requirements Checklist](#2-case-study-2-requirements-checklist)
3. [Free Resources](#3-free-resources)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [OOP Module Breakdown](#5-oop-module-breakdown)
6. [Folder Structure](#6-folder-structure)
7. [Updated 10-Day Timeline](#7-updated-10-day-timeline)
8. [Innovative Feature Engineering Strategy](#8-innovative-feature-engineering-strategy)
9. [Deliverables](#9-deliverables)
10. [Evaluation Metrics & Targets](#10-evaluation-metrics--targets)
11. [Privacy & Data Integrity](#11-privacy--data-integrity)
12. [Scope Control — What to Cut](#12-scope-control--what-to-cut)

---

## 1. Project Summary

A real-time fraud detection system protecting unbanked and low-literacy digital wallet users in ASEAN. Every incoming transaction is scored in milliseconds and returned with an **Approve / Flag / Block** decision and a **plain-language reason code** so users know exactly why a payment was held.

### Core Innovations

| Innovation | Description | Why It Matters |
|---|---|---|
| **UID Reconstruction** | Reverse-engineer user identities from `card1`, `card2`, `addr1`, `P_emaildomain` — IEEE-CIS has no explicit user ID | Enables per-user behavioral features — the single biggest signal gap between average and top solutions |
| **Group Aggregation Features** | Per-UID, per-card, per-email-domain rolling stats (mean, std, count, deviation) | Answers "is this unusual FOR THIS USER?" not just "is this amount globally large?" |
| **Sequential Behaviour Modelling** | User transaction history modelled as a time-series using a lightweight LSTM | Catches slow-burn fraud — small probing transactions before a big theft |
| **SHAP Reason Codes** | Every decision includes top-3 plain-language explanations from SHAP feature attribution | Unbanked users get a human-readable SMS instead of a confusing block |
| **Geo-Velocity Detection** | Computes implied travel speed between consecutive transaction locations | Flags physically impossible travel (e.g., KL → Bangkok in 20 minutes) |
| **Null-as-Signal** | Missing identity fields treated as fraud signal, not just missing data | Missing device OS / browser info is genuinely correlated with fraud in IEEE-CIS |

### User Persona

> **Azri, 24, GrabFood rider — Petaling Jaya, Malaysia**
> - Earns ~RM 1,800/month entirely through TNG eWallet
> - No bank account; entire income sits in a digital wallet
> - Low digital literacy — doesn't understand "two-factor authentication"
> - A single fraudulent RM 500 withdrawal means he can't pay rent

---

## 2. Case Study 2 Requirements Checklist

| Requirement | Status | How We Address It |
|---|---|---|
| Behavioural Profiling (frequency, amount, location, time) | ✅ Core | UID reconstruction enables true per-user profiling; `FeatureEngineer` builds velocity, geo-velocity, temporal, and merchant pattern features |
| Real-Time Anomaly Scoring → Approve / Flag / Block | ✅ Core | `DecisionEngine` maps risk score to 3-tier decision with tunable thresholds |
| Imbalanced Class Handling (SMOTE / focal loss) | ✅ Core | `LightGBMModel` uses `scale_pos_weight` (~28 for IEEE-CIS); `DataProcessor` optionally applies SMOTE |
| Contextual Data Integration (device, IP) | ✅ Core | `FeatureEngineer` includes device change flags, email domain risk, address distance; IEEE-CIS `id_*` columns are direct inputs |
| Low Latency (real-time, sub-200ms) | ✅ Core | LightGBM ~10ms inference; SHAP ~15–50ms; total budget <200ms |
| False Positive Control (<2% FPR) | ✅ Core | `DecisionEngine` thresholds tuned on validation set to target FPR <2% |
| Privacy-First / Ethical Data Handling | ✅ Core | `PrivacyGuard` class — PII masking, hashed user IDs, no raw sensitive data in logs |
| Fraud Detection Engine (trained model + metrics) | ✅ Deliverable 1 | LightGBM + optional LSTM with ablation study table |
| Risk API Prototype | ✅ Deliverable 2 | FastAPI `/score` and `/health` endpoints with Swagger UI |
| Recommended Dataset: IEEE-CIS | ✅ Used | Primary training dataset (~590K transactions, 434 features) |
| Recommended Framework: XGBoost / LightGBM | ✅ Used | LightGBM as primary fast-path model |
| Recommended Framework: PyOD | ✅ Used | Optional Isolation Forest baseline via PyOD |
| Deployment: FastAPI | ✅ Used | Full FastAPI application with async endpoints |

---

## 3. Free Resources

### Datasets (100% Free)

| Dataset | Link | Size | Notes |
|---|---|---|---|
| **IEEE-CIS Fraud Detection** *(Primary)* | https://www.kaggle.com/c/ieee-fraud-detection | ~590K transactions, 434 features | Requires free Kaggle account — **CONFIRMED DOWNLOADED** ✅ |
| **Credit Card Fraud Detection** *(Fallback)* | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | 284K transactions | Simpler, faster to use if time is tight |
| **PaySim Synthetic Mobile Money** *(Archive)* | https://www.kaggle.com/datasets/ealaxi/paysim1 | 6.3M transactions | Used in Days 1–4; retired as primary dataset |

### Libraries & Frameworks (All Open-Source / Free)

| Category | Library | Install Command |
|---|---|---|
| Data processing | `pandas`, `numpy` | `pip install pandas numpy` |
| ML model | `lightgbm` | `pip install lightgbm` |
| Imbalanced handling | `imbalanced-learn` | `pip install imbalanced-learn` |
| Deep learning (LSTM) | `torch` | `pip install torch` |
| Anomaly detection | `pyod` | `pip install pyod` |
| Explainability | `shap` | `pip install shap` |
| API server | `fastapi`, `uvicorn` | `pip install fastapi uvicorn` |
| API data validation | `pydantic` | `pip install pydantic` |
| Dashboard (optional) | `streamlit` | `pip install streamlit` |
| Testing | `pytest` | `pip install pytest` |
| Math / distances | `scipy`, `geopy` | `pip install scipy geopy` |

---

## 4. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INCOMING TRANSACTION                      │
│   { user_id, amount, merchant, location, device, time }     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │      PrivacyGuard       │  ← Mask PII, hash user_id
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │     FeatureEngineer     │  ← UID Reconstruction,
              │                         │    Group Aggregations,
              │                         │    Velocity, Geo-velocity,
              │                         │    Behavioural, Contextual,
              │                         │    Null-as-Signal flags
              └──────────┬──────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
     ┌──────────────┐      ┌──────────────┐
     │ LightGBMModel│      │   LSTMModel  │
     │  (~10ms)     │      │  (sequence)  │  [STRETCH]
     └──────┬───────┘      └──────┬───────┘
            │                     │
            └──────────┬──────────┘
                       ▼
             ┌───────────────────┐
             │   FusionModel     │  ← Weighted avg: α·lgbm + (1-α)·lstm
             └────────┬──────────┘
                      │
                      ▼
             ┌───────────────────┐
             │  DecisionEngine   │  ← p<0.2 APPROVE / 0.2-0.6 FLAG / ≥0.6 BLOCK
             └────────┬──────────┘
                      │
                      ▼
             ┌───────────────────┐
             │ ExplanationService│  ← SHAP top-3 → plain-language reason codes
             └────────┬──────────┘
                      │
                      ▼
     ┌────────────────────────────────────┐
     │         FastAPI Response           │
     │  { decision, risk_score, reasons } │
     └────────────────────────────────────┘
```

### Latency Budget

| Stage | Target Time |
|---|---|
| Feature Engineering | ~30ms |
| LightGBM Inference | ~10ms |
| SHAP Explanation | ~15–50ms |
| API Overhead | ~20ms |
| **Total End-to-End** | **< 200ms** |

---

## 5. OOP Module Breakdown

> **No changes from v1.0.** All source code lives in `src/`. Each module is a Python class in its own file.
> Dependencies flow in one direction: API → Services → Models → Features → Entities.

---

### 5.1 Entities Layer (`src/entities/`)

#### `Transaction` — `transaction.py`

```python
class Transaction:
    transaction_id: str
    user_id: str          # hashed before processing
    amount: float
    merchant_id: str
    merchant_category: str
    location_lat: float
    location_lon: float
    device_id: str
    ip_address: str
    timestamp: datetime

    def to_dict() -> dict
    def from_dict(data: dict) -> Transaction
    def validate() -> bool
```

#### `UserProfile` — `user_profile.py`

```python
class UserProfile:
    user_id: str                      # hashed
    recent_transactions: list[dict]
    avg_amount_7d: float
    std_amount_7d: float
    avg_amount_30d: float
    known_devices: set[str]
    known_locations: list[tuple]
    known_countries: set[str]
    usual_hour_range: tuple[int, int]

    def update(transaction: Transaction) -> None
    def is_known_device(device_id: str) -> bool
    def is_known_country(country_code: str) -> bool
    def get_recent_sequence(n: int) -> list[dict]
```

#### `ScoringResult` — `scoring_result.py`

```python
class ScoringResult:
    transaction_id: str
    decision: str          # "APPROVE" | "FLAG" | "BLOCK"
    risk_score: float      # 0.0 – 1.0
    reasons: list[str]
    latency_ms: float
    timestamp: datetime

    def to_json() -> dict
    def is_blocked() -> bool
    def is_flagged() -> bool
```

---

### 5.2 Features Layer (`src/features/`)

#### `FeatureEngineer` — `feature_engineer.py`

> **v2.0 addition:** UID reconstruction and group aggregation methods are now first-class features.

```python
class FeatureEngineer:
    feature_names: list[str]

    def build_features(transaction: Transaction, profile: UserProfile) -> pd.Series
    def _reconstruct_uid(df: pd.DataFrame) -> pd.Series        # NEW v2.0
    def _group_aggregation_features(df: pd.DataFrame) -> pd.DataFrame  # NEW v2.0
    def _amount_decomposition(df: pd.DataFrame) -> pd.DataFrame        # NEW v2.0
    def _null_signal_flags(df: pd.DataFrame) -> pd.DataFrame           # NEW v2.0
    def _velocity_features(transaction, profile) -> dict
    def _geo_velocity_features(transaction, profile) -> dict
    def _behavioural_features(transaction, profile) -> dict
    def _contextual_features(transaction, profile) -> dict
    def get_feature_names() -> list[str]
```

**UID Reconstruction (v2.0):**
```python
df['uid'] = (df['card1'].astype(str) + '_' +
             df['card2'].astype(str) + '_' +
             df['addr1'].astype(str) + '_' +
             df['P_emaildomain'].astype(str))
```

**Group Aggregation Features (v2.0):**
```python
for col in ['card1', 'card2', 'P_emaildomain', 'uid']:
    df[f'{col}_amt_mean']      = df.groupby(col)['TransactionAmt'].transform('mean')
    df[f'{col}_amt_std']       = df.groupby(col)['TransactionAmt'].transform('std')
    df[f'{col}_txn_count']     = df.groupby(col)['TransactionAmt'].transform('count')
    df[f'{col}_amt_deviation'] = df['TransactionAmt'] - df[f'{col}_amt_mean']
```

**Amount Decomposition (v2.0):**
```python
df['amt_integer']    = np.floor(df['TransactionAmt']).astype(int)
df['amt_decimal']    = df['TransactionAmt'] - df['amt_integer']
df['is_round_amount'] = (df['amt_decimal'] == 0).astype(int)
# Fraudsters test cards with round amounts (100.00, 50.00)
# Real users spend oddly (47.83, 312.99)
```

**Null-as-Signal Flags (v2.0):**
```python
identity_cols = ['id_30', 'id_31', 'id_33', 'DeviceType', 'DeviceInfo']
for col in identity_cols:
    df[f'{col}_was_null'] = df[col].isnull().astype(int)
# Fill numerical nulls with -999 (not median)
# Fill categorical nulls with 'unknown'
# -999 is out-of-range so LightGBM learns "missing = different pattern"
```

**Velocity features computed:**
- `txn_count_1h`, `txn_count_24h`
- `amount_sum_1h`
- `velocity_ratio` = `txn_count_1h / avg_hourly_txn_30d`
- `amount_ratio` = `amount / avg_amount_30d`

**Geo-velocity features computed:**
- `distance_km`, `time_delta_hours`
- `implied_speed_kmh` = `distance_km / time_delta_hours`
- `is_impossible_travel` — flag if speed > 900 km/h
- `is_new_country`

**Behavioural features computed:**
- `avg_amount_7d`, `std_amount_7d`
- `hour_of_day`, `day_of_week`
- `is_unusual_hour`
- `is_new_device`
- `merchant_category_match`

#### `GeoVelocityCalculator` — `geo_velocity.py`

```python
class GeoVelocityCalculator:
    IMPOSSIBLE_SPEED_KMH: float = 900.0

    def haversine_distance(lat1, lon1, lat2, lon2) -> float
    def implied_speed(distance_km, time_delta_hours) -> float
    def is_impossible_travel(speed_kmh: float) -> bool
    def get_country_code(lat, lon) -> str
```

#### `FeatureValidator` — `validators.py`

```python
class FeatureValidator:
    def validate(features: pd.Series) -> bool
    def fill_missing(features: pd.Series) -> pd.Series
    def clip_outliers(features: pd.Series) -> pd.Series
```

---

### 5.3 Models Layer (`src/models/`)

#### `BaseFraudModel` — `base_model.py` *(Abstract)*

```python
from abc import ABC, abstractmethod

class BaseFraudModel(ABC):
    model_name: str
    is_trained: bool

    @abstractmethod
    def train(X_train, y_train) -> None

    @abstractmethod
    def predict_score(features: pd.Series) -> float

    @abstractmethod
    def save(path: str) -> None

    @abstractmethod
    def load(path: str) -> None

    def is_ready() -> bool
```

#### `LightGBMModel` — `lightgbm_model.py`

```python
class LightGBMModel(BaseFraudModel):
    model_name = "lightgbm"

    model: lgb.Booster
    feature_names: list[str]
    scale_pos_weight: float     # ~28 for IEEE-CIS (derive live, never hardcode)
    threshold_approve: float    # default 0.2
    threshold_block: float      # default 0.6

    def train(X_train, y_train, X_val=None, y_val=None) -> None
    def predict_score(features: pd.Series) -> float
    def predict_batch(X: pd.DataFrame) -> np.ndarray
    def get_feature_importance() -> pd.DataFrame
    def save(path: str) -> None
    def load(path: str) -> None
    def evaluate(X_val, y_val) -> dict
```

#### `LSTMModel` — `lstm_model.py` *(Stretch)*

```python
class LSTMModel(BaseFraudModel):
    model_name = "lstm"

    sequence_length: int = 5
    input_size: int = 5
    hidden_size: int = 32
    model: nn.Module

    def build_sequences(profile: UserProfile) -> torch.Tensor
    def train(sequences, labels) -> None
    def predict_score(profile: UserProfile) -> float
    def save(path: str) -> None
    def load(path: str) -> None
```

#### `FusionModel` — `fusion_model.py`

```python
class FusionModel(BaseFraudModel):
    model_name = "fusion"

    lgbm_model: LightGBMModel
    lstm_model: LSTMModel | None
    alpha: float = 0.65

    def predict_score(features: pd.Series, profile: UserProfile) -> float
    # final_score = alpha * lgbm_score + (1 - alpha) * lstm_score
    # Falls back to lgbm_score alone if lstm_model is None
    def set_alpha(alpha: float) -> None
    def save(path: str) -> None
    def load(path: str) -> None
```

---

### 5.4 Services Layer (`src/services/`)

#### `DecisionEngine` — `decision_engine.py`

```python
class DecisionEngine:
    threshold_approve: float = 0.2
    threshold_block: float = 0.6

    def decide(risk_score: float) -> str
    def set_thresholds(approve: float, block: float) -> None
    def tune_thresholds(y_true, y_scores, target_fpr=0.02) -> tuple
```

#### `ExplanationService` — `explanation_service.py`

```python
class ExplanationService:
    explainer: shap.TreeExplainer
    top_n: int = 3
    reason_templates: dict

    def explain(features: pd.Series) -> list[str]
    def _get_shap_values(features: pd.Series) -> np.ndarray
    def _top_features(shap_values: np.ndarray) -> list[str]
    def _format_reason(feature_name: str, feature_value, shap_value: float) -> str

    # Reason templates (v2.0 additions marked):
    # uid_amt_deviation      → "This amount is {X}× higher than your usual spending"  [NEW]
    # is_round_amount        → "Suspiciously round amount — common in card testing"    [NEW]
    # id_30_was_null         → "Transaction from an unrecognised device type"          [NEW]
    # velocity_ratio         → "You are spending {X}× faster than usual"
    # implied_speed_kmh      → "Transaction from {city_B} — you were in {city_A} {N} min ago"
    # hour_of_day            → "Transaction at {time} — you usually transact {start}–{end}"
    # amount_ratio           → "Amount is {X}× your average transaction"
    # is_new_device          → "Transaction from a device you haven't used before"
    # is_impossible_travel   → "Physically impossible location change detected"
```

#### `PrivacyGuard` — `privacy_guard.py`

```python
class PrivacyGuard:
    ALLOWED_LOG_FIELDS: set = {"transaction_id", "decision", "risk_score", "timestamp"}

    def hash_user_id(user_id: str) -> str
    def mask_ip(ip_address: str) -> str
    def mask_device_id(device_id: str) -> str
    def sanitize_for_log(result: ScoringResult) -> dict
    def validate_input_fields(transaction: Transaction) -> bool
```

#### `DataProcessor` — `data_processor.py`

> **v2.0:** `load_ieee_cis` is now the primary method. PaySim loader is archived.

```python
class DataProcessor:
    raw_data_path: str
    processed_data_path: str

    def load_ieee_cis(transaction_path: str, identity_path: str) -> pd.DataFrame
    def merge_tables(transactions: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame
    # Always: pd.merge(transactions, identity, how="left", on="TransactionID")
    def clean(df: pd.DataFrame) -> pd.DataFrame
    # Numerical nulls  → fill with -999 (NOT median)
    # Categorical nulls → fill with "unknown"
    # Columns >95% null → drop
    def time_based_split(df, train_ratio=0.8) -> tuple
    def apply_smote(X_train, y_train) -> tuple
    def save_processed(df: pd.DataFrame, path: str) -> None
```

#### `RiskScoringService` — `risk_scoring_service.py`

```python
class RiskScoringService:
    privacy_guard: PrivacyGuard
    feature_engineer: FeatureEngineer
    feature_validator: FeatureValidator
    fusion_model: FusionModel
    decision_engine: DecisionEngine
    explanation_service: ExplanationService
    user_profiles: dict[str, UserProfile]

    def score(transaction: Transaction) -> ScoringResult
    # Pipeline:
    # 1. privacy_guard.hash_user_id + mask_ip
    # 2. Load or create UserProfile
    # 3. feature_engineer.build_features(transaction, profile)
    # 4. feature_validator.validate + fill_missing
    # 5. fusion_model.predict_score(features, profile)
    # 6. decision_engine.decide(score)
    # 7. explanation_service.explain(features)
    # 8. profile.update(transaction)
    # 9. Return ScoringResult

    def get_user_profile(user_id: str) -> UserProfile
    def health_check() -> dict
```

---

### 5.5 API Layer (`src/api/`)

```python
# routes.py
router = APIRouter()

@router.post("/score", response_model=ScoringResultSchema)
async def score_transaction(request: TransactionSchema):
    transaction = Transaction.from_dict(request.dict())
    result = risk_scoring_service.score(transaction)
    return result.to_json()

@router.get("/health")
async def health():
    return risk_scoring_service.health_check()

@router.get("/profile/{user_id}")
async def get_profile(user_id: str):
    return risk_scoring_service.get_user_profile(user_id)
```

```python
# schemas.py
class TransactionSchema(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    merchant_id: str
    merchant_category: str
    location_lat: float
    location_lon: float
    device_id: str
    ip_address: str
    timestamp: datetime

class ScoringResultSchema(BaseModel):
    transaction_id: str
    decision: str
    risk_score: float
    reasons: list[str]
    latency_ms: float
```

---

### 5.6 Utilities (`src/utils/`)

| File | Class / Purpose |
|---|---|
| `config.py` | `Config` — loads env vars and model paths from `.env` |
| `logger.py` | `FraudLogger` — structured JSON logging, PII-safe output |
| `metrics.py` | `ModelEvaluator` — PR-AUC, F1, FPR, confusion matrix, latency timer |
| `serializer.py` | `ModelSerializer` — save/load `.pkl` and `.pt` files consistently |

---

## 6. Folder Structure

```
fraud-shield/
│
├── data/
│   ├── raw/                         # gitignored
│   │   ├── train_transaction.csv    # IEEE-CIS ✅
│   │   ├── train_identity.csv       # IEEE-CIS ✅
│   │   └── test_transaction.csv     # IEEE-CIS ✅
│   └── processed/
│       ├── features_train.parquet   # Will be regenerated on IEEE-CIS
│       └── features_val.parquet     # Will be regenerated on IEEE-CIS
│
├── notebooks/
│   ├── 01_eda.ipynb                 # REDO on IEEE-CIS (Day 5 Phase 1)
│   ├── 02_features.ipynb            # REDO on IEEE-CIS (Day 5 Phase 1)
│   ├── 03_model.ipynb               # RETRAIN on IEEE-CIS (Day 5 Phase 1)
│   ├── 04_lstm.ipynb                # [STRETCH] LSTM sequential model
│   └── 05_evaluation.ipynb          # Ablation study + final metrics
│
├── src/                             # All OOP modules — unchanged from v1.0
│   ├── entities/
│   ├── features/
│   ├── models/
│   ├── services/
│   ├── api/
│   └── utils/
│
├── models/
│   ├── lgbm_baseline_paysim.pkl     # Archive — Day 4 PaySim model
│   └── lgbm_baseline_ieee.pkl       # Target — IEEE-CIS retrained model
│
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 7. Updated 10-Day Timeline

> **Days 1–4: COMPLETE ✅** (PaySim baseline, EDA, feature engineering, LightGBM trained)
> **Day 5 onwards: IEEE-CIS dataset, innovative features, full pipeline**

| Day | Date | Phase | Tasks | Output |
|---|---|---|---|---|
| **Day 1** | 8 Mar | ~~Setup~~ | ~~Repo, folder structure, dependencies, class skeletons~~ | ✅ Done |
| **Day 2** | 9 Mar | ~~EDA~~ | ~~PaySim EDA — fraud rate, distributions, class imbalance~~ | ✅ Done |
| **Day 3** | 10 Mar | ~~Data Pipeline~~ | ~~Feature engineering on PaySim, train/val split, parquet files~~ | ✅ Done |
| **Day 4** | 11 Mar | ~~LightGBM Baseline~~ | ~~Train on PaySim, PR-AUC 0.97, save lgbm_baseline_paysim.pkl~~ | ✅ Done |
| **Day 5** | 12 Mar | **Dataset Switch + Feature Innovation** | **Phase 1:** Redo EDA + features on IEEE-CIS; implement UID reconstruction, group aggregations, decimal splits, null-as-signal flags; retrain LightGBM on IEEE-CIS | New parquets + `lgbm_baseline_ieee.pkl` |
| **Day 6** | 13 Mar | **Decision Engine + Privacy** | Build `DecisionEngine` — tune Approve/Flag/Block thresholds on IEEE-CIS validation set; build `PrivacyGuard` — hash user IDs, mask IPs, PII-safe logging | Decision layer working end-to-end, FPR < 2% confirmed |
| **Day 7** | 14 Mar | **SHAP Explainability** | Build `ExplanationService` — TreeExplainer on IEEE-CIS LightGBM, extract top-3 features, map to plain-language templates including v2.0 reason codes (uid_amt_deviation, is_round_amount, null flags) | Reason codes working for all sample personas |
| **Day 8** | 15 Mar | **FastAPI Risk API** | Build `RiskScoringService` orchestrator; build `routes.py` + `schemas.py`; wire full pipeline; test with `curl` and Swagger UI | Live `/score` and `/health` endpoints |
| **Day 9** | 16 Mar | **Testing + Polish** | Write key tests; run full latency measurement; produce ablation study table (PaySim baseline → IEEE-CIS raw → +UID features → +group agg → +SHAP) | Test suite passing, latency confirmed <200ms |
| **Day 10** | 17 Mar | **Stretch + Demo Prep** | [STRETCH] LSTM or Streamlit dashboard; prepare slides with Azri persona story, architecture, results, live API demo | Final demo-ready submission |

---

## 8. Innovative Feature Engineering Strategy

> This section is the core competitive differentiator of v2.0. Every team uses the same IEEE-CIS dataset.
> This is how NuroGuard separates from the pack.

### Why Standard Cleaning Is Not Enough

Every competing team will:
- Merge the two tables
- Impute nulls with median
- Train LightGBM on raw + basic features
- Get PR-AUC ~0.70–0.80

NuroGuard targets **PR-AUC ≥ 0.88** using techniques from Kaggle top-10 solutions applied to the ASEAN fraud context.

---

### 8.1 UID Reconstruction (Highest Priority)

IEEE-CIS deliberately removed the user ID column. **Reconstructing it is the single biggest signal gain** — it unlocks true per-user behavioral profiling.

```python
# Combine card + address + email domain signals to fingerprint a user
df['uid'] = (df['card1'].astype(str) + '_' +
             df['card2'].astype(str) + '_' +
             df['addr1'].astype(str) + '_' +
             df['P_emaildomain'].astype(str))

df['uid2'] = (df['card1'].astype(str) + '_' +
              df['card2'].astype(str) + '_' +
              df['addr1'].astype(str))
# uid2 as fallback when email domain is null
```

**Why this works:** Card number prefix + billing address + email domain together uniquely identify a cardholder even without an explicit ID. This lets us ask: *"Is this transaction unusual for THIS person?"* not just globally.

---

### 8.2 Group Aggregation Features (High Priority)

Once UIDs are reconstructed, build per-user and per-card statistical profiles:

```python
agg_groups = ['card1', 'card2', 'P_emaildomain', 'uid', 'uid2', 'addr1']

for col in agg_groups:
    df[f'{col}_amt_mean']       = df.groupby(col)['TransactionAmt'].transform('mean')
    df[f'{col}_amt_std']        = df.groupby(col)['TransactionAmt'].transform('std')
    df[f'{col}_txn_count']      = df.groupby(col)['TransactionAmt'].transform('count')
    df[f'{col}_amt_deviation']  = df['TransactionAmt'] - df[f'{col}_amt_mean']
    df[f'{col}_amt_zscore']     = df[f'{col}_amt_deviation'] / (df[f'{col}_amt_std'] + 1e-9)
```

**Key feature — `amt_zscore`:** How many standard deviations above this user's normal is this transaction? A z-score > 3 is a very strong fraud signal regardless of absolute amount.

---

### 8.3 TransactionAmt Decomposition (Quick Win)

```python
df['amt_integer']     = np.floor(df['TransactionAmt']).astype(int)
df['amt_decimal']     = df['TransactionAmt'] - df['amt_integer']
df['is_round_amount'] = (df['amt_decimal'] == 0).astype(int)
df['log_amt']         = np.log1p(df['TransactionAmt'])  # stabilise skewed distribution
```

**Why this works:** Card testing fraud involves charging round numbers (50.00, 100.00). Real purchases almost never end in .00. The `amt_decimal` feature alone has meaningful fraud correlation in IEEE-CIS.

---

### 8.4 Null-as-Signal (Beats Median Imputation)

```python
# Step 1: Flag which identity columns are missing BEFORE imputation
identity_signal_cols = ['id_30', 'id_31', 'id_33', 'id_36',
                        'DeviceType', 'DeviceInfo']
for col in identity_signal_cols:
    df[f'{col}_was_null'] = df[col].isnull().astype(int)

# Step 2: Fill with -999 (not median) for numerical columns
# LightGBM will learn that -999 = a distinct pattern, not just a small number
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(-999)

# Step 3: Fill categorical nulls with 'unknown'
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna('unknown')

# Step 4: Drop columns with >95% nulls entirely
null_rates = df.isnull().mean()
df = df.drop(columns=null_rates[null_rates > 0.95].index)
```

**Why -999 beats median:** Median tells the model "this is a typical value." -999 tells the model "this field was absent" — a fundamentally different signal that LightGBM can branch on explicitly.

---

### 8.5 Time Feature Extraction

The `TransactionDT` column is seconds elapsed — extract meaningful time components:

```python
# TransactionDT is seconds from a reference point — convert to interpretable features
df['hour']        = (df['TransactionDT'] // 3600) % 24
df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
df['is_night']    = ((df['hour'] < 6) | (df['hour'] >= 23)).astype(int)
# Late night / early morning transactions are higher fraud risk
```

---

### 8.6 Feature Importance — Two-Pass Training Strategy

Do NOT guess which features are noise. Let the model tell you:

1. **Pass 1:** Train LightGBM on all engineered features (~80–120 features total)
2. **Extract importance:** `model.feature_importance(importance_type='gain')`
3. **Drop bottom 20%** of features by gain importance
4. **Pass 2:** Retrain on reduced clean feature set → this is your final model

This is safer and more accurate than manual feature selection.

---

### 8.7 Summary — v2.0 Feature Priority Table

| Feature Group | v1.0 | v2.0 | Competitive Impact |
|---|---|---|---|
| UID reconstruction | ❌ | ✅ | 🔴 Highest — unlocks per-user profiling |
| Group aggregations (mean, std, z-score) | ❌ | ✅ | 🔴 Highest — most teams skip this |
| TransactionAmt decimal split | ❌ | ✅ | 🟡 Medium — quick win, 3 lines |
| Null-as-signal (-999 + `_was_null`) | ❌ | ✅ | 🟡 Medium — beats median imputation |
| Time feature extraction | ❌ | ✅ | 🟡 Medium — night/weekend fraud patterns |
| Velocity features | ✅ | ✅ | 🟢 Standard — expected |
| Geo-velocity features | ✅ | ✅ | 🟢 Standard — expected |
| SHAP reason codes | ✅ | ✅ | 🔴 Highest for presentation — judges remember this |

---

## 9. Deliverables

### Deliverable 1 — Fraud Detection Engine *(Case Study 2 Required)*
- [ ] Trained `LightGBMModel` on IEEE-CIS saved as `lgbm_baseline_ieee.pkl`
- [ ] Feature engineering pipeline with v2.0 innovative features (~80–120 features)
- [ ] Ablation study table: PaySim baseline → IEEE-CIS raw → +UID features → +group agg → +SHAP tuned
- [ ] Evaluation report: PR-AUC, F1, Recall, FPR, Inference Latency

### Deliverable 2 — Risk API Prototype *(Case Study 2 Required)*
- [ ] FastAPI application running on `localhost:8000`
- [ ] `POST /score` — accepts `TransactionSchema`, returns `ScoringResultSchema`
- [ ] `GET /health` — confirms model loaded and latency stats
- [ ] Swagger UI at `localhost:8000/docs`
- [ ] Sample request/response JSON in README

### Deliverable 3 — Demo & Presentation *(Hackathon Required)*
- [ ] Live API demo scoring 3 personas (Approve, Flag, Block)
- [ ] Architecture diagram
- [ ] Azri persona narrative — plain-language SMS reason codes
- [ ] Ablation study results table in slides
- [ ] Privacy & latency budget slide

### Stretch Deliverables *(Only if V1 complete)*
- [ ] LSTM sequential model + fusion scoring
- [ ] Streamlit dashboard with live scoring and user profile view
- [ ] Imbalanced handling comparison: `scale_pos_weight` vs SMOTE vs focal loss

---

## 10. Evaluation Metrics & Targets

> **Never use plain accuracy** — meaningless for 3.5% fraud rate data.

| Metric | v1.0 Target | v2.0 Target | Why |
|---|---|---|---|
| **Precision-Recall AUC** | ≥ 0.75 | **≥ 0.88** | Innovative features should push well above baseline |
| **Recall @ Precision=0.5** | ≥ 0.85 | ≥ 0.85 | Unchanged |
| **F1 Score** | ≥ 0.70 | **≥ 0.78** | Higher with better features |
| **False Positive Rate** | < 2% | < 2% | Case Study 2 hard constraint |
| **Inference Latency** | < 200ms | < 200ms | Case Study 2 hard constraint |

### Ablation Study Template (v2.0)

| Configuration | PR-AUC | F1 | FPR | Notes |
|---|---|---|---|---|
| PaySim baseline (archive) | 0.9703 | 0.9834 | 0.0003 | Synthetic — not meaningful |
| IEEE-CIS raw features only | TBD | TBD | TBD | Everyone's baseline |
| + UID reconstruction | TBD | TBD | TBD | Our first differentiator |
| + Group aggregations | TBD | TBD | TBD | Per-user behavioural features |
| + Decimal split + null flags | TBD | TBD | TBD | Fine-grained signal |
| + SHAP threshold tuning | TBD | TBD | TBD | FPR control layer |
| + LSTM fusion [STRETCH] | TBD | TBD | TBD | Sequential behaviour |

---

## 11. Privacy & Data Integrity

| Requirement | Implementation |
|---|---|
| No raw PII in logs | `PrivacyGuard.sanitize_for_log()` strips user_id, IP, device_id |
| Hashed user identifiers | SHA-256 one-way hash; original ID never stored |
| Masked IP addresses | `x.x.x.XXX` format |
| No PAN / CVV in inputs | `PrivacyGuard.validate_input_fields()` rejects card numbers |
| Data minimisation | Only ~80–120 engineered features enter the model; raw 434-column dataset never touches API |
| UID reconstruction privacy note | Reconstructed UIDs are hashed before any logging — never stored as raw card+address strings |

---

## 12. Scope Control — What to Cut

### Version 1 (Must Ship by Day 8)
```
DataProcessor (IEEE-CIS) → FeatureEngineer (v2.0) → LightGBMModel → DecisionEngine → ExplanationService → FastAPI
```

### Version 2 (Day 9–10, stretch only)
```
LSTMModel → FusionModel → Streamlit Dashboard
```

### What to Cut If Behind Schedule

1. Drop Streamlit dashboard — demo via Swagger UI
2. Skip LSTM — LightGBM + v2.0 features + SHAP is already a strong submission
3. Drop geo-velocity if location data is messy — use `dist1`, `dist2` proxy columns from IEEE-CIS instead
4. Skip PyOD anomaly baseline

### What to NEVER Cut

- ✅ UID reconstruction — this is what separates NuroGuard from other teams
- ✅ SHAP explanations — the differentiator judges remember
- ✅ Time-based train/val split — random split inflates metrics
- ✅ Privacy handling — explicitly required by Case Study 2
- ✅ FPR < 2% — explicitly required by Case Study 2
- ✅ Azri persona narrative — makes the presentation human and memorable

### Risk Matrix

| Risk | Likelihood | Mitigation |
|---|---|---|
| IEEE-CIS nulls break pipeline | Medium | -999 + null flags strategy handles all cases |
| UID reconstruction produces too many unique IDs | Medium | Fall back to `card1` + `card2` only; still better than no UID |
| Group agg features cause data leakage | Low | Compute aggregations on train set only, apply to val — use `transform` not `merge` |
| LSTM training exceeds time budget | High | Skip LSTM; LightGBM v2.0 features alone are sufficient |
| SHAP takes >200ms | Low | Use background thread; return decision immediately |

---

*Plan version: 2.0 | Updated: 12 March 2026 | NuroGuard | USM VHack 2026*
*v1.0 preserved in: `USM VHack Implementation Plan.md`*
