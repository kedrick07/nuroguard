# 🛡️ Fraud Shield — Project Plan
> **Varsity Hackathon 2026 | Case Study 2: Digital Trust — Real-Time Fraud Shield for the Unbanked**
> Track: Machine Learning (Fraud & Anomaly Detection) | SDG 8: Decent Work and Economic Growth (Target 8.10)

---

## Table of Contents
1. [Project Summary](#1-project-summary)
2. [Case Study 2 Requirements Checklist](#2-case-study-2-requirements-checklist)
3. [Free Resources](#3-free-resources)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [OOP Module Breakdown](#5-oop-module-breakdown)
6. [Folder Structure](#6-folder-structure)
7. [10-Day Timeline](#7-10-day-timeline)
8. [Deliverables](#8-deliverables)
9. [Evaluation Metrics & Targets](#9-evaluation-metrics--targets)
10. [Privacy & Data Integrity](#10-privacy--data-integrity)
11. [Scope Control — What to Cut](#11-scope-control--what-to-cut)

---

## 1. Project Summary

A real-time fraud detection system protecting unbanked and low-literacy digital wallet users in ASEAN. Every incoming transaction is scored in milliseconds and returned with an **Approve / Flag / Block** decision and a **plain-language reason code** so users know exactly why a payment was held.

### Core Innovations

| Innovation | Description | Why It Matters |
|---|---|---|
| **Sequential Behaviour Modelling** | User transaction history modelled as a time-series using a lightweight LSTM | Catches slow-burn fraud — small probing transactions before a big theft |
| **SHAP Reason Codes** | Every decision includes top-3 plain-language explanations from SHAP feature attribution | Unbanked users get a human-readable SMS instead of a confusing block |
| **Geo-Velocity Detection** | Computes implied travel speed between consecutive transaction locations | Flags physically impossible travel (e.g., KL → Bangkok in 20 minutes) |

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
| Behavioural Profiling (frequency, amount, location, time) | ✅ Core | `FeatureEngineer` class builds velocity, geo-velocity, temporal, and merchant pattern features |
| Real-Time Anomaly Scoring → Approve / Flag / Block | ✅ Core | `DecisionEngine` class maps risk score to 3-tier decision with tunable thresholds |
| Imbalanced Class Handling (SMOTE / focal loss) | ✅ Core | `LightGBMModel` uses `scale_pos_weight`; `DataProcessor` optionally applies SMOTE |
| Contextual Data Integration (device, IP) | ✅ Core | `FeatureEngineer` includes device change flags, email domain risk, address distance |
| Low Latency (real-time, sub-200ms) | ✅ Core | LightGBM ~10ms inference; SHAP ~15–50ms; total budget <200ms |
| False Positive Control (<2% FPR) | ✅ Core | `DecisionEngine` thresholds tuned on validation set to target FPR <2% |
| Privacy-First / Ethical Data Handling | ✅ Core | `PrivacyGuard` class — PII masking, hashed user IDs, no raw sensitive data in logs |
| Fraud Detection Engine (trained model + metrics) | ✅ Deliverable 1 | LightGBM + optional LSTM with ablation study table |
| Risk API Prototype | ✅ Deliverable 2 | FastAPI `/score` and `/health` endpoints with Swagger UI |
| Recommended Dataset: IEEE-CIS | ✅ Used | Primary training dataset (~590K transactions) |
| Recommended Framework: XGBoost / LightGBM | ✅ Used | LightGBM as primary fast-path model |
| Recommended Framework: PyOD | ✅ Used | Optional Isolation Forest baseline via PyOD |
| Deployment: FastAPI | ✅ Used | Full FastAPI application with async endpoints |

---

## 3. Free Resources

### Datasets (100% Free)

| Dataset | Link | Size | Notes |
|---|---|---|---|
| **IEEE-CIS Fraud Detection** *(Primary)* | https://www.kaggle.com/c/ieee-fraud-detection | ~590K transactions, 434 features | Requires free Kaggle account |
| **Credit Card Fraud Detection** *(Fallback)* | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | 284K transactions | Simpler, faster to use if time is tight |
| **PaySim Synthetic Mobile Money** *(Optional)* | https://www.kaggle.com/datasets/ealaxi/paysim1 | 6.3M transactions | Closer to ASEAN e-wallet context |

> **Download instructions:** Create a free Kaggle account → go to the dataset page → click Download.
> All datasets are licensed for educational/research use at no cost.

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

### Free Development Environments

| Tool | Link | Notes |
|---|---|---|
| **Google Colab** | https://colab.research.google.com | Free GPU for LSTM training |
| **Kaggle Notebooks** | https://www.kaggle.com/code | Free GPU + datasets already available in-kernel |
| **VS Code** | https://code.visualstudio.com | Local IDE, free |
| **GitHub** | https://github.com | Free repo hosting + version control |

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
              │     FeatureEngineer     │  ← Velocity, Geo-velocity,
              │                         │    Behavioural, Contextual
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

All source code lives in `src/`. Each module is a Python class in its own file.
Dependencies flow in one direction: API → Services → Models → Features → Entities.

---

### 5.1 Entities Layer (`src/entities/`)

#### `Transaction` — `transaction.py`
Represents a single incoming payment request. Acts as the data contract between the API and the rest of the system.

```python
class Transaction:
    # Attributes
    transaction_id: str
    user_id: str          # hashed before processing
    amount: float
    merchant_id: str
    merchant_category: str
    location_lat: float
    location_lon: float
    device_id: str
    ip_address: str       # passed to PrivacyGuard before use
    timestamp: datetime

    # Methods
    def to_dict() -> dict
    def from_dict(data: dict) -> Transaction
    def validate() -> bool        # check required fields, type safety
```

#### `UserProfile` — `user_profile.py`
Stores the behavioural history for a user, used by `FeatureEngineer` to compute rolling aggregates.

```python
class UserProfile:
    # Attributes
    user_id: str                      # hashed
    recent_transactions: list[dict]   # last N transactions (rolling window)
    avg_amount_7d: float
    std_amount_7d: float
    avg_amount_30d: float
    known_devices: set[str]
    known_locations: list[tuple]      # (lat, lon) history
    known_countries: set[str]
    usual_hour_range: tuple[int, int] # e.g., (8, 22)

    # Methods
    def update(transaction: Transaction) -> None   # add new txn, recompute stats
    def is_known_device(device_id: str) -> bool
    def is_known_country(country_code: str) -> bool
    def get_recent_sequence(n: int) -> list[dict]  # for LSTM input
```

#### `ScoringResult` — `scoring_result.py`
Immutable output object returned by the full pipeline and serialized by the API.

```python
class ScoringResult:
    # Attributes
    transaction_id: str
    decision: str          # "APPROVE" | "FLAG" | "BLOCK"
    risk_score: float      # 0.0 – 1.0
    reasons: list[str]     # plain-language explanation strings
    latency_ms: float      # total processing time
    timestamp: datetime

    # Methods
    def to_json() -> dict
    def is_blocked() -> bool
    def is_flagged() -> bool
```

---

### 5.2 Features Layer (`src/features/`)

#### `FeatureEngineer` — `feature_engineer.py`
Takes a `Transaction` and a `UserProfile` and outputs a feature vector for the model. Central to the system — most detection value lives here.

```python
class FeatureEngineer:
    # Attributes
    feature_names: list[str]    # ordered list of output feature names

    # Methods
    def build_features(transaction: Transaction, profile: UserProfile) -> pd.Series
    def _velocity_features(transaction, profile) -> dict
    def _geo_velocity_features(transaction, profile) -> dict
    def _behavioural_features(transaction, profile) -> dict
    def _contextual_features(transaction, profile) -> dict
    def get_feature_names() -> list[str]
```

**Velocity features computed:**
- `txn_count_1h`, `txn_count_24h` — transaction frequency
- `amount_sum_1h` — spend in last hour
- `velocity_ratio` = `txn_count_1h / avg_hourly_txn_30d`
- `amount_ratio` = `amount / avg_amount_30d`

**Geo-velocity features computed:**
- `distance_km` — great-circle distance from last transaction
- `time_delta_hours` — time since last transaction
- `implied_speed_kmh` = `distance_km / time_delta_hours`
- `is_impossible_travel` — flag if speed > 900 km/h
- `is_new_country` — first time transacting from this country

**Behavioural features computed:**
- `avg_amount_7d`, `std_amount_7d`
- `hour_of_day`, `day_of_week`
- `is_unusual_hour` — outside user's normal hours
- `is_new_device` — device not in known_devices
- `merchant_category_match` — matches usual category

**Contextual / identity features computed:**
- `email_domain_risk` — 0=corporate, 1=free, 2=disposable
- `addr_distance` — billing vs transaction address gap
- `card_type_mismatch` — binary flag

#### `GeoVelocityCalculator` — `geo_velocity.py`
Standalone utility class for geographic distance and speed calculations.

```python
class GeoVelocityCalculator:
    IMPOSSIBLE_SPEED_KMH: float = 900.0   # class constant

    # Methods
    def haversine_distance(lat1, lon1, lat2, lon2) -> float   # returns km
    def implied_speed(distance_km, time_delta_hours) -> float
    def is_impossible_travel(speed_kmh: float) -> bool
    def get_country_code(lat, lon) -> str
```

#### `FeatureValidator` — `validators.py`
Validates feature vectors before passing to models. Guards against nulls, infinities, and out-of-range values.

```python
class FeatureValidator:
    # Methods
    def validate(features: pd.Series) -> bool
    def fill_missing(features: pd.Series) -> pd.Series    # impute with safe defaults
    def clip_outliers(features: pd.Series) -> pd.Series   # cap extreme values
```

---

### 5.3 Models Layer (`src/models/`)

#### `BaseFraudModel` — `base_model.py` *(Abstract)*
Defines the interface all models must implement. Enforces consistency across LightGBM, LSTM, and any future models.

```python
from abc import ABC, abstractmethod

class BaseFraudModel(ABC):
    model_name: str
    is_trained: bool

    @abstractmethod
    def train(X_train, y_train) -> None

    @abstractmethod
    def predict_score(features: pd.Series) -> float   # returns 0.0–1.0

    @abstractmethod
    def save(path: str) -> None

    @abstractmethod
    def load(path: str) -> None

    def is_ready() -> bool
```

#### `LightGBMModel` — `lightgbm_model.py`
Primary fast-path classifier. Extends `BaseFraudModel`.

```python
class LightGBMModel(BaseFraudModel):
    model_name = "lightgbm"

    # Attributes
    model: lgb.Booster
    feature_names: list[str]
    scale_pos_weight: float     # set to neg/pos ratio (~28 for IEEE-CIS)
    threshold_approve: float    # default 0.2
    threshold_block: float      # default 0.6

    # Methods
    def train(X_train, y_train, X_val=None, y_val=None) -> None
    def predict_score(features: pd.Series) -> float      # ~10ms
    def predict_batch(X: pd.DataFrame) -> np.ndarray     # for evaluation
    def get_feature_importance() -> pd.DataFrame
    def save(path: str) -> None          # saves as .pkl
    def load(path: str) -> None
    def evaluate(X_val, y_val) -> dict   # returns PR-AUC, F1, FPR metrics
```

#### `LSTMModel` — `lstm_model.py` *(Stretch)*
Sequential model for capturing patterns across transaction history. Extends `BaseFraudModel`.

```python
class LSTMModel(BaseFraudModel):
    model_name = "lstm"

    # Attributes
    sequence_length: int = 5       # last N transactions
    input_size: int = 5            # features per timestep
    hidden_size: int = 32
    model: nn.Module               # PyTorch model

    # Methods
    def build_sequences(profile: UserProfile) -> torch.Tensor
    def train(sequences, labels) -> None
    def predict_score(profile: UserProfile) -> float
    def save(path: str) -> None    # saves as .pt
    def load(path: str) -> None
```

#### `FusionModel` — `fusion_model.py`
Combines LightGBM and LSTM outputs into a single risk score. Extends `BaseFraudModel`.

```python
class FusionModel(BaseFraudModel):
    model_name = "fusion"

    # Attributes
    lgbm_model: LightGBMModel
    lstm_model: LSTMModel | None   # None if LSTM not trained yet
    alpha: float = 0.65            # weight for LightGBM score

    # Methods
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
Maps a risk score to a 3-tier decision. Responsible for threshold logic and false-positive control.

```python
class DecisionEngine:
    # Attributes
    threshold_approve: float = 0.2    # below this → APPROVE
    threshold_block: float = 0.6      # above this → BLOCK
                                      # between → FLAG

    # Methods
    def decide(risk_score: float) -> str    # returns "APPROVE" | "FLAG" | "BLOCK"
    def set_thresholds(approve: float, block: float) -> None
    def tune_thresholds(y_true, y_scores, target_fpr=0.02) -> tuple
    # Returns tuned (approve_threshold, block_threshold)
```

#### `ExplanationService` — `explanation_service.py`
Generates human-readable reason codes from SHAP feature attributions.

```python
class ExplanationService:
    # Attributes
    explainer: shap.TreeExplainer   # built from LightGBMModel
    top_n: int = 3                  # number of reasons to return
    reason_templates: dict          # feature → plain-language template

    # Methods
    def explain(features: pd.Series) -> list[str]
    # Returns top-3 plain-language strings, e.g.:
    # ["Amount is 4× your average", "New city detected", "3:24 AM — unusual hour"]

    def _get_shap_values(features: pd.Series) -> np.ndarray
    def _top_features(shap_values: np.ndarray) -> list[str]
    def _format_reason(feature_name: str, feature_value, shap_value: float) -> str

    # Reason templates:
    # velocity_ratio      → "You are spending {X}× faster than usual"
    # implied_speed_kmh   → "Transaction from {city_B} — you were in {city_A} {N} min ago"
    # hour_of_day         → "Transaction at {time} — you usually transact {start}–{end}"
    # amount_ratio        → "Amount is {X}× your average transaction"
    # is_new_device       → "Transaction from a device you haven't used before"
    # is_impossible_travel→ "Physically impossible location change detected"
```

#### `PrivacyGuard` — `privacy_guard.py`
Handles PII masking and ethical data handling. Added to satisfy the Case Study 2 privacy-first constraint.

```python
class PrivacyGuard:
    ALLOWED_LOG_FIELDS: set = {"transaction_id", "decision", "risk_score", "timestamp"}
    # user_id, ip_address, device_id are NEVER logged in plain text

    # Methods
    def hash_user_id(user_id: str) -> str         # SHA-256 one-way hash
    def mask_ip(ip_address: str) -> str            # returns "x.x.x.XXX"
    def mask_device_id(device_id: str) -> str      # returns first 8 chars + "****"
    def sanitize_for_log(result: ScoringResult) -> dict   # strips PII from log entry
    def validate_input_fields(transaction: Transaction) -> bool
    # Rejects requests containing raw PAN, CVV, or password fields
```

#### `DataProcessor` — `data_processor.py`
Handles dataset loading, cleaning, train/val splitting, and class imbalance handling.

```python
class DataProcessor:
    # Attributes
    raw_data_path: str
    processed_data_path: str

    # Methods
    def load_ieee_cis(transaction_path: str, identity_path: str) -> pd.DataFrame
    def merge_tables(transactions: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame
    def clean(df: pd.DataFrame) -> pd.DataFrame        # handle nulls, types, duplicates
    def time_based_split(df, train_ratio=0.8) -> tuple # NEVER random split
    def apply_smote(X_train, y_train) -> tuple         # optional oversampling
    def save_processed(df: pd.DataFrame, path: str) -> None
```

#### `RiskScoringService` — `risk_scoring_service.py`
Orchestrator. Ties all components together. The API controller calls only this class.

```python
class RiskScoringService:
    # Attributes
    privacy_guard: PrivacyGuard
    feature_engineer: FeatureEngineer
    feature_validator: FeatureValidator
    fusion_model: FusionModel
    decision_engine: DecisionEngine
    explanation_service: ExplanationService
    user_profiles: dict[str, UserProfile]  # in-memory store (demo purposes)

    # Methods
    def score(transaction: Transaction) -> ScoringResult
    # Full pipeline:
    # 1. privacy_guard.hash_user_id + mask_ip
    # 2. Load or create UserProfile for this user
    # 3. feature_engineer.build_features(transaction, profile)
    # 4. feature_validator.validate + fill_missing
    # 5. fusion_model.predict_score(features, profile)
    # 6. decision_engine.decide(score)
    # 7. explanation_service.explain(features)
    # 8. profile.update(transaction)
    # 9. Return ScoringResult

    def get_user_profile(user_id: str) -> UserProfile
    def health_check() -> dict    # returns model status + latency stats
```

---

### 5.5 API Layer (`src/api/`)

#### `FraudAPIController` — `routes.py`
Thin FastAPI router. Contains no business logic — only HTTP handling and serialization.

```python
# FastAPI router (not a class, but a module using APIRouter)

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

#### `schemas.py`
Pydantic models for request/response validation. Auto-generates Swagger UI docs.

```python
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
    decision: str        # "APPROVE" | "FLAG" | "BLOCK"
    risk_score: float
    reasons: list[str]
    latency_ms: float
```

---

### 5.6 Utilities (`src/utils/`)

| File | Class / Purpose |
|---|---|
| `config.py` | `Config` — loads env vars and model paths from a `.env` file |
| `logger.py` | `FraudLogger` — structured JSON logging, PII-safe output |
| `metrics.py` | `ModelEvaluator` — PR-AUC, F1, FPR, confusion matrix, latency timer |
| `serializer.py` | `ModelSerializer` — save/load `.pkl` and `.pt` files consistently |

---

## 6. Folder Structure

```
fraud-shield/
│
├── data/
│   ├── raw/                        # Downloaded CSVs — gitignored
│   │   ├── train_transaction.csv
│   │   ├── train_identity.csv
│   │   └── test_transaction.csv
│   └── processed/
│       ├── features_train.csv
│       └── features_val.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                # Explore IEEE-CIS distributions
│   ├── 02_features.ipynb           # Build + validate feature pipeline
│   ├── 03_lightgbm.ipynb           # Train, tune, evaluate LightGBM
│   ├── 04_lstm.ipynb               # [STRETCH] LSTM sequential model
│   └── 05_evaluation.ipynb         # Ablation study + final metrics
│
├── src/
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── transaction.py          # Transaction class
│   │   ├── user_profile.py         # UserProfile class
│   │   └── scoring_result.py       # ScoringResult class
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engineer.py     # FeatureEngineer class
│   │   ├── geo_velocity.py         # GeoVelocityCalculator class
│   │   └── validators.py           # FeatureValidator class
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py           # BaseFraudModel (abstract)
│   │   ├── lightgbm_model.py       # LightGBMModel class
│   │   ├── lstm_model.py           # LSTMModel class [STRETCH]
│   │   └── fusion_model.py         # FusionModel class
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── risk_scoring_service.py # RiskScoringService (orchestrator)
│   │   ├── decision_engine.py      # DecisionEngine class
│   │   ├── explanation_service.py  # ExplanationService class
│   │   ├── privacy_guard.py        # PrivacyGuard class
│   │   └── data_processor.py       # DataProcessor class
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py               # FastAPI router + endpoints
│   │   ├── schemas.py              # Pydantic request/response schemas
│   │   └── main.py                 # App factory + startup
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Config class
│       ├── logger.py               # FraudLogger class
│       ├── metrics.py              # ModelEvaluator class
│       └── serializer.py           # ModelSerializer class
│
├── models/
│   ├── lightgbm_model.pkl          # Trained LightGBM (gitignored if large)
│   └── lstm_model.pt               # Trained LSTM weights [STRETCH]
│
├── tests/
│   ├── test_feature_engineer.py
│   ├── test_decision_engine.py
│   ├── test_privacy_guard.py
│   └── test_api.py
│
├── .env                            # Model paths, thresholds — NOT committed
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 7. 10-Day Timeline

> **Guiding rule:** Ship Version 1 (LightGBM + features + SHAP + FastAPI) first.
> Only attempt LSTM after V1 is fully working and tested.

### Day-by-Day Plan

| Day | Phase | Tasks | Output |
|---|---|---|---|
| **Day 1** | Setup & Planning | Set up repo, folder structure, install dependencies, create class skeletons with `pass`, download IEEE-CIS dataset from Kaggle | Empty class files, working imports, data downloaded |
| **Day 2** | EDA & Data Understanding | Run `01_eda.ipynb` — explore fraud rate (3.5%), missing values, feature distributions, join transaction + identity tables | EDA notebook with key findings, data quality notes |
| **Day 3** | Data Pipeline | Build `DataProcessor` — clean, merge, time-based train/val split; build `Transaction` and `UserProfile` entities; build `FeatureValidator` | `features_train.csv`, `features_val.csv` |
| **Day 4** | Feature Engineering | Build `FeatureEngineer` and `GeoVelocityCalculator` — implement all velocity, geo-velocity, behavioural, and contextual features | Feature matrix with ~40–60 engineered features |
| **Day 5** | LightGBM Model | Build `LightGBMModel` — train baseline, add engineered features, tune `scale_pos_weight`, evaluate PR-AUC / F1 / FPR on validation set | Trained `.pkl`, metrics table |
| **Day 6** | Decision Engine + Privacy | Build `DecisionEngine` — tune Approve/Flag/Block thresholds; build `PrivacyGuard` — hash user IDs, mask IPs, PII-safe logging | Decision layer working end-to-end, privacy controls in place |
| **Day 7** | SHAP Explainability | Build `ExplanationService` — TreeExplainer on LightGBM, extract top-3 features, map to plain-language templates; test with sample transactions | Reason codes working for all sample personas |
| **Day 8** | FastAPI Risk API | Build `RiskScoringService` orchestrator; build `routes.py` + `schemas.py`; wire up full pipeline; test with `curl` and Swagger UI | Live `/score` and `/health` endpoints |
| **Day 9** | Testing + Polish | Write key tests in `tests/`; run full latency measurement; produce ablation study table (baseline vs +features vs +LSTM); polish API responses | Test suite passing, latency confirmed <200ms |
| **Day 10** | Stretch + Demo Prep | [STRETCH] Attempt LSTM or Streamlit dashboard; prepare slides with persona story, architecture, results, and live API demo | Final demo-ready submission |

### What to Cut If Behind Schedule

1. Drop Streamlit dashboard — demo via Swagger UI instead (still looks professional)
2. Skip LSTM — LightGBM + engineered features + SHAP is already a complete, strong submission
3. Reduce to velocity features only — skip geo-velocity if location data proves messy
4. Skip PyOD anomaly baseline — it is a sanity check, not a core deliverable

### What to NEVER Cut

- ✅ SHAP explanations — this is the differentiator judges will remember
- ✅ Time-based train/val split — random split inflates metrics and looks amateur
- ✅ Privacy handling — explicitly required by Case Study 2
- ✅ False positive rate target <2% — explicitly required by Case Study 2
- ✅ Persona narrative (Azri) — makes the presentation human and memorable

---

## 8. Deliverables

### Deliverable 1 — Fraud Detection Engine *(Case Study 2 Required)*
- [ ] Trained `LightGBMModel` saved as `.pkl` with reproducible training script
- [ ] Feature engineering pipeline (`FeatureEngineer`) producing ~40–60 features
- [ ] Ablation study table: Baseline → +Velocity → +Geo-velocity → +SHAP → +LSTM
- [ ] Evaluation report: PR-AUC, F1, Recall, FPR, Inference Latency

### Deliverable 2 — Risk API Prototype *(Case Study 2 Required)*
- [ ] FastAPI application running on `localhost:8000`
- [ ] `POST /score` — accepts `TransactionSchema`, returns `ScoringResultSchema`
- [ ] `GET /health` — confirms model loaded and latency stats
- [ ] Swagger UI auto-generated at `localhost:8000/docs` for live demo
- [ ] Sample request/response JSON documented in README

### Deliverable 3 — Demo & Presentation *(Hackathon Required)*
- [ ] Live API demo scoring at least 3 personas (Approve, Flag, Block scenarios)
- [ ] Architecture diagram (ASCII or drawn)
- [ ] Persona-driven narrative (Azri's story)
- [ ] Ablation study results table in slides
- [ ] Privacy & latency budget slide

### Stretch Deliverables *(Do only if V1 is complete)*
- [ ] LSTM sequential model + fusion scoring
- [ ] Streamlit dashboard with live scoring and user profile view
- [ ] Imbalanced handling comparison: `scale_pos_weight` vs SMOTE vs focal loss

---

## 9. Evaluation Metrics & Targets

> **Never use plain accuracy** — it is meaningless for 3.5% fraud rate data.

| Metric | Target | Why |
|---|---|---|
| **Precision-Recall AUC** | ≥ 0.75 | Primary metric — balances catching fraud vs blocking real users |
| **Recall @ Precision=0.5** | ≥ 0.85 | At 50% precision, catch 85%+ of all fraud |
| **F1 Score** | ≥ 0.70 | Harmonic mean of precision and recall |
| **False Positive Rate** | < 2% | Case Study 2 explicit constraint |
| **Inference Latency** | < 200ms total | Case Study 2 explicit constraint |

### Ablation Study Template

| Configuration | PR-AUC | F1 | FPR | Latency |
|---|---|---|---|---|
| Baseline (raw features, LightGBM) | TBD | TBD | TBD | TBD |
| + Velocity features | TBD | TBD | TBD | TBD |
| + Geo-velocity features | TBD | TBD | TBD | TBD |
| + SHAP tuned thresholds | TBD | TBD | TBD | TBD |
| + LSTM fusion [STRETCH] | TBD | TBD | TBD | TBD |

---

## 10. Privacy & Data Integrity

This section directly addresses the **Privacy-First** constraint in Case Study 2.

| Requirement | Implementation |
|---|---|
| No raw PII in logs | `PrivacyGuard.sanitize_for_log()` strips user_id, IP, device_id from all log entries |
| Hashed user identifiers | `PrivacyGuard.hash_user_id()` uses SHA-256 one-way hash; original ID never stored |
| Masked IP addresses | `PrivacyGuard.mask_ip()` returns `x.x.x.XXX` format |
| No PAN / CVV in inputs | `PrivacyGuard.validate_input_fields()` rejects requests containing card numbers |
| Data minimisation | Only the ~40–60 engineered features enter the model; raw 434-column dataset never touches the API |
| Demo data only for presentation | All demo transactions use synthetic data or anonymised samples |
| Encrypted transit (production note) | FastAPI would be deployed behind HTTPS in production; for demo, localhost is acceptable |

---

## 11. Scope Control — What to Cut

### Version 1 (Must Ship by Day 8)
```
DataProcessor → FeatureEngineer → LightGBMModel → DecisionEngine → ExplanationService → FastAPI
```
This is a complete, judge-ready submission on its own.

### Version 2 (Day 9–10, stretch only)
```
LSTMModel → FusionModel → Streamlit Dashboard
```
Only attempt if Version 1 is fully working and tested.

### Risk Matrix

| Risk | Likelihood | Mitigation |
|---|---|---|
| IEEE-CIS download is slow | Medium | Use Credit Card Fraud (smaller) as fallback |
| LSTM training exceeds time budget | High | Skip LSTM; LightGBM alone is sufficient |
| Geo-velocity needs clean location data | Medium | IEEE-CIS has `dist` features as proxy; skip lat/lon if unavailable |
| SHAP takes >200ms on large inputs | Low | Use background thread; return decision immediately, explanation async |
| FastAPI routing errors | Low | Test with `curl` early on Day 8 before full demo integration |

---

*Plan version: 1.0 | Generated: March 2026 | Varsity Hackathon 2026*
