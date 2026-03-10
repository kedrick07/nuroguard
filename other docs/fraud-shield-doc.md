# Digital Trust — Real-Time Fraud Shield for the Unbanked

> **Varsity Hackathon 2026 — Case Study 2**
> Track: Machine Learning (Fraud & Anomaly Detection)
> SDG 8: Decent Work and Economic Growth (Target 8.10)

---

## 1. Project Overview

### 1.1 What We Are Building

A fraud detection system that protects unbanked and low-literacy digital wallet users in ASEAN from fraudulent transactions. The system will:

1. **Learn normal user behaviour** over time (behavioural profiling).
2. **Score every incoming transaction** in real-time with an Approve / Flag / Block decision.
3. **Explain the decision** in plain, human-readable language so users understand *why* a transaction was blocked.

### 1.2 Our Core Innovation (What Makes Us Different)

Most teams will train XGBoost on a flat Kaggle CSV, get an AUC number, and wrap it in an API. We go further with three deliberate design choices:

| Innovation | What it means | Why it matters |
|---|---|---|
| **Sequential behaviour modelling** | We model each user's *transaction history as a time series* using a lightweight LSTM, not just the current transaction in isolation. | Catches slow-burn fraud (e.g., a compromised account that probes with small amounts before one big theft). A single-transaction model misses this pattern entirely. |
| **SHAP-powered reason codes** | Every scored transaction comes with a plain-language explanation generated from SHAP feature attributions. | An unbanked gig worker can't call a bank helpline — they need to understand an SMS saying *"Blocked: 4× your average spend, new city, 3:24 AM"*. |
| **Geo-velocity contextual feature** | We engineer an "impossible travel" feature that calculates the physical speed implied between consecutive transaction locations. | If a user transacts in KL and then in Bangkok 20 minutes later, that is physically impossible and strongly signals credential theft. |

These are not pie-in-the-sky ideas — each one has published research backing and is implementable within a hackathon timeframe.

### 1.3 Why This Problem Matters Right Now

- Malaysia recorded **67,735 online fraud cases** from January to November 2025, with losses exceeding **RM 2.7 billion**.
- In the first three months of 2025 alone, Malaysia saw **12,110 scam cases** and **RM 573.7 million** in losses.
- Only **RM 34 million** was recovered from mule accounts in 2025 — a tiny fraction of what was stolen.
- Across ASEAN, **84% of consumers** say they are "very worried" about scams, and **63% experienced scams** in the past year.
- Fraudulent QR code payments rose by **60%** in Thailand and Vietnam in 2024–2025.
- ASEAN's digital payment transaction volume is expected to hit **$417 billion by 2028**, expanding the attack surface.

The people who suffer most are not tech-savvy bank customers — they are gig workers, rural merchants, and migrants whose entire income sits in a digital wallet. One fraudulent transaction can wipe out a week's earnings.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   INCOMING TRANSACTION                    │
│  { user_id, amount, merchant, location, device, time }   │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   FEATURE ENGINEERING       │
         │                             │
         │  • Velocity features        │
         │  • Geo-velocity score       │
         │  • Behavioural aggregates   │
         │  • Device/IP context        │
         └──────────┬──────────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   ┌─────────────┐    ┌──────────────┐
   │  LightGBM   │    │  LSTM        │
   │  (fast path │    │  (sequence   │
   │   ~10ms)    │    │   model)     │
   └──────┬──────┘    └──────┬───────┘
          │                  │
          └───────┬──────────┘
                  ▼
        ┌──────────────────┐
        │  FUSION LAYER    │
        │  Weighted avg /  │
        │  stacking logic  │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  DECISION ENGINE │
        │  p < 0.2 → ✅    │
        │  0.2–0.6 → ⚠️    │
        │  p ≥ 0.6 → 🚫    │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  SHAP EXPLAINER  │
        │  Top-3 reasons   │
        │  in plain text   │
        └────────┬─────────┘
                 │
                 ▼
   ┌───────────────────────────────┐
   │  FASTAPI RISK API RESPONSE   │
   │                               │
   │  {                            │
   │    "decision": "BLOCK",       │
   │    "risk_score": 0.91,        │
   │    "reasons": [               │
   │      "4x above avg spend",    │
   │      "New city (Bangkok)",    │
   │      "3:24 AM local time"     │
   │    ]                          │
   │  }                            │
   └───────────────────────────────┘
```

---

## 3. Technical Approach (Step by Step)

### 3.1 Data

We will use **one primary dataset** and optionally a secondary dataset:

| Dataset | Size | Features | Purpose |
|---|---|---|---|
| **IEEE-CIS Fraud Detection** (Kaggle) | ~590K transactions (3.5% fraud) | 434 features: transaction amount, time delta, card info, address, device identity, Vesta risk features | Primary training and evaluation dataset. Rich enough for behavioural profiling and identity context. |
| **PaySim** (Kaggle) | ~6.3M transactions | 9 features: type, amount, balances, sender/receiver IDs, isFraud | Optional secondary dataset for stress-testing on mobile money / e-wallet scenarios closer to the ASEAN context. |

**Why IEEE-CIS?** It contains both transaction and identity tables joined by `TransactionID`. This gives us device fingerprint data, email domains, and distance features that most simpler datasets lack — exactly what we need for contextual data integration.

### 3.2 Feature Engineering

This is where most of the differentiation happens. We engineer four categories of features:

#### A. Velocity Features (Transaction Speed)
- `txn_count_1h`: Number of transactions by this user in the last 1 hour.
- `txn_count_24h`: Number of transactions by this user in the last 24 hours.
- `amount_sum_1h`: Total spend in the last 1 hour.
- `velocity_ratio`: `txn_count_1h / avg_hourly_txn_30d` — how many times faster than usual.
- `amount_ratio`: `amount / avg_transaction_amount_30d` — how large compared to usual.

#### B. Geo-Velocity (Impossible Travel Detection)
- From consecutive transactions, compute:
  - `distance_km`: Great-circle distance between locations.
  - `time_delta_hours`: Time between transactions.
  - `implied_speed_kmh`: `distance_km / time_delta_hours`.
- If `implied_speed_kmh > 900` (faster than a commercial flight), flag as impossible travel.
- Additional binary flag: `is_new_country` — has this user ever transacted from this country before?

#### C. Behavioural Aggregates (User Profiling)
- `avg_txn_amount_7d`, `std_txn_amount_7d`: Mean and variance of recent transaction amounts.
- `most_frequent_merchant_category`: Does this transaction match their usual spending pattern?
- `hour_of_day`, `day_of_week`: Temporal pattern (a 3 AM transaction from someone who only transacts 9–6 is suspicious).
- `device_change_flag`: Is this a new device/browser for this user?

#### D. Contextual / Identity Features
- `email_domain_risk`: Free email vs corporate email vs disposable email.
- `card_type_mismatch`: Does the card type match the user's profile?
- `addr_distance`: Distance between billing address and transaction address (from IEEE-CIS `dist` features).

### 3.3 Model 1 — LightGBM (Fast Path)

**Purpose:** Score every transaction within ~10ms for the real-time latency requirement.

- **Input:** All engineered features above (velocity, geo-velocity, behavioural, contextual) — approximately 40–60 final features after selection.
- **Algorithm:** LightGBM classifier.
- **Imbalanced handling:** `scale_pos_weight` parameter set to the ratio of negatives to positives (~28:1 for IEEE-CIS). If this is insufficient, apply SMOTE on the training set only and compare.
- **Evaluation:** Precision-Recall AUC (more informative than ROC-AUC for imbalanced data), plus F1 at the chosen operating threshold.
- **Why LightGBM:** Published benchmarks show LightGBM is the most advantageous gradient boosting model for latency control in fraud detection, enabling sub-50ms inference.

### 3.4 Model 2 — LSTM Sequential Model (Deep Path)

**Purpose:** Capture sequential patterns across a user's recent transaction history that a single-transaction model cannot see.

- **Input:** For each transaction, extract the user's **last N transactions** (we start with N=5, tune up to 10) as a sequence.
- Each step in the sequence includes: `[amount, time_since_last, merchant_category, is_domestic, hour_of_day]`.
- **Architecture:**
  ```
  Input (N × 5) → LSTM(32 units) → Dropout(0.3) → Dense(16, ReLU) → Dense(1, Sigmoid)
  ```
- **Training:** Binary cross-entropy loss with class weights. We keep this lightweight — no attention layers unless we have time.
- **Why this helps:** Published research shows LSTM provides better detection for in-person/card-present transactions by learning sequential degradation patterns. A combined LightGBM + LSTM approach achieves F1 of 87% and AUC of 96% on European credit card data, outperforming either model alone.

### 3.5 Fusion Layer

We combine the two models with a simple weighted average:

```
final_score = α × lightgbm_score + (1 - α) × lstm_score
```

We determine `α` via a small validation set (likely `α ≈ 0.6–0.7` favouring LightGBM for speed and calibration). If time permits, we replace this with a logistic regression stacking layer trained on the two models' outputs.

**Fallback (if LSTM is not ready in time):** The system works perfectly fine with LightGBM alone. The LSTM is an enhancement, not a dependency.

### 3.6 Decision Engine

Map `final_score` to actions using two thresholds tuned on the validation set:

| Risk Score | Decision | Action |
|---|---|---|
| < 0.2 | ✅ **APPROVE** | Transaction goes through silently. |
| 0.2 – 0.6 | ⚠️ **FLAG** | Transaction goes through but user receives a notification to verify. |
| ≥ 0.6 | 🚫 **BLOCK** | Transaction is held; user must verify via OTP or in-app confirmation. |

We tune these thresholds to target:
- **Recall ≥ 85%** — catch most fraud.
- **False Positive Rate < 2%** — don't annoy legitimate users.

### 3.7 SHAP Explainability Layer

For every transaction scored, we generate a human-readable explanation:

1. Use **SHAP TreeExplainer** on the LightGBM model (runs in ~15–50ms for tree-based models).
2. Extract the **top 3 features** by absolute SHAP value.
3. Map each feature to a **plain-language template:**

| Feature | Template |
|---|---|
| `velocity_ratio` is high | "You are spending {X}× faster than usual" |
| `implied_speed_kmh` > 900 | "Transaction from {city_B} — you were in {city_A} {N} minutes ago" |
| `hour_of_day` outside norm | "Transaction at {time} — you usually transact between {start}–{end}" |
| `amount_ratio` is high | "Amount is {X}× your average transaction" |
| `is_new_device` = True | "Transaction from a device you haven't used before" |

**Why this matters:** In regulated financial services, SHAP explanations are increasingly required for audit and compliance. For unbanked users, a clear reason turns a frustrating block into an understandable safety measure.

**Latency note:** SHAP TreeExplainer for LightGBM runs in 15–50ms, which is acceptable within our ~200ms total budget. If needed, we can precompute explanations in a background thread while the decision is returned immediately.

---

## 4. The Persona — Who We Protect

We ground every design decision in a real persona:

> **Azri, 24, GrabFood rider in Petaling Jaya, Malaysia**
> - Earns ~RM 1,800/month, all through TNG eWallet.
> - No bank account — his entire income sits in a digital wallet.
> - Low digital literacy — he doesn't know what "two-factor authentication" means.
> - A single fraudulent withdrawal of RM 500 means he can't pay rent this week.

Every technical choice traces back to Azri:

| Technical Choice | How It Helps Azri |
|---|---|
| Low false positives | His real orders don't get blocked |
| Plain-language reason codes | He understands *why* a payment was held |
| Geo-velocity detection | If someone in another state uses his credentials, it's caught |
| Low latency scoring | His GrabFood customer isn't kept waiting |

---

## 5. Tech Stack

| Component | Tool | Why |
|---|---|---|
| Data processing | pandas, numpy | Standard, fast, you already know them |
| Feature engineering | pandas + custom functions | Rolling windows, velocity calculations |
| Primary model | LightGBM | Fast inference (~10ms), handles tabular data well, native categorical support |
| Sequential model | PyTorch (LSTM) | Lightweight, flexible, you already have PyTorch experience from FYP |
| Imbalanced handling | `scale_pos_weight`, SMOTE (imblearn) | Two approaches to compare |
| Anomaly baseline | PyOD (Isolation Forest) | Optional unsupervised sanity-check layer |
| Explainability | SHAP (TreeExplainer) | Fast for tree models, theoretically grounded |
| API | FastAPI | Async, auto-docs via Swagger, easy to demo |
| Dashboard (optional) | Streamlit | Quick interactive demo for presentation |
| Experiment tracking | Notebooks with clear sections | Keep it simple for a hackathon |

---

## 6. Evaluation Strategy

### 6.1 Metrics

We do NOT rely on accuracy (useless for imbalanced data). Our metrics:

| Metric | Target | Why |
|---|---|---|
| **Precision-Recall AUC** | ≥ 0.75 | Primary metric — tells us how well we balance catching fraud vs not blocking real users |
| **Recall @ Precision=0.5** | ≥ 0.85 | At 50% precision, we want to catch 85%+ of all fraud |
| **F1 Score** | ≥ 0.70 | Harmonic mean of precision and recall |
| **False Positive Rate** | < 2% | Azri's real transactions should go through |
| **Inference Latency** | < 200ms end-to-end | Real-time constraint |

### 6.2 Experiment Comparisons

We will present results for at least these configurations to show we didn't just pick one model blindly:

1. **Baseline:** LightGBM with raw features only (no velocity, no sequence).
2. **+ Velocity features:** LightGBM with velocity and geo-velocity features.
3. **+ LSTM fusion:** LightGBM + LSTM combined.
4. **Imbalanced handling comparison:** `scale_pos_weight` vs SMOTE vs focal loss.

This ablation study demonstrates that each component adds value — judges appreciate this rigour.

### 6.3 Validation Approach

- **Time-based split** (not random): Train on earlier transactions, validate on later ones. This simulates real deployment where you never see future data.
- This is critical and often overlooked — random splits leak future information and inflate metrics.

---

## 7. Expected Deliverables

### Deliverable 1: Fraud Detection Engine
- A trained LightGBM model (+ optional LSTM) with documented performance metrics.
- Feature engineering pipeline as clean, reproducible Python scripts.
- Comparison table showing baseline vs enhanced model performance.

### Deliverable 2: Risk API Prototype
- FastAPI application with endpoints:
  - `POST /score` — accepts transaction JSON, returns `{ decision, risk_score, reasons[] }`.
  - `GET /health` — API health check.
- Swagger UI auto-generated for live demo.

### Deliverable 3: Demo & Presentation
- Live API demo scoring sample transactions.
- Persona-driven narrative (Azri's story).
- Architecture diagram.
- Ablation study results table.

---

## 8. Project Timeline (Hackathon Sprint)

Assuming a 2–3 day hackathon window:

| Phase | Duration | Tasks |
|---|---|---|
| **Setup & EDA** | 3–4 hours | Download IEEE-CIS dataset, explore distributions, understand feature types, identify missing data patterns |
| **Feature Engineering** | 4–6 hours | Build velocity features, geo-velocity calculator, behavioural aggregates, handle missing values |
| **LightGBM Baseline** | 3–4 hours | Train baseline → train with engineered features → tune hyperparameters → evaluate with PR-AUC |
| **LSTM Sequential Model** | 4–6 hours | Build user transaction sequences, train small LSTM, evaluate standalone and fused |
| **SHAP Integration** | 2–3 hours | Add TreeExplainer, build reason code templates, test on sample transactions |
| **FastAPI Risk API** | 2–3 hours | Build `/score` endpoint, load model, add SHAP explanations to response, test with sample requests |
| **Dashboard (stretch)** | 2 hours | Streamlit page showing live scoring, explanation display, user profile view |
| **Slides & Demo** | 2–3 hours | Build presentation with persona story, architecture, results, live API demo |

**Total: ~22–30 hours of focused work.**

### What to cut if running out of time (priority order):
1. Drop the Streamlit dashboard (demo via Swagger instead).
2. Simplify LSTM to just LightGBM alone (still a strong submission).
3. Reduce feature engineering to velocity features only (skip geo-velocity).

### What NOT to cut:
- SHAP explanations — this is your differentiator.
- The persona narrative — this is what makes judges remember you.
- Time-based validation split — this is what makes your results credible.

---

## 9. Folder Structure

```
fraud-shield/
├── data/
│   ├── raw/                  # Downloaded CSVs (gitignored)
│   └── processed/            # Engineered feature CSVs
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_features.ipynb     # Feature engineering pipeline
│   ├── 03_lightgbm.ipynb     # LightGBM training & evaluation
│   ├── 04_lstm.ipynb         # LSTM sequential model
│   └── 05_evaluation.ipynb   # Ablation study & final metrics
├── src/
│   ├── features.py           # Feature engineering functions
│   ├── model.py              # Model loading & prediction
│   ├── explainer.py          # SHAP reason code generator
│   └── api.py                # FastAPI application
├── models/
│   ├── lightgbm_model.pkl    # Saved LightGBM model
│   └── lstm_model.pt         # Saved LSTM weights
├── requirements.txt
└── README.md
```

---

## 10. Key References

1. **IEEE-CIS Fraud Detection Dataset** — https://www.kaggle.com/c/ieee-fraud-detection
2. **PaySim Synthetic Dataset** — https://www.kaggle.com/datasets/ealaxi/paysim1
3. **LightGBM + LSTM for Sequential Fraud Detection** — Elsevier Expert Systems with Applications, Feb 2025. Distribution-preserving resampling combined with LightGBM-LSTM achieving F1=87%, AUC=96%.
4. **SHAP for Model Explainability** — Lundberg & Lee, NeurIPS 2017. TreeExplainer runs in 15–50ms for tree-based models.
5. **Group SHAP for Financial Fraud** — ScienceDirect, 2022. Demonstrates group-level feature attribution reducing SHAP computation time.
6. **Geo-Velocity Fraud Detection** — Verisoul, 2026. Impossible travel detection flagging physically impossible location changes.
7. **ASEAN Consumer Scam Report 2025** — GSMA, Sep 2025. 84% of ASEAN consumers worried about scams; 8% scammed in the past year.
8. **Malaysia Scam Statistics** — Malay Mail, Dec 2025. 67,735 cases, RM 2.7 billion losses Jan–Nov 2025.
9. **PyOD: A Python Toolbox for Scalable Outlier Detection** — JMLR, 2019. 20+ unsupervised anomaly detection algorithms.
10. **Velocity Features for Fraud Detection** — AI Infrastructure Alliance, 2023. Feature engineering patterns for real-time fraud systems.

---

## 11. How This Connects to Your FYP (NuroQuant)

This hackathon project deliberately shares skill overlap with your FYP:

| Skill | Hackathon (Fraud Shield) | FYP (NuroQuant) |
|---|---|---|
| Time-series feature engineering | Velocity features, rolling windows | Technical indicators, lag features |
| LightGBM / XGBoost training | Fraud classifier | Return prediction model |
| LSTM modelling | User transaction sequences | Financial time-series forecasting |
| Model evaluation | PR-AUC, ablation study | Walk-forward validation, IC |
| FastAPI deployment | Risk scoring API | (Future) signal serving API |
| SHAP explainability | Reason codes for users | Feature importance for thesis |
| pandas data pipelines | Transaction cleaning | OHLCV + news data processing |

**Nothing here is wasted effort.** Everything you build or learn transfers directly.

---

*Last updated: March 2026*
