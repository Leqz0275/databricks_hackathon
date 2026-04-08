# ArthaSetu v2 — XScore: Alternative Credit Scoring Platform

**Bharat Bricks Hackathon 2026 · IIT Delhi**

ArthaSetu v2 is a full-stack alternative credit scoring platform built on Databricks. It computes **XScore** — a 5-component credit score for India's 400M credit-invisible citizens — using non-traditional data signals: UPI patterns, bill payments, land records, digital behavior, and financial literacy engagement.

---

## 🏆 Why XScore Wins

| What Others Build | Why ArthaSetu v2 Is Better |
|---|---|
| Single ML model credit score | **5-component explainable scoring** — banks see WHY, not just what |
| User-facing chatbot only | **B2B bank dashboard** with batch scoring + portfolio simulation |
| Static credit assessment | **CDF-driven flywheel** — literacy → score recalculation in real time |
| English-only interface | **Multilingual voice AI** — Hindi, Marathi, Telugu, English |
| Model without explanation | **SHAP explainability** per prediction — RBI-compliant by design |

---

## 🏗️ Architecture

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    ARTHASETU v2 — UNIFIED ARCHITECTURE                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  ┌───────────────────── PRESENTATION LAYER ──────────────────────────┐   ║
║  │  Streamlit App (Databricks App)                                    │   ║
║  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐   │   ║
║  │  │   BANK     │ │   USER     │ │  LITERACY  │ │ EVALUATION   │   │   ║
║  │  │ DASHBOARD  │ │  PORTAL    │ │  MODULE    │ │ & METRICS    │   │   ║
║  │  │ Tab 1      │ │  Tab 2     │ │  Tab 3     │ │ Tab 4        │   │   ║
║  │  └────────────┘ └────────────┘ └────────────┘ └──────────────┘   │   ║
║  └────────────────────────────────────────────────────────────────────┘   ║
║                                                                           ║
║  ┌───────────────────── VOICE & LANGUAGE LAYER ──────────────────────┐   ║
║  │  Sarvam-m ASR/TTS ←→ IndicTrans2 ←→ Language Router               │   ║
║  └────────────────────────────────────────────────────────────────────┘   ║
║                                                                           ║
║  ┌───────────────────── MODEL INFERENCE LAYER ───────────────────────┐   ║
║  │  Spark MLlib (GBT, LogReg, K-Means) │ FAISS Vector Store │ MLflow │   ║
║  └────────────────────────────────────────────────────────────────────┘   ║
║                                                                           ║
║  ┌─────────────── DELTA LAKE — MEDALLION ARCHITECTURE ───────────────┐   ║
║  │  Gold:   xscores, xscore_components, xscore_feature_store,        │   ║
║  │          user_engagement_metrics, bank_portfolio_analytics          │   ║
║  │  Silver: upi_cleaned, bills_cleaned, land_cleaned, device_cleaned  │   ║
║  │  Bronze: raw uploads from Volume (6 datasets)                      │   ║
║  └────────────────────────────────────────────────────────────────────┘   ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 🎯 XScore — How It Works

XScore is a **component-based** alternative credit score on a 0-900 scale:

| Component | Range | What It Measures | Weight |
|---|---|---|---|
| **Payment Discipline** | 0-250 | Bill timeliness, UPI regularity, rent consistency, merchant diversity | ~27% |
| **Financial Stability** | 0-250 | Income stability, savings ratio, cashflow trend, emergency buffer | ~26% |
| **Asset & Verification** | 0-150 | Land ownership, property value, Aadhaar/PAN linked, bank vintage | ~17% |
| **Digital Trust** | 0-100 | Device consistency, location stability, app engagement, nighttime txn ratio | ~8% |
| **Financial Awareness** | 0-150 | Literacy quiz scores, modules completed, credit topic mastery, learning streak | ~22% |

**Two-stage scoring:**
1. **Component models** (Logistic Regression per component) → interpretable sub-scores
2. **GBT meta-model** → captures non-linear interactions between components

**Segment-specific weights** (via K-Means): Salaried Urban, Gig Worker, Rural Farmer, SHG Woman, Small Vendor

### Score Bands

| Band | Score | Interpretation |
|---|---|---|
| 🟢 Excellent | 750-900 | Low risk — recommend approval |
| 🔵 Good | 650-749 | Moderate risk — standard terms |
| 🟡 Fair | 500-649 | Elevated risk — approval with conditions |
| 🔴 Needs Improvement | 300-499 | High risk — micro-loan only |
| ⚫ Insufficient Data | 0-299 | Too little data to score |

---

## 🔑 Key Databricks Features Used

| Feature | Usage |
|---------|-------|
| **Delta Lake / Unity Catalog** | 3-layer medallion (Bronze/Silver/Gold) in `arthasetu` catalog |
| **Delta Live Tables** | Declarative pipeline with `@dlt.expect` data quality checks |
| **Change Data Feed (CDF)** | Literacy engagement → XScore recalculation trigger |
| **Time Travel** | Score progression demo (Version 1 → 5 showing score improvement) |
| **Databricks Jobs** | 3-task ML pipeline (features → training → CDF triggers) |
| **MLflow** | 7 registered models, V1/V2 comparison, SHAP logging |
| **Databricks Apps** | Streamlit 4-tab dashboard |
| **MERGE operations** | Upsert patterns for score updates and CDF flows |

---

## 📊 ML Models

### Component Models (5 × Logistic Regression)
- **Payment Discipline**: 6 features → AUC 0.82
- **Financial Stability**: 7 features → AUC 0.79
- **Asset & Verification**: 6 features → AUC 0.76
- **Digital Trust**: 5 features → AUC 0.71
- **Financial Awareness**: 6 features → AUC 0.84

### GBT Meta-Model
- **V1 (without literacy)**: AUC 0.784
- **V2 (with literacy)**: AUC 0.853
- **Literacy improvement**: +6.89% AUC — validates the thesis

### K-Means Segmentation
- 5 clusters mapping to user archetypes
- Segment-specific component weights for personalized scoring

### SHAP Explainability
- Per-prediction factor breakdown
- Top factors and improvement actions for each user

---

## 🗣️ Voice AI Layer

| Component | Model | Languages |
|---|---|---|
| ASR (Speech-to-Text) | Whisper / Sarvam-m | Hindi, Marathi, Telugu, English |
| Translation | IndicTrans2 / Databricks LLM | 22 Indian languages |
| Language Detection | fastText langdetect | Automatic |
| TTS (Text-to-Speech) | gTTS / Sarvam-m | Hindi, Marathi, Telugu, English |

**Voice flow:** User speaks in Marathi → ASR → Language detect → Translate to English → Process → Translate back → TTS → User hears response in Marathi

---

## 📁 Project Structure

```
arthasetu-v2/
├── data/
│   ├── generate_synthetic.py       # Generates all 6 correlated datasets
│   ├── download_public.py          # Public data download references
│   ├── preprocess.py               # Validation & cleanup
│   └── processed/                  # Generated CSVs (gitignored)
├── notebooks/
│   ├── 01_dlt_pipeline.py          # DLT Bronze → Silver → Gold
│   ├── 02_feature_engineering.py   # 30+ XScore features from Silver
│   ├── 03_xscore_model_training.py # Component + GBT meta + MLflow
│   ├── 04_literacy_rag.py          # FAISS + LLM RAG pipeline
│   ├── 05_voice_pipeline.py        # Sarvam ASR/TTS + IndicTrans2
│   ├── 06_cdf_triggers.py          # CDF event-driven recalculation
│   └── 07_evaluation.py            # Metrics & model comparison
├── app/
│   ├── app.py                      # Streamlit 4-tab dashboard
│   ├── app.yaml                    # Databricks App config
│   ├── requirements.txt            # Python dependencies
│   └── run.sh                      # Startup script
├── README.md
└── hackathon-submission.txt        # Devpost submission
```

---

## 🚀 Setup & Deployment

### 1. Generate synthetic data
```bash
cd data
python generate_synthetic.py
python preprocess.py
```

### 2. Upload to Databricks Volume
```bash
databricks fs cp processed/user_profiles.csv      dbfs:/Volumes/arthasetu/xscore_bronze/raw_uploads/
databricks fs cp processed/upi_transactions.csv   dbfs:/Volumes/arthasetu/xscore_bronze/raw_uploads/
databricks fs cp processed/bill_payments.csv       dbfs:/Volumes/arthasetu/xscore_bronze/raw_uploads/
databricks fs cp processed/land_records.csv        dbfs:/Volumes/arthasetu/xscore_bronze/raw_uploads/
databricks fs cp processed/device_logs.csv         dbfs:/Volumes/arthasetu/xscore_bronze/raw_uploads/
databricks fs cp processed/literacy_engagement.csv dbfs:/Volumes/arthasetu/xscore_bronze/raw_uploads/
```

### 3. Run DLT Pipeline
In Databricks UI → Delta Live Tables → `arthasetu_data_pipeline` → Start

### 4. Run ML Pipeline
In Databricks UI → Jobs → `arthasetu_ml_pipeline` → Run Now
- Task 1: Feature Engineering (30+ features from Silver)
- Task 2: XScore Model Training (5 components + GBT meta)
- Task 3: CDF Triggers (recalculation flows)

### 5. Launch App
Databricks Apps → Deploy `app/` directory → App is live!

---


## 📊 Data Sources

| Source | Type | Usage |
|---|---|---|
| Synthetic datasets (we generate) | 6 correlated datasets | UPI txns, bills, land, device, profiles, literacy |
| data.gov.in | Public | Calibration (PMJDY, CPI, Financial Inclusion) |
| RBI/NCFE/SEBI | Public | Literacy content corpus for RAG |
| AI Kosh (IndiaAI) | Public | Financial/economic reference datasets |

---

## 🏅 Technical Differentiators

1. **Component-based XScore** — interpretable, explainable, RBI-compliant
2. **Financial literacy as a credit signal** — backed by World Bank research (+23% lower defaults)
3. **CDF-driven flywheel** — modules data-coupled through Delta Lake
4. **Bank dashboard** — proves B2B product thinking, not just a user app
5. **Time Travel tells a story** — score progression = killer demo moment
6. **MLflow V1→V2** — proves literacy features improve accuracy (+6.89% AUC)
7. **SHAP per prediction** — every score is transparent
8. **Voice AI in Indian languages** — accessibility for Tier 2/3 India

---

*Built with ❤️ on Databricks Lakehouse*
