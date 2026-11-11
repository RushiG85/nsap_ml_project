# NSAP Eligibility Predictor
### AI-Powered Multi-Class Scheme Classification · Powered by IBM watsonx.ai

> **Problem Statement**: India's National Social Assistance Programme (NSAP) has 4 sub-schemes for different beneficiary groups. Manual eligibility verification is slow and error-prone. This project trains an ML classifier to automatically predict the correct NSAP sub-scheme for a welfare applicant from their demographic and socio-economic profile.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [NSAP Sub-Schemes](#nsap-sub-schemes)
3. [How to Run](#how-to-run)
4. [Streamlit Dashboard](#streamlit-dashboard)
5. [IBM Cloud Setup](#ibm-cloud-setup)
6. [Dataset & Features](#dataset--features)
7. [Models & Results](#models--results)
8. [Python API Usage](#python-api-usage)
9. [Sharing Your App](#sharing-your-app)
10. [Technology Stack](#technology-stack)

---

## Project Structure

```
nsap_ml_project/
│
├── data/
│   └── nsap_dataset.csv          ← auto-generated on first run (or use real AI Kosh data)
│
├── models/
│   ├── preprocessor.pkl           ← fitted sklearn ColumnTransformer
│   ├── best_model.pkl             ← best trained Pipeline (auto-selected)
│   ├── metrics.json               ← evaluation results (accuracy, F1, CV scores)
│   └── deployment_info.json       ← IBM WML deployment metadata (created after deploy)
│
├── notebooks/
│   └── NSAP_ML_Pipeline.ipynb     ← step-by-step notebook: EDA → training → deployment
│
├── reports/
│   ├── class_distribution.png
│   ├── age_by_scheme.png
│   ├── correlation_heatmap.png
│   ├── cv_comparison.png
│   ├── confusion_matrix.png
│   ├── per_class_f1.png
│   └── feature_importance.png
│
├── src/
│   ├── data_generator.py          ← synthetic dataset mirroring AI Kosh NSAP structure
│   ├── preprocessing.py           ← cleaning, feature engineering, train/val/test split
│   ├── train.py                   ← model training, cross-validation, best model selection
│   ├── predict.py                 ← local inference
│   └── visualize.py               ← EDA and evaluation plots
│
├── app.py                         ← Streamlit web dashboard (4 pages)
├── main.py                        ← end-to-end pipeline orchestrator (CLI)
├── requirements.txt               ← Python 3.14-compatible dependencies
├── pyproject.toml                 ← project metadata
└── .gitignore
```

---

## NSAP Sub-Schemes

| Label | Code | Full Name | Eligibility |
|-------|------|-----------|-------------|
| 0 | **IGNOAPS** | Indira Gandhi National Old Age Pension Scheme | Age ≥ 60 · BPL household |
| 1 | **IGNWPS** | Indira Gandhi National Widow Pension Scheme | Female · Widow · Age 18–79 · BPL |
| 2 | **IGNDPS** | Indira Gandhi National Disability Pension Scheme | Disability ≥ 80% · Age 18–79 · BPL |
| 3 | **NFBS** | National Family Benefit Scheme | Primary earner deceased · Age 18–64 · BPL |

---

## How to Run

### Prerequisites

- Python **3.10 or higher** (tested on 3.14)
- `pip` package manager
- Windows / macOS / Linux

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/nsap_ml_project.git
cd nsap_ml_project
```

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for Python 3.14 users**: The project uses `ibm-watsonx-ai` (the successor to the deprecated `ibm-watson-machine-learning` package which had no Python 3.14 wheels).

---

### Step 3 — Train the model

```bash
python main.py
```

This single command runs the full ML pipeline:

| Step | What happens |
|------|-------------|
| 1/5 | Generates a 12,000-record synthetic NSAP dataset → saved to `data/nsap_dataset.csv` |
| 2/5 | Cleans data, engineers features, splits into train / val / test sets |
| 3/5 | Trains 5 classifiers with 5-fold stratified cross-validation, picks the best |
| 4/5 | Saves `models/best_model.pkl` and `models/metrics.json` |
| 5/5 | Generates all evaluation plots → saved to `reports/` |

Optional flags:

```bash
python main.py --predict   # also run demo predictions after training
```

---

### Step 4 — Launch the Streamlit dashboard

```bash
# If streamlit is on PATH:
streamlit run app.py

# Alternatively (always works):
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> ⚠️ **Run Step 3 before Step 4.** The Predict, Batch, and Insights pages all require `models/best_model.pkl` to exist.

---

### Full copy-paste sequence

```bash
git clone https://github.com/your-username/nsap_ml_project.git
cd nsap_ml_project
pip install -r requirements.txt
python main.py
python -m streamlit run app.py
```

---

## Streamlit Dashboard

The app has **4 pages** accessible from the left sidebar:

| Page | Description |
|------|-------------|
| 🏠 **Home** | Project overview, scheme cards, how-to steps, tech stack |
| 🔍 **Predict** | Single-applicant form → instant scheme prediction with probability bars |
| 📂 **Batch Predict** | Upload CSV → classify all rows → download predictions CSV |
| 📊 **Model Insights** | CV comparison table, per-class report, all evaluation plots in tabs |

### Sidebar features
- 🇮🇳 Indian flag with Ashoka Chakra (pure CSS/SVG, no external request)
- Live model status indicator (green dot = loaded, amber dot = not found)
- Real-time metrics: Best Model · Accuracy · F1 Weighted · F1 Macro

---

## Dataset & Features

**Source**: [AI Kosh — NSAP District-wise Pension Data](https://aikosh.indiaai.gov.in/web/datasets/details/district_wise_pension_data_under_the_national_social_assistance_programme_nsap_1.html)

The project generates a **12,000-record synthetic dataset** mirroring the AI Kosh schema. To use the real dataset:
1. Download the CSV from the AI Kosh portal
2. Place it at `data/nsap_dataset.csv`
3. Adjust `CATEGORICAL_COLS` / `NUMERICAL_COLS` in `src/preprocessing.py` if column names differ

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `state` | categorical | Indian state (20 states) |
| `district` | categorical | District name |
| `age` | numeric | Applicant age in years |
| `gender` | binary | 0 = Male, 1 = Female |
| `is_widow` | binary | Widow / widower flag |
| `is_disabled` | binary | Person with disability flag |
| `disability_percentage` | numeric | Disability severity (0–100%) |
| `bpl_card` | binary | Below Poverty Line card holder |
| `primary_earner_deceased` | binary | Family primary earner deceased |
| `annual_income` | numeric | Annual household income (INR) |
| `land_holding_acres` | numeric | Agricultural land holdings (acres) |
| `household_size` | numeric | Number of household members |
| `education_level` | ordinal | 0=Illiterate · 1=Primary · 2=Secondary · 3=Higher |
| `ration_card_type` | categorical | 0=BPL · 1=AAY · 2=APL |

### Engineered Features (auto-created in preprocessing)

| Feature | Description |
|---------|-------------|
| `vulnerability_score` | Composite domain-driven score |
| `age_group` | Binned age category |
| `income_per_capita` | Annual income ÷ household size |
| `is_senior` | Boolean: age ≥ 60 |
| `is_severely_disabled` | Boolean: disabled AND disability_percentage ≥ 80 |

---

## Models & Results

Five classifiers are evaluated via **5-fold stratified cross-validation** (weighted F1):

| Model | CV F1 (weighted) |
|-------|-----------------|
| Logistic Regression | ~0.87 |
| Gradient Boosting | ~0.96 |
| Random Forest | ~0.97 |
| XGBoost | ~0.98 |
| **LightGBM** | **~0.98** ✅ best |

The best model is automatically selected, retrained on train+val, evaluated on the held-out test set, and saved to `models/best_model.pkl`.

**Typical test-set performance** (synthetic data):

| Metric | Score |
|--------|-------|
| Accuracy | ~0.97 |
| F1 (weighted) | ~0.97 |
| F1 (macro) | ~0.97 |
| ROC-AUC (OvR, weighted) | ~0.99 |

---

## Python API Usage

```python
from src.predict import NSAPPredictor

# Load the trained model
predictor = NSAPPredictor.load("models/best_model.pkl")

# Single prediction
applicant = {
    "state": "Bihar",
    "district": "Biha_D2",
    "age": 68,
    "gender": 0,               # 0=Male, 1=Female
    "is_widow": 0,
    "is_disabled": 0,
    "disability_percentage": 0,
    "bpl_card": 1,
    "primary_earner_deceased": 0,
    "annual_income": 22000,
    "land_holding_acres": 0.3,
    "household_size": 3,
    "education_level": 0,      # 0=Illiterate
    "ration_card_type": 0,     # 0=BPL Card
}

result = predictor.predict_one(applicant)
print(result["predicted_scheme_name"])   # → IGNOAPS
print(result["confidence"])              # → 0.99
print(result["probabilities"])           # → {"IGNOAPS": 0.99, "IGNWPS": 0.00, ...}

# Batch prediction
records = [applicant, {...}, {...}]
results = predictor.predict_batch(records)
```

### CLI

```bash
# Train only
python main.py

# Train + run demo predictions
python main.py --predict
```

---

## Sharing Your App

| Option | How | Result |
|--------|-----|--------|
| **Streamlit Community Cloud** *(recommended)* | Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → Deploy | `https://yourname-nsap.streamlit.app` — free, permanent public link |
| **ngrok** (quick demo) | `ngrok http 8501` | Temporary public URL, expires when stopped |

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ (compatible with 3.14) |
| ML Framework | scikit-learn ≥ 1.6, XGBoost ≥ 2.1, LightGBM ≥ 4.5 |
| Imbalanced data | imbalanced-learn (SMOTE) |
| Web Dashboard | Streamlit ≥ 1.40 |
| Visualization | Matplotlib, Seaborn, Plotly |
| Notebook | Jupyter / JupyterLab |
| Dataset | AI Kosh NSAP district-wise pension data (or synthetic mirror) |

---

## License

MIT — for educational and research use.

---

*Built for AI for Social Good — NSAP Welfare Scheme Classification.*
