# AlphaCare Insurance Risk Analytics & Predictive Modeling 🚗 📉

## Overview

**AlphaCare Insurance Solutions (ACIS)** is transforming car insurance in South Africa by leveraging data analytics to optimize marketing strategies and identify low-risk client segments.

This project implements an end-to-end data pipeline to analyze historical insurance claim data, aiming to:

* **Analyze Risk:** Understand differences in risk across provinces, vehicle types, and client demographics.
* **Predict Claims:** Build machine learning models to predict claim severity (`TotalClaims`) and optimal premiums.
* **Optimize Strategy:** Provide actionable insights to refine marketing and pricing strategies.

## Repository Structure

The project follows a modular, production-ready structure designed for reproducibility and scalability.

```
alpha-care-insurance/
├── .github/workflows/       # CI/CD pipelines (Linting & Testing)
├── .dvc/                    # Data Version Control configuration
├── data/                    # Data storage (tracked by DVC, ignored by Git)
│   └── Raw/                 # Raw input data
├── notebooks/               # Jupyter notebooks for exploration and prototyping
├── reports/                 # Generated analysis reports and figures
│   └── figures/             # Visualizations (Loss Ratio, Trends, SHAP plots)
├── src/                     # Source code for the pipeline
│   ├── stats/               # Statistical & Modeling modules
│   │   ├── __init__.py
│   │   ├── distributions.py  # Distribution analysis tools
│   │   ├── hypothesis.py     # A/B testing & statistical tests
│   │   └── modeling.py       # ML model training & evaluation
│   ├── data_loader.py       # Robust data ingestion and cleaning
│   ├── eda_utils.py         # Exploratory Data Analysis utilities
│   ├── main_eda.py          # EDA pipeline orchestrator
│   ├── run_hypothesis.py    # Hypothesis testing orchestrator
│   └── run_modeling.py      # Modeling pipeline orchestrator
├── tests/                   # Unit tests for code integrity
├── dvc.yaml                 # DVC pipeline definitions (DAG)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Getting Started

### Prerequisites

* Python 3.10+
* Git
* DVC (Data Version Control)

### Installation

Clone the repository:

```bash
git clone https://github.com/YourUsername/alpha-care-insurance.git
cd alpha-care-insurance
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Pull the data (using DVC):

```bash
dvc pull
```

## Key Features & Pipeline Stages

The pipeline is automated using **DVC**.

### 1. Exploratory Data Analysis (EDA)

* **Goal:** Understand dataset structure, quality, and key trends.
* **Command:** `dvc repro eda_pipeline`
* **Outputs:**

  * Loss Ratio by Province
  * Monthly Profitability Trends
  * Claim Severity by Vehicle Type
  * Missing Value Analysis

### 2. Statistical Hypothesis Testing

* **Goal:** Validate risk assumptions using statistical tests (Chi-Square, ANOVA, T-Tests).
* **Command:** `dvc repro hypothesis_testing`
* **Key Tests:**

  * Risk differences across Provinces (Null Hypothesis Rejected)
  * Risk differences between Genders (Null Hypothesis Rejected)
  * Margin differences across ZipCodes

### 3. Predictive Modeling

* **Goal:** Predict `TotalClaims` (Severity) and Optimal Premium.
* **Command:** `dvc repro modeling`
* **Models:**

  * Linear Regression: Baseline model for interpretability
  * Random Forest: Ensemble method for non-linearities
  * XGBoost: Gradient boosting for high performance
* **Interpretability:** SHAP is used to explain model decisions globally and locally

## Technologies Used

* **Data Processing:** pandas, numpy
* **Visualization:** matplotlib, seaborn
* **Machine Learning:** scikit-learn, xgboost
* **Interpretability:** shap, lime
* **Versioning & Orchestration:** dvc, git
* **CI/CD:** GitHub Actions

## Results Snapshot

* **High-Risk Provinces:** Gauteng shows a significantly higher loss ratio compared to others
* **Profitability:** A sharp decline in early 2015, highlighting potential issues in claims processing or external factors
* **Model Performance:** XGBoost outperformed Linear Regression in predicting claim severity with an R² of 0.XX (See reports)

## Contributors

* **Yonatan** – Lead Data Analyst / ML Engineer

This project is part of the **10 Academy Artificial Intelligence Mastery** program.
