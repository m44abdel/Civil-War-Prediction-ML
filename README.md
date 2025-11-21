# Civil-War-Prediction-ML

A predictive modeling project that uses machine learning to forecast civil war onset based on political regime characteristics and historical conflict data.

**Authors:** Moheb Abdelmasih, Rayan Ahmad, Connor Hughes

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

This project develops a machine learning model to predict the onset of civil wars using historical political and conflict data. By analyzing regime characteristics, institutional stability, and political transitions across 150+ countries over multiple decades, we identify key risk factors that precede civil conflict.

### Motivation

Civil wars are among the most devastating forms of conflict, causing immense human suffering and economic destruction. Early prediction of civil war risk can enable:
- Preventive diplomatic interventions
- Resource allocation for conflict prevention
- Policy decisions based on data-driven risk assessment
- Academic research on state fragility and conflict dynamics

---

## Key Features

- **Comprehensive Data Integration**: Combines UCDP/PRIO Armed Conflict Dataset with Polity5 political regime data
- **Advanced Feature Engineering**: 40+ engineered features including lagged variables, regime change indicators, and volatility metrics
- **Class Imbalance Handling**: Implements SMOTE (Synthetic Minority Over-sampling Technique) to address rare event prediction
- **Multiple Model Comparison**: Evaluates Logistic Regression, Random Forest, and Gradient Boosting algorithms
- **Threshold Optimization**: Tunes decision boundaries for optimal precision-recall tradeoff
- **Interpretable Results**: Provides coefficient analysis to identify key risk factors

---

## Dataset

### Data Sources

1. **UCDP/PRIO Armed Conflict Dataset (v25.1)**
   - Source: Uppsala Conflict Data Program
   - Coverage: 2,752 conflict observations
   - Time Period: 1946-2024
   - Variables: Conflict type, intensity, location, parties involved

2. **Polity5 Dataset (2018)**
   - Source: Center for Systemic Peace
   - Coverage: 17,574 country-year observations
   - Variables: Democracy scores, autocracy scores, regime characteristics, institutional measures

3. **Economic Indicators (GEM Data)**
   - GDP, unemployment, inflation, trade data
   - Exchange rates and foreign reserves
   - Industrial production indices

### Target Variable

- **Civil War Onset**: Binary indicator for the first year of internal armed conflict (UCDP conflict types 3 & 4)
- **Class Distribution**: Highly imbalanced (~200:1 ratio of non-onset to onset years)

---

## Methodology

### 1. Data Preprocessing

- Merged conflict data with political regime indicators by country-year
- Handled missing values using forward/backward fill within countries
- Converted country codes to ensure consistent joining
- Created binary onset indicator for civil war initiation

### 2. Feature Engineering

**Political Regime Features:**
- Democracy and autocracy scores (polity2, democ, autoc)
- Executive constraints (xconst, exrec)
- Political competition measures (parcomp, xrcomp)
- Regime durability and stability

**Temporal Features:**
- Lagged variables (1, 2, and 3 years)
- Regime change indicators (magnitude and direction)
- Rolling volatility measures (3-year and 5-year windows)
- Historical conflict indicators

**Derived Features:**
- Anocracy indicators (partial democracies)
- New regime flags (durability ≤ 2 years)
- Democratization vs. autocratization trends
- Interaction terms (e.g., anocracy × new regime)

### 3. Model Development

**Baseline Model:**
- Logistic Regression with balanced class weights
- ROC-AUC: 0.7036
- F1-Score: 0.0403

**Enhanced Model:**
- SMOTE oversampling + random undersampling
- 40+ engineered features
- Hyperparameter tuning with GridSearchCV
- ROC-AUC: 0.6811
- F1-Score: 0.1022 (with optimized threshold)

**Ensemble Models:**
- Random Forest (200 trees): ROC-AUC 0.6766
- Gradient Boosting: ROC-AUC 0.6433

### 4. Evaluation Metrics

- **ROC-AUC**: Primary metric for imbalanced classification
- **F1-Score**: Balance between precision and recall
- **Precision-Recall Curves**: Threshold optimization
- **5-Fold Cross-Validation**: Model robustness assessment
- **Confusion Matrix**: Error analysis

---

## Results

### Model Performance Summary

| Model | Accuracy | ROC-AUC | F1-Score | Notes |
|-------|----------|---------|----------|-------|
| Baseline Logistic Regression | 71.3% | 0.704 | 0.040 | With balanced class weights |
| Enhanced LR (SMOTE) | 81.0% | 0.681 | 0.040 | Better feature set |
| Enhanced LR (Optimal Threshold) | 96.3% | 0.681 | 0.102 | Tuned for F1-score |
| Random Forest | 98.0% | 0.677 | 0.057 | Ensemble method |
| Gradient Boosting | 98.5% | 0.643 | 0.077 | Ensemble method |

### Performance Improvements

- **F1-Score**: 154% improvement (0.040 → 0.102) through threshold optimization
- **Recall**: Achieved 19.4% recall at optimal threshold, detecting ~1 in 5 actual civil war onsets
- **Cross-Validation**: Mean ROC-AUC of 0.71 with ±0.051 standard deviation

### Feature Importance

**Top 5 Risk Factors (Logistic Regression Coefficients):**

1. **Executive Recruitment (exrec)**: +1.57 — How chief executives are selected
2. **Political Competition (parcomp)**: +0.89 — Extent of political participation
3. **Regime Durability (durable)**: -0.76 — Longer-lasting regimes = lower risk
4. **Executive Constraints (xconst)**: +0.45 — Checks on executive power
5. **Anocracy × New Regime**: +1.23 — Partial democracies with recent transitions

---

## Installation

### Prerequisites

- Python 3.10+

### Setup
**Create a virtual environment (recommended):**
python -m venv civilwarvenv
source civilwarvenv/bin/activate  # On Windows: civilwarvenv\Scripts\activate3. **Install required packages:**
pip install -r requirements.txt### Required Packages
