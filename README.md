# BuyerSense

A machine learning project that predicts whether an online shopper will complete a purchase, based on their on-site browsing behavior.

## Overview

Most e-commerce recommendation systems focus on suggesting products to customers. BuyerSense takes a different angle: given a visitor's session behavior (pages viewed, time spent, bounce/exit rates, etc.), it predicts the *probability that the visitor will make a purchase* during that session.

This repository covers the full pipeline from raw data to a trained, evaluated model.

## Dataset

- **Source:** [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Size:** 12,330 sessions, 18 original features
- **Target variable:** `Revenue` (boolean) — whether the session ended in a purchase

### Features include:
- Page-level engagement: `Administrative`, `Informational`, `ProductRelated` (counts and durations)
- Behavioral signals: `BounceRates`, `ExitRates`, `PageValues`
- Contextual info: `Month`, `SpecialDay`, `Weekend`, `VisitorType`
- Technical/traffic info: `OperatingSystems`, `Browser`, `Region`, `TrafficType`

## Exploratory Data Analysis (EDA)

Key findings from the EDA phase:

- **No missing values**, but 125 duplicate rows were found and removed.
- **Significant class imbalance**: ~84.5% of sessions did not result in a purchase, ~15.5% did. This meant accuracy alone would not be a reliable evaluation metric.
- **`PageValues` was the strongest predictor** of purchase intent (correlation ≈ 0.49 with the target).
- **`BounceRates` and `ExitRates` were highly correlated** with each other (≈ 0.90), indicating multicollinearity.
- **Seasonality mattered**: purchase rates were noticeably higher in November and May.
- **New visitors converted at a higher rate** than returning visitors.

## Feature Engineering

Based on the EDA findings, the following steps were applied:

1. **Outlier handling (IQR method)** — Duration-based features (e.g. `ProductRelated_Duration`) had extreme outliers. Bounds were computed as `Q1 - 1.5×IQR` and `Q3 + 1.5×IQR`. Outliers tied to the minority (purchasing) class were checked before removal, to avoid discarding valuable signal.
2. **Categorical encoding** — `Month` and `VisitorType` have no inherent order, so one-hot encoding was used instead of arbitrary numeric labels.
3. **Dimensionality reduction (PCA)** — Highly correlated features (e.g. `BounceRates`/`ExitRates`, `ProductRelated`/`ProductRelated_Duration`) were combined via PCA to reduce redundancy while retaining signal.
4. **Class imbalance handling (SMOTE)** — Applied only to the training set (never to the test set) to generate synthetic samples of the minority (purchasing) class, giving the model a fairer chance to learn purchase patterns.

## Model Training

- **Algorithm:** XGBoost classifier (compared against a Logistic Regression baseline)
- **Train/test split:** 80/20, stratified on the target to preserve class ratio
- **Imbalance handling:** SMOTE on training data (with `scale_pos_weight` also evaluated as an alternative)
- The trained model was serialized and saved as `model.pkl` for reuse.

## Model Evaluation

The model was evaluated on the held-out test set using:

- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- ROC-AUC

**Note:** Given the class imbalance, accuracy alone is not a meaningful metric. Recall (catching actual purchasers) and precision (avoiding false alarms) were weighted more heavily during evaluation, since missing a genuine high-intent customer is more costly than a false positive.

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine learning | scikit-learn, XGBoost |
| Imbalance handling | imbalanced-learn (SMOTE) |

## Project Status

This repository currently covers data exploration, feature engineering, model training, and evaluation. Deployment, real-time inference, and dashboarding are not yet part of this codebase.
