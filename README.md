---
title: Procurement Lead Time Variance Predictor
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Procurement Lead Time Variance — ML System

An end-to-end machine learning system to predict delivery variance for Olist e-commerce orders.

## Project Overview

Predicts how many days early or late an order will be delivered relative to the estimated delivery date using the Brazilian Olist e-commerce dataset.

## Model

- **Algorithm:** XGBoost (Optuna-tuned, 50 trials)
- **Test MAE:** 4.58 days
- **Test RMSE:** 7.41 days
- **Directional Accuracy:** 91.67%
- **Prediction Intervals:** Conformal prediction (90% coverage)

## Features Used

44 engineered features including:
- Estimated delivery window (strongest predictor)
- Seller historical performance
- Geographic features (same state / region)
- Seasonal features (purchase month, day of week)
- Product dimensions and weight
- Payment features

## App Structure

- **User Tab** — Single order predictor with confidence intervals
- **Developer Tab** — MLflow experiment tracking + Evidently AI drift monitoring + Model performance

## Tech Stack

- Python, XGBoost, LightGBM, Scikit-learn
- MLflow (experiment tracking)
- Evidently AI (drift monitoring)
- SHAP (model interpretation)
- Streamlit (frontend)
- Docker (containerization)
- GitHub Actions (CI/CD)
- Hugging Face Spaces (deployment)

## Pipeline Phases

1. Problem Definition
2. Data Collection (Supabase + SQLAlchemy)
3. Target Construction + Train/Val/Test Split
4. EDA (train split only — no leakage)
5. Data Cleaning
6. Feature Engineering
7. Model Training (Baseline → Linear → RF → XGBoost → LightGBM)
8. Hyperparameter Tuning (Optuna)
9. Model Validation + Conformal Prediction Intervals
10. Model Interpretation (SHAP)
11. Deployment (Streamlit + Docker + GitHub Actions + Hugging Face)
12. Monitoring (Evidently AI)