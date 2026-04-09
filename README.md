# Real Estate Price Predictor — Linear Regression & Random Forest

Machine Learning | Algonquin College

## Overview
Supervised machine learning app that predicts residential property prices using Linear Regression and Random Forest regression models trained on 1,860 pre-encoded property records.

## Live Demo
[https://real-estate-anam.streamlit.app/]

## Dataset
`final.csv` — 1,860 rows, 13 pre-encoded numeric features, no missing values, no categorical columns requiring encoding.

## Features
- **Tab 1 — Data Overview:** Dataset shape, sample records, summary statistics, feature distributions
- **Tab 2 — Train & Evaluate:** Trains both models side by side, displays MAE comparison, feature importance chart
- **Tab 3 — Predict:** Enter property details and get an instant price prediction from both models

## Models
- Linear Regression — baseline model
- Random Forest Regressor — main model, achieves ~46% lower MAE than baseline
- Split: 80/20 stratified on property_type_Condo column
- Metric: Mean Absolute Error (MAE)

## Results
| Model | MAE |
|---|---|
| Linear Regression | ~$79,000 |
| Random Forest | ~$43,000 |

