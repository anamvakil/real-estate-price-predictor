# Real Estate Price Predictor

**CST2216 — Modularising and Deploying ML Code | Algonquin College**

A deployed machine learning web application that predicts property prices using Linear Regression and Random Forest. Built as part of a modular ML deployment project, converted from a Jupyter notebook into a production-ready Streamlit app.

🔗 **Live App:** [real-estate-anam.streamlit.app](https://real-estate-anam.streamlit.app)

---

## Overview

This app compares two regression models on 1,860 property records with 13 numeric features. The interface lets users explore the dataset, train models with adjustable hyperparameters and generate live price predictions — all in the browser.

| Model | Test MAE |
|---|---|
| Linear Regression | ~$88K |
| Random Forest | ~$42K |

Random Forest reduces prediction error by ~52%, capturing non-linear feature interactions that Linear Regression cannot.

---

## Project Structure

```
real-estate-price-predictor/
├── app.py                  # Streamlit entry point
├── requirements.txt        # Dependencies (no pinned versions)
├── data/
│   └── final.csv           # 1,860 property records, 13 features
└── src/
    ├── __init__.py
    ├── data_loader.py      # Loads and validates the dataset
    ├── preprocessor.py     # Feature scaling and encoding
    ├── model.py            # Linear Regression and Random Forest training
    └── utils.py            # Shared helper functions
```

---

## App Features

**Tab 1 — Data Overview**
- Dataset shape, sample rows and summary statistics
- Feature distribution visualisations

**Tab 2 — Train & Evaluate**
- Adjustable hyperparameters (n_estimators for Random Forest)
- One-click training pipeline
- Side-by-side MAE comparison for both models
- Training complete confirmation banner

**Tab 3 — Predict**
- Input sliders and dropdowns for all 13 features
- Instant price prediction from the trained Random Forest model

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Streamlit | UI framework and cloud deployment |
| scikit-learn | Linear Regression and Random Forest |
| pandas / NumPy | Data loading and preprocessing |
| Matplotlib / Seaborn | Visualisations |
| GitHub | Version control and auto-deploy trigger |

---

## Running Locally

```bash
git clone https://github.com/anamvakil/real-estate-price-predictor
cd real-estate-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

> Requires Python 3.9 or higher.

---

## Key Technical Notes

- **No pinned versions** in `requirements.txt` — compatible with Streamlit Cloud's Python 3.14 runtime
- **Pandas 2.x compatible** — all `inplace=True` calls replaced with explicit assignment syntax
- **data/ folder committed to repo** — required for Streamlit Cloud to locate the CSV at runtime
- `plt.close(fig)` called after every `st.pyplot()` call to prevent matplotlib memory accumulation on the cloud

---

## Dataset

`final.csv` — 1,860 rows, 13 numeric features, no missing values. Pre-encoded and ready for modelling. Stratified 80/20 train/test split on `property_type_Condo`.

---

## Results

| Model | Train MAE | Test MAE |
|---|---|---|
| Linear Regression | $86,554 | $87,709 |
| Random Forest | $17,214 | $41,940 |

Random Forest outperforms Linear Regression across both train and test sets, with a ~52% reduction in test MAE.

---

## Author

**Anam Vakil**  
BISI Graduate Certificate — Algonquin College  
[github.com/anamvakil](https://github.com/anamvakil)
