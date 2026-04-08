# Project 1 — Real Estate Price Predictor

**Course:** CST2216 — Modularizing and Deploying ML Code  
**Student:** Anam Vakil

## Overview
Predicts property prices using Linear Regression (MAE ~$85k) and Random Forest (MAE ~$47k) trained on 1,860 pre-encoded real estate records.

## App Design
- **Data Overview** — dataset stats, price distribution, price vs sqft scatter
- **Train & Evaluate** — trains both models, shows MAE comparison, actual vs predicted chart, feature importances
- **Predict** — enter property details and get a price prediction

## Project Structure
```
p1_real_estate/
├── app.py
├── src/
│   ├── data_loader.py
│   └── model.py
├── data/
│   └── final.csv        ← commit this file
├── requirements.txt
└── README.md
```

## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Cloud
1. Push repo to GitHub — make sure `data/final.csv` is included
2. Go to share.streamlit.io → New app → point to `app.py`
3. No secrets needed
