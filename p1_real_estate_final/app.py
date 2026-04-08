"""
app.py
------
Streamlit app — Real Estate Price Prediction
Tabs: Data Overview | Train & Evaluate | Predict
Dataset loads automatically from data/final.csv
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data, split_features_target, FEATURE_COLS
from src.model import split_data, train_linear_regression, train_random_forest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "final.csv")

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon=None,
    layout="wide",
)

st.title("Real Estate Price Prediction")
st.caption("CST2216 — Modularizing and Deploying ML Code | Algonquin College")
st.markdown("---")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return load_data(DATA_PATH)

try:
    raw_df = load_raw()
except FileNotFoundError:
    st.error("Dataset not found at data/final.csv. Please ensure the file is committed to the repository.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Data Overview", "Train & Evaluate", "Predict"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Data Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Dataset Overview")
    st.write("Real estate dataset with 1,860 properties and 13 pre-encoded features.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", raw_df.shape[0])
    col2.metric("Features", raw_df.shape[1] - 1)
    col3.metric("Missing Values", int(raw_df.isnull().sum().sum()))

    st.subheader("Sample Records")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(raw_df.describe(), use_container_width=True)

    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(raw_df['price'], bins=40, color='steelblue', edgecolor='white')
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Property Prices")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Price vs Square Footage")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(raw_df['sqft'], raw_df['price'], alpha=0.3, color='steelblue', s=10)
    ax2.set_xlabel("Square Footage")
    ax2.set_ylabel("Price ($)")
    ax2.set_title("Price vs Sqft")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Train & Evaluate
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Train & Evaluate Models")

    with st.expander("⚙️ Hyperparameter Settings", expanded=True):
        n_estimators = st.slider("Random Forest — n_estimators", 50, 300, 200, step=50)

    if st.button("Run Training Pipeline"):
        try:
            with st.spinner("Splitting data..."):
                x, y = split_features_target(raw_df)
                x_train, x_test, y_train, y_test = split_data(x, y)

            with st.spinner("Training Linear Regression..."):
                lr_model = train_linear_regression(x_train, y_train)
                lr_train_mae = mean_absolute_error(lr_model.predict(x_train), y_train)
                lr_test_mae = mean_absolute_error(lr_model.predict(x_test), y_test)

            with st.spinner(f"Training Random Forest (n={n_estimators})..."):
                rf_model = train_random_forest(x_train, y_train, n_estimators=n_estimators)
                rf_train_mae = mean_absolute_error(rf_model.predict(x_train), y_train)
                rf_test_mae = mean_absolute_error(rf_model.predict(x_test), y_test)

            st.success("Training complete!")

            # ── Results table
            st.subheader("Model Comparison")
            results = pd.DataFrame([
                {"Model": "Linear Regression", "Train MAE": f"${lr_train_mae:,.0f}", "Test MAE": f"${lr_test_mae:,.0f}"},
                {"Model": "Random Forest", "Train MAE": f"${rf_train_mae:,.0f}", "Test MAE": f"${rf_test_mae:,.0f}"},
            ])
            st.table(results)

            # ── Actual vs Predicted chart
            st.subheader("Actual vs Predicted Prices — Random Forest")
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            rf_preds = rf_model.predict(x_test)
            ax3.scatter(y_test, rf_preds, alpha=0.4, color='steelblue', s=15)
            min_val = min(y_test.min(), rf_preds.min())
            max_val = max(y_test.max(), rf_preds.max())
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect Prediction')
            ax3.set_xlabel("Actual Price ($)")
            ax3.set_ylabel("Predicted Price ($)")
            ax3.set_title("Actual vs Predicted (Random Forest)")
            ax3.legend()
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

            # ── Feature importance
            st.subheader("Feature Importances — Random Forest")
            fi = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            fi.plot.barh(ax=ax4, color='steelblue', edgecolor='white')
            ax4.set_title("Feature Importances")
            ax4.set_xlabel("Importance")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

            # Save to session state
            st.session_state["re_state"] = {
                "lr_model": lr_model,
                "rf_model": rf_model,
                "feature_names": FEATURE_COLS,
            }

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            logger.exception("Tab2 error")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Predict
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Predict Price for a New Property")

    if "re_state" not in st.session_state:
        st.info("Please train the models first in the Train & Evaluate tab.")
    else:
        rs = st.session_state["re_state"]

        model_choice = st.selectbox("Select Model", ["Random Forest", "Linear Regression"])

        col1, col2 = st.columns(2)
        with col1:
            year_sold    = st.number_input("Year Sold", 2000, 2024, 2013)
            property_tax = st.number_input("Property Tax ($/yr)", 0, 50000, 5000, step=100)
            insurance    = st.number_input("Insurance ($/yr)", 0, 10000, 1000, step=50)
            beds         = st.number_input("Bedrooms", 1, 10, 3)
            baths        = st.number_input("Bathrooms", 1, 10, 2)
            sqft         = st.number_input("Square Footage", 200, 10000, 1800, step=50)
            year_built   = st.number_input("Year Built", 1900, 2024, 2000)

        with col2:
            lot_size     = st.number_input("Lot Size (sqft)", 0, 100000, 6000, step=100)
            basement     = st.selectbox("Basement", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            popular      = st.selectbox("Popular Area", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            recession    = st.selectbox("Sold During Recession", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            property_age = st.number_input("Property Age (years)", 0, 150, 10)
            property_type_condo = st.selectbox("Property Type", [0, 1],
                                               format_func=lambda x: "Condo" if x == 1 else "House")

        if st.button("Predict Price", type="primary"):
            try:
                input_data = pd.DataFrame([{
                    'year_sold': year_sold,
                    'property_tax': property_tax,
                    'insurance': insurance,
                    'beds': beds,
                    'baths': baths,
                    'sqft': sqft,
                    'year_built': year_built,
                    'lot_size': lot_size,
                    'basement': basement,
                    'popular': popular,
                    'recession': recession,
                    'property_age': property_age,
                    'property_type_Condo': property_type_condo,
                }])

                model = rs["rf_model"] if model_choice == "Random Forest" else rs["lr_model"]
                prediction = model.predict(input_data)[0]

                st.markdown("---")
                st.success(f"### Predicted Price: **${prediction:,.0f}**")
                st.caption(f"Model used: {model_choice}")

            except Exception as e:
                st.error(f"Prediction error: {e}")
