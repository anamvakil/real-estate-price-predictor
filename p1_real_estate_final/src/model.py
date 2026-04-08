"""
model.py
--------
Trains LinearRegression and RandomForestRegressor exactly as in the notebook.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def split_data(x, y):
    # Notebook uses stratify=x.property_type_Condo
    return train_test_split(x, y, test_size=0.2, stratify=x['property_type_Condo'])


def train_linear_regression(x_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, n_estimators: int = 200) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=n_estimators, criterion='absolute_error')
    model.fit(x_train, y_train)
    return model


def evaluate(model, x_test, y_test) -> dict:
    ypred = model.predict(x_test)
    mae = mean_absolute_error(ypred, y_test)
    train_pred = model.predict(x_test)
    return {
        "mae": mae,
        "predictions": ypred,
    }
