"""
data_loader.py
--------------
Loads final.csv — already clean and pre-encoded, no preprocessing needed.
"""
import pandas as pd

FEATURE_COLS = [
    'year_sold', 'property_tax', 'insurance', 'beds', 'baths',
    'sqft', 'year_built', 'lot_size', 'basement', 'popular',
    'recession', 'property_age', 'property_type_Condo'
]
TARGET_COL = 'price'


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame):
    x = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    return x, y
