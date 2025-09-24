
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH_DEFAULT = os.path.join('data', 'sample_noise.csv')
MODEL_PATH = os.path.join('models', 'noise_rf.joblib')

@st.cache_data(show_spinner=False)
def load_data(path: str = None) -> pd.DataFrame:
    path = path or DATA_PATH_DEFAULT
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce').dt.tz_convert(None)
    df = df.dropna(subset=['timestamp', 'lat', 'lon', 'laeq_db'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    return df

def _build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("zone", OneHotEncoder(handle_unknown="ignore"), ["zone_id"]),
            ("num", "passthrough", ["flights_count", "heavy_aircraft_ratio", "altitude_mean_m", "hour", "dow", "is_weekend"]),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=160, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    return Pipeline(steps=[("pre", pre), ("rf", rf)])

@st.cache_resource(show_spinner=False)
def train_or_load_model(df: pd.DataFrame):
    # Try to load packaged model first
    try:
        payload = joblib.load(MODEL_PATH)
        model = payload.get('model')
        metrics = payload.get('metrics', {})
        return model, metrics
    except Exception:
        pass

    # Train on provided data
    features = ["zone_id", "flights_count", "heavy_aircraft_ratio", "altitude_mean_m", "hour", "dow", "is_weekend"]
    target = "laeq_db"
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred))
    }
    return pipe, metrics

def predict_interval(model, X_df, mae):
    preds = model.predict(X_df)
    # Symmetric interval around prediction using MAE as proxy of expected absolute deviation
    lower = preds - mae
    upper = preds + mae
    return preds, lower, upper

def aggregate_by_zone(df, start=None, end=None):
    if start is not None and end is not None:
        m = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        df = df.loc[m]
    agg = df.groupby(['zone_id', 'zone_name', 'lat', 'lon'], as_index=False).agg(
        laeq_mean=('laeq_db', 'mean'),
        laeq_max=('laeq_db', 'max'),
        count=('laeq_db', 'size')
    )
    return agg

def color_for_db(db):
    # Approximate color scale (green -> yellow -> red)
    if db < 50:
        return [34, 197, 94, 180]   # green
    elif db < 65:
        return [234, 179, 8, 200]   # yellow
    else:
        return [239, 68, 68, 220]   # red
