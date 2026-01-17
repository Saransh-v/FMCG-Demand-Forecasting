from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class Metrics:
    mae: float
    rmse: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    return Metrics(mae=mae, rmse=rmse)


def fit_predict_arima(y_train: pd.Series, steps: int) -> np.ndarray:
    """Simple SARIMAX with weekly seasonality (s=7)."""
    # Guard: if series is too short, fallback to last value
    if len(y_train) < 30:
        return np.repeat(y_train.iloc[-1], steps)

    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=steps)
    return np.asarray(fc)


def fit_predict_rf(supervised: pd.DataFrame, test_size: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, RandomForestRegressor, list]:
    """Train RandomForest on supervised features; predict last test_size rows."""
    feature_cols = [
        'price_mean', 'promo_sum', 'supplier_cost_mean', 'lead_time_mean', 'stock_mean',
        'dow', 'month', 'year',
        'lag_1', 'lag_7', 'lag_14',
        'roll_mean_7', 'roll_std_7',
    ]
    feature_cols = [c for c in feature_cols if c in supervised.columns]

    X = supervised[feature_cols].values
    y = supervised['y'].values

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred, model, feature_cols


def baseline_naive(y_train: pd.Series, steps: int) -> np.ndarray:
    return np.repeat(float(y_train.iloc[-1]), steps)
