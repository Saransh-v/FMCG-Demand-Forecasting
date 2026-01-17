import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def baseline_naive(train_series, horizon):
    """Naive baseline: repeat last observed value."""
    last_value = train_series.iloc[-1]
    return np.array([last_value] * horizon)


def fit_predict_arima(train_series, horizon, order=(1, 1, 1)):
    """Fit ARIMA (SARIMAX) model and forecast."""
    model = SARIMAX(
        train_series,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    forecast = results.forecast(steps=horizon)
    return forecast.values


def compute_metrics(y_true, y_pred):
    """Compute MAE and RMSE (version-safe)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    }
