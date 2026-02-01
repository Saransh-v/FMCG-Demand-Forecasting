import numpy as np
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _resolve_steps(steps=None, horizon=None):
    if steps is None and horizon is None:
        raise TypeError("Missing required argument: 'steps' (or 'horizon').")
    return int(steps if steps is not None else horizon)


@dataclass
class Metrics:
    mae: float
    rmse: float


def compute_metrics(y_true, y_pred) -> Metrics:
    """MAE + RMSE (version-safe)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    return Metrics(mae=mae, rmse=rmse)


def baseline_naive(train_series, steps=None, horizon=None, **kwargs):
    """Naive baseline: repeat last observed value."""
    n = _resolve_steps(steps=steps, horizon=horizon)
    last_value = float(train_series.iloc[-1])
    return np.array([last_value] * n, dtype=float)


def fit_predict_arima(train_series, steps=None, horizon=None, order=(1, 1, 1), **kwargs):
    """ARIMA via SARIMAX, forecast next n steps."""
    n = _resolve_steps(steps=steps, horizon=horizon)
    model = SARIMAX(
        train_series,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=n)
    return fc.to_numpy(dtype=float)


def fit_predict_rf(supervised_df, test_size=30, random_state=42):
    """
    Train RF on supervised features and predict last test_size rows.
    Returns: y_true, y_pred, feature_cols, model
    """
    feature_cols = [
        'price_mean','promo_sum','supplier_cost_mean','lead_time_mean','stock_mean',
        'dow','month','year','lag_1','lag_7','lag_14','roll_mean_7','roll_std_7'
    ]
    feature_cols = [c for c in feature_cols if c in supervised_df.columns]

    df = supervised_df.dropna(subset=feature_cols + ['y']).copy()
    if len(df) <= test_size + 10:
        test_size = max(5, min(test_size, len(df)//3))

    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]

    X_train = train_df[feature_cols].values
    y_train = train_df['y'].values
    X_test = test_df[feature_cols].values
    y_true = test_df['y'].values

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_true, y_pred, feature_cols, model
