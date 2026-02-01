import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.data_prep import load_data, make_daily_segment, make_supervised_features
from src.models import baseline_naive, fit_predict_arima, compute_metrics
from sklearn.ensemble import RandomForestRegressor

# Better default styling for charts (no extra dependency needed)
plt.style.use("seaborn-v0_8-darkgrid")


@st.cache_data
def get_raw(path: str) -> pd.DataFrame:
    return load_data(path)


def _metric_value(metrics_obj, key: str, default=np.nan):
    """
    compute_metrics might return:
    - an object with attributes (mae/rmse)
    - a dict-like {"MAE": ..., "RMSE": ...}
    This helper safely extracts values.
    """
    if metrics_obj is None:
        return default

    # attribute style: m.mae, m.rmse
    if hasattr(metrics_obj, key.lower()):
        return float(getattr(metrics_obj, key.lower()))

    # dict style: {"MAE": ..., "RMSE": ...} or {"mae": ...}
    if isinstance(metrics_obj, dict):
        if key in metrics_obj:
            return float(metrics_obj[key])
        if key.lower() in metrics_obj:
            return float(metrics_obj[key.lower()])
        if key.upper() in metrics_obj:
            return float(metrics_obj[key.upper()])

    return default


def rf_recursive_forecast(daily: pd.DataFrame, horizon: int, random_state: int = 42):
    """Train RF on supervised features and recursively forecast next horizon days."""
    supervised = make_supervised_features(daily)

    feature_cols = [
        "price_mean", "promo_sum", "supplier_cost_mean", "lead_time_mean", "stock_mean",
        "dow", "month", "year", "lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_std_7"
    ]
    feature_cols = [c for c in feature_cols if c in supervised.columns]

    X = supervised[feature_cols].values
    y = supervised["y"].values

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X, y)

    # Start from the last available daily history
    hist = daily.copy().sort_values("ds").reset_index(drop=True)
    preds = []

    for _ in range(horizon):
        next_date = hist["ds"].iloc[-1] + pd.Timedelta(days=1)

        # carry forward exogenous values, promo default 0
        row = {
            "ds": next_date,
            "y": np.nan,
            "price_mean": float(hist["price_mean"].iloc[-1]) if "price_mean" in hist.columns else np.nan,
            "promo_sum": 0.0,
            "supplier_cost_mean": float(hist["supplier_cost_mean"].iloc[-1]) if "supplier_cost_mean" in hist.columns else np.nan,
            "lead_time_mean": float(hist["lead_time_mean"].iloc[-1]) if "lead_time_mean" in hist.columns else 7.0,
            "stock_mean": float(hist["stock_mean"].iloc[-1]) if "stock_mean" in hist.columns else 0.0,
            "dow": int(next_date.dayofweek),
            "month": int(next_date.month),
            "year": int(next_date.year),
        }

        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)

        tmp = make_supervised_features(hist)
        last = tmp.iloc[-1]
        X_last = last[feature_cols].values.reshape(1, -1)

        yhat = float(model.predict(X_last)[0])
        preds.append(max(0.0, yhat))  # keep demand non-negative
        hist.loc[hist.index[-1], "y"] = yhat

    future_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    return future_dates, np.array(preds)


def plot_time_series(ax, x, y, label, linestyle="-", linewidth=2):
    ax.plot(x, y, label=label, linestyle=linestyle, linewidth=linewidth)


def main():
    st.set_page_config(page_title="FMCG Demand Forecasting", layout="wide")
    st.title("AI-Driven Demand Forecasting for FMCG (Statistical + ML)")

    data_path = "data/extended_fmcg_demand_forecasting.csv"
    raw = get_raw(data_path)

    with st.sidebar:
        st.header("Controls")
        cat = st.selectbox("Product Category", sorted(raw["Product_Category"].unique()))
        loc = st.selectbox("Store Location", sorted(raw["Store_Location"].unique()))
        horizon = st.select_slider("Forecast Horizon (days)", options=[7, 14, 30], value=14)
        model_name = st.radio("Model", ["Naive Baseline", "ARIMA (SARIMAX)", "Random Forest"])
        service_level = st.selectbox("Service Level", [0.90, 0.95, 0.99], index=1)

    daily = make_daily_segment(raw, cat, loc).sort_values("ds").reset_index(drop=True)

    # Guard: if too little data
    if len(daily) < 40:
        st.error("Not enough data for this segment. Try another Product Category / Store Location.")
        return

    # evaluation split
    test_days = min(60, max(30, len(daily) // 5))
    train = daily.iloc[:-test_days].copy()
    test = daily.iloc[-test_days:].copy()

    # pick model and forecast for test
    if model_name == "Naive Baseline":
        test_pred = baseline_naive(train["y"], steps=test_days)

    elif model_name == "ARIMA (SARIMAX)":
        test_pred = fit_predict_arima(train["y"], steps=test_days)

    else:
        # RF evaluation via supervised features
        sup = make_supervised_features(daily)
        test_size = min(test_days, max(10, len(sup) // 3))

        # if your src.models includes fit_predict_rf, use it; else fallback to naive
        try:
            from src.models import fit_predict_rf
            y_true, y_pred, _, _ = fit_predict_rf(sup, test_size=test_size)
            # align to test_pred length with leading NaNs
            test_pred = np.concatenate([np.repeat(np.nan, test_days - test_size), y_pred])
        except Exception:
            test_pred = baseline_naive(train["y"], steps=test_days)

    # metrics ignoring NaNs
    mask = ~np.isnan(test_pred)
    m = compute_metrics(test["y"].to_numpy()[mask], test_pred[mask]) if mask.sum() > 0 else None

    mae = _metric_value(m, "MAE", default=np.nan)
    rmse = _metric_value(m, "RMSE", default=np.nan)

    # =======================
    # KPI Row (executive view)
    # =======================
    st.subheader("📌 Segment Snapshot")
    c1, c2, c3, c4 = st.columns(4)

    avg_demand = float(test["y"].mean())
    last_demand = float(daily["y"].iloc[-1])
    demand_30d_avg = float(daily["y"].iloc[-30:].mean())

    c1.metric("Latest Demand", f"{last_demand:,.0f}")
    c2.metric("30-Day Avg Demand", f"{demand_30d_avg:,.0f}")
    c3.metric("MAE (Holdout)", f"{mae:,.0f}" if np.isfinite(mae) else "NA")
    c4.metric("RMSE (Holdout)", f"{rmse:,.0f}" if np.isfinite(rmse) else "NA")

    # Context metric
    st.caption(
        f"Context: Avg demand in holdout window = {avg_demand:,.0f}. "
        f"MAE% = {(mae/avg_demand)*100:.1f}%." if (np.isfinite(mae) and avg_demand > 0) else
        "Context: MAE% could not be computed for this segment."
    )

    st.divider()

    # =======================
    # Clean Historical Plot
    # =======================
    st.subheader("📊 Historical Demand Trend (Clean View)")

    fig1, ax1 = plt.subplots(figsize=(11, 4))
    # last 180 days for readability (not the whole history)
    view = daily.iloc[-180:] if len(daily) > 180 else daily
    plot_time_series(ax1, view["ds"], view["y"], "Demand", linewidth=2)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sales Volume")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend()

    st.pyplot(fig1, use_container_width=True)

    # =======================
    # Holdout Forecast vs Actual
    # =======================
    st.subheader("📈 Holdout Forecast vs Actual")

    fig2, ax2 = plt.subplots(figsize=(11, 4))
    plot_time_series(ax2, test["ds"], test["y"], "Actual", linewidth=2)
    plot_time_series(ax2, test["ds"], test_pred, "Predicted", linestyle="--", linewidth=2)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sales Volume")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend()

    st.pyplot(fig2, use_container_width=True)

    st.divider()

    # =======================
    # Next Horizon Forecast
    # =======================
    st.subheader(f"🔮 Next {horizon} Days Demand Forecast")

    if model_name == "Naive Baseline":
        future_pred = baseline_naive(daily["y"], steps=horizon)
        future_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")

    elif model_name == "ARIMA (SARIMAX)":
        future_pred = fit_predict_arima(daily["y"], steps=horizon)
        future_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")

    else:
        future_dates, future_pred = rf_recursive_forecast(daily, horizon)

    future_pred = np.maximum(np.array(future_pred, dtype=float), 0.0)
    forecast_df = pd.DataFrame({"ds": future_dates, "forecast_demand": future_pred})

    fig3, ax3 = plt.subplots(figsize=(11, 4))

    # show recent 45 days history + forecast for a clean story
    hist_view = daily.iloc[-45:] if len(daily) > 45 else daily

    plot_time_series(ax3, hist_view["ds"], hist_view["y"], "Recent History", linewidth=2)
    plot_time_series(ax3, forecast_df["ds"], forecast_df["forecast_demand"], "Forecast", linestyle="--", linewidth=2)

    ax3.axvline(daily["ds"].iloc[-1], linestyle=":", linewidth=2, label="Forecast Start")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Sales Volume")
    ax3.tick_params(axis="x", rotation=45)
    ax3.legend()

    st.pyplot(fig3, use_container_width=True)

    # =======================
    # Inventory Recommendation (ROP + reorder)
    # =======================
    st.subheader("📦 Inventory Recommendation (Reorder Suggestion)")

    lead_time = float(daily["lead_time_mean"].iloc[-1]) if "lead_time_mean" in daily.columns else 7.0
    current_stock = float(daily["stock_mean"].iloc[-1]) if "stock_mean" in daily.columns else 0.0

    z_map = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    z = z_map.get(service_level, 1.65)

    # If rmse not available, fallback to a rough proxy based on demand std
    daily_rmse = rmse if np.isfinite(rmse) else float(np.std(daily["y"].values))

    safety_stock = z * daily_rmse * np.sqrt(max(1.0, lead_time))
    avg_daily_forecast = float(forecast_df["forecast_demand"].mean())
    rop = avg_daily_forecast * lead_time + safety_stock
    reorder_qty = max(0.0, rop - current_stock)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Lead Time (days)", f"{lead_time:.1f}")
    r2.metric("Current Avg Stock", f"{current_stock:,.0f}")
    r3.metric("Reorder Point (ROP)", f"{rop:,.0f}")
    r4.metric(
        "Recommended Reorder Qty",
        f"{reorder_qty:,.0f}",
        delta="Action Required" if reorder_qty > 0 else "Stock Sufficient"
    )

    st.info(
        f"Insight: Expected average daily demand for this horizon ≈ **{avg_daily_forecast:,.0f}** units. "
        f"At **{service_level*100:.0f}%** service level, recommended reorder quantity is **{reorder_qty:,.0f}** units."
    )

    st.divider()

    # =======================
    # Table + Download
    # =======================
    st.subheader("📄 Forecast Table")
    st.dataframe(forecast_df, use_container_width=True)

    st.download_button(
        "Download Forecast CSV",
        forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="forecast.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
