import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.data_prep import load_data, make_daily_segment, make_supervised_features
from src.models import baseline_naive, fit_predict_arima, compute_metrics
from sklearn.ensemble import RandomForestRegressor


@st.cache_data
def get_raw(path: str) -> pd.DataFrame:
    return load_data(path)


def rf_recursive_forecast(daily: pd.DataFrame, horizon: int, random_state: int = 42):
    """Train RF on supervised features and recursively forecast next horizon days."""
    supervised = make_supervised_features(daily)
    feature_cols = [
        'price_mean','promo_sum','supplier_cost_mean','lead_time_mean','stock_mean',
        'dow','month','year','lag_1','lag_7','lag_14','roll_mean_7','roll_std_7'
    ]
    feature_cols = [c for c in feature_cols if c in supervised.columns]

    X = supervised[feature_cols].values
    y = supervised['y'].values

    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    model.fit(X, y)

    # Start from the last available daily history
    hist = daily.copy().sort_values('ds').reset_index(drop=True)
    preds = []
    for step in range(1, horizon + 1):
        next_date = hist['ds'].iloc[-1] + pd.Timedelta(days=1)
        # For exogenous: carry forward last known values, promo default 0
        row = {
            'ds': next_date,
            'y': np.nan,
            'price_mean': float(hist['price_mean'].iloc[-1]),
            'promo_sum': 0.0,
            'supplier_cost_mean': float(hist['supplier_cost_mean'].iloc[-1]),
            'lead_time_mean': float(hist['lead_time_mean'].iloc[-1]),
            'stock_mean': float(hist['stock_mean'].iloc[-1]),
            'dow': int(next_date.dayofweek),
            'month': int(next_date.month),
            'year': int(next_date.year),
        }
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        # recompute features for last row only
        tmp = make_supervised_features(hist)
        last = tmp.iloc[-1]
        X_last = last[feature_cols].values.reshape(1, -1)
        yhat = float(model.predict(X_last)[0])
        preds.append(yhat)
        hist.loc[hist.index[-1], 'y'] = yhat

    future_dates = pd.date_range(daily['ds'].max() + pd.Timedelta(days=1), periods=horizon, freq='D')
    return future_dates, np.array(preds)


def main():
    st.set_page_config(page_title='FMCG Demand Forecasting – Capstone', layout='wide')
    st.title('AI-Driven Demand Forecasting for FMCG (Statistical + ML)')

    data_path = 'data/extended_fmcg_demand_forecasting.csv'
    raw = get_raw(data_path)

    with st.sidebar:
        st.header('Controls')
        cat = st.selectbox('Product Category', sorted(raw['Product_Category'].unique()))
        loc = st.selectbox('Store Location', sorted(raw['Store_Location'].unique()))
        horizon = st.select_slider('Forecast Horizon (days)', options=[7, 14, 30], value=14)
        model_name = st.radio('Model', ['Naive Baseline', 'ARIMA (SARIMAX)', 'Random Forest'])
        service_level = st.selectbox('Service Level', [0.90, 0.95, 0.99], index=1)

    daily = make_daily_segment(raw, cat, loc)

    # evaluation split
    test_days = min(60, max(30, len(daily)//5))
    train = daily.iloc[:-test_days].copy()
    test = daily.iloc[-test_days:].copy()

    # pick model and forecast for test
    if model_name == 'Naive Baseline':
        test_pred = baseline_naive(train['y'], steps=test_days)
    elif model_name == 'ARIMA (SARIMAX)':
        test_pred = fit_predict_arima(train['y'], steps=test_days)
    else:
        # for evaluation, use RF on supervised with last test_days approx
        sup = make_supervised_features(daily)
        test_size = min(test_days, len(sup)//3)
        from src.models import fit_predict_rf
        y_true, y_pred, _, _ = fit_predict_rf(sup, test_size=test_size)
        test_pred = np.concatenate([np.repeat(np.nan, test_days - test_size), y_pred])

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader('Historical Demand (Segment)')
        fig = plt.figure()
        plt.plot(daily['ds'], daily['y'])
        plt.xlabel('Date')
        plt.ylabel('Sales Volume')
        st.pyplot(fig)

    with col2:
        st.subheader('Model Accuracy (Holdout)')
        # compute metrics ignoring nan
        mask = ~np.isnan(test_pred)
        m = compute_metrics(test['y'].to_numpy()[mask], test_pred[mask])
        st.metric('MAE', f"{m.mae:,.2f}")
        st.metric('RMSE', f"{m.rmse:,.2f}")

    st.subheader('Holdout Forecast vs Actual')
    fig2 = plt.figure()
    plt.plot(test['ds'], test['y'], label='Actual')
    plt.plot(test['ds'], test_pred, label='Predicted')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Sales Volume')
    st.pyplot(fig2)

    st.subheader(f'Next {horizon} Days Forecast')

    # Fit on full data and forecast future
    if model_name == 'Naive Baseline':
        future_pred = baseline_naive(daily['y'], steps=horizon)
        future_dates = pd.date_range(daily['ds'].max() + pd.Timedelta(days=1), periods=horizon, freq='D')
    elif model_name == 'ARIMA (SARIMAX)':
        future_pred = fit_predict_arima(daily['y'], steps=horizon)
        future_dates = pd.date_range(daily['ds'].max() + pd.Timedelta(days=1), periods=horizon, freq='D')
    else:
        future_dates, future_pred = rf_recursive_forecast(daily, horizon)

    forecast_df = pd.DataFrame({'ds': future_dates, 'forecast_demand': future_pred})

    # Simple replenishment recommendation
    lead_time = float(daily['lead_time_mean'].iloc[-1])
    z_map = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    z = z_map.get(service_level, 1.65)
    daily_rmse = m.rmse
    safety_stock = z * daily_rmse * np.sqrt(max(1.0, lead_time))
    avg_daily_forecast = float(np.maximum(forecast_df['forecast_demand'], 0).mean())
    rop = avg_daily_forecast * lead_time + safety_stock
    current_stock = float(daily['stock_mean'].iloc[-1])
    reorder_qty = max(0.0, rop - current_stock)

    c1, c2, c3 = st.columns(3)
    c1.metric('Lead Time (days)', f"{lead_time:.1f}")
    c2.metric('Current Stock (avg)', f"{current_stock:,.0f}")
    c3.metric('Suggested Reorder Qty', f"{reorder_qty:,.0f}")

    fig3 = plt.figure()
    plt.plot(forecast_df['ds'], forecast_df['forecast_demand'])
    plt.xlabel('Date')
    plt.ylabel('Forecast Demand')
    st.pyplot(fig3)

    st.dataframe(forecast_df, use_container_width=True)
    st.download_button('Download Forecast CSV', forecast_df.to_csv(index=False).encode('utf-8'), file_name='forecast.csv')


if __name__ == '__main__':
    main()
