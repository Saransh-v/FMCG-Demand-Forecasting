import json
from pathlib import Path

import pandas as pd

from data_prep import load_data, make_daily_segment, make_supervised_features
from models import (
    baseline_naive,
    compute_metrics,
    fit_predict_arima,
    fit_predict_rf,
)


def run_for_segment(data_path: str, product_category: str, store_location: str, test_days: int = 60):
    df = load_data(data_path)
    daily = make_daily_segment(df, product_category, store_location)

    # Train-test split: last N days
    daily = daily.sort_values('ds').reset_index(drop=True)
    if len(daily) <= test_days + 30:
        test_days = max(30, min(45, len(daily)//3))

    train = daily.iloc[:-test_days].copy()
    test = daily.iloc[-test_days:].copy()

    y_train = train['y']
    y_test = test['y'].to_numpy()

    future_dates = test['ds']

    preds = {}
    preds['naive'] = baseline_naive(y_train, steps=test_days)
    preds['arima'] = fit_predict_arima(y_train, steps=test_days)

    # Random Forest
    supervised = make_supervised_features(daily)
    # map test_size in supervised space: last test_days that also exist after dropna
    # approximate by selecting last test_days rows
    test_size = min(test_days, len(supervised)//3)
    rf_y_true, rf_y_pred, rf_model, rf_features = fit_predict_rf(supervised, test_size=test_size)

    # metrics
    results = {}
    for k in ['naive', 'arima']:
        m = compute_metrics(y_test, preds[k])
        results[k] = {'MAE': m.mae, 'RMSE': m.rmse}

    mrf = compute_metrics(rf_y_true, rf_y_pred)
    results['random_forest'] = {'MAE': mrf.mae, 'RMSE': mrf.rmse, 'test_size': int(test_size), 'features': rf_features}

    return daily, train, test, preds, results


def main():
    base = Path(__file__).resolve().parents[1]
    data_path = str(base / 'data' / 'extended_fmcg_demand_forecasting.csv')
    out_dir = base / 'outputs'
    out_dir.mkdir(exist_ok=True, parents=True)

    df = load_data(data_path)
    segments = df.groupby(['Product_Category', 'Store_Location']).size().reset_index(name='n')
    # pick top 3 segments by row count for a representative evaluation
    top = segments.sort_values('n', ascending=False).head(3)

    summary = []
    for _, row in top.iterrows():
        cat = row['Product_Category']
        loc = row['Store_Location']
        daily, train, test, preds, results = run_for_segment(data_path, cat, loc)

        # save segment outputs
        seg_key = f"{cat.replace(' ', '_')}_{loc}"
        daily.to_csv(out_dir / f"daily_{seg_key}.csv", index=False)
        test_out = test[['ds', 'y']].copy()
        for k, v in preds.items():
            test_out[f"pred_{k}"] = v
        test_out.to_csv(out_dir / f"forecast_test_{seg_key}.csv", index=False)

        summary.append({'segment': seg_key, 'rows': int(row['n']), 'metrics': results})

    (out_dir / 'metrics_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
