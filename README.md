# AI-Driven Demand Forecasting for FMCG (Capstone)

This project uses the provided dataset `extended_fmcg_demand_forecasting.csv` to build a demand forecasting system and a simple Streamlit dashboard.

Models included:
- Baseline (Naive / last value)
- ARIMA (SARIMAX)
- Random Forest (feature-engineered lags + rolling statistics)

## Project Structure
- `data/` – dataset (CSV)
- `src/` – data prep + model code
- `notebooks/` – exploration & modeling notebook
- `outputs/` – saved daily aggregates, holdout forecasts, metrics JSON
- `report/` – final report (DOCX) + charts
- `slides/` – presentation deck (PPTX)
- `app.py` – Streamlit dashboard

## Setup
```bash
pip install -r requirements.txt
```

## Run evaluation scripts
```bash
python src/train_and_evaluate.py
```

## Run Streamlit app
```bash
streamlit run app.py
```

