import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for forecasting models.
    Streamlit Cloud & scikit-learn version safe.
    """
    mae = float(mean_absolute_error(y_true, y_pred))

    # RMSE computed manually to avoid version issues
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    }
