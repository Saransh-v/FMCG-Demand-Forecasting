import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the FMCG dataset and apply minimal type cleaning."""
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Coerce numeric fields
    for col in [
        'Sales_Volume',
        'Price',
        'Supplier_Cost',
        'Replenishment_Lead_Time',
        'Stock_Level',
        'Weekday',
        'Promotion',
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Date', 'Product_Category', 'Store_Location', 'Sales_Volume']).copy()

    # Ensure ints
    df['Promotion'] = df['Promotion'].fillna(0).astype(int)
    df['Weekday'] = df['Weekday'].fillna(df['Date'].dt.dayofweek).astype(int)
    df['Sales_Volume'] = df['Sales_Volume'].astype(float)

    # Normalize text columns
    df['Product_Category'] = df['Product_Category'].astype(str).str.strip()
    df['Store_Location'] = df['Store_Location'].astype(str).str.strip()
    return df


def make_daily_segment(df: pd.DataFrame, product_category: str, store_location: str) -> pd.DataFrame:
    """Aggregate to daily level for a given (category, location) segment.

    Returns a daily DataFrame with:
      - ds: date
      - y: sales volume (target)
      - exogenous features aggregated per day (mean/sum)

    Missing days are filled with y=0 and exogenous fields forward-filled.
    """
    seg = df[(df['Product_Category'] == product_category) & (df['Store_Location'] == store_location)].copy()
    if seg.empty:
        raise ValueError(f"No rows for segment: {product_category} / {store_location}")

    daily = seg.groupby('Date', as_index=False).agg(
        y=('Sales_Volume', 'sum'),
        price_mean=('Price', 'mean'),
        promo_sum=('Promotion', 'sum'),
        supplier_cost_mean=('Supplier_Cost', 'mean'),
        lead_time_mean=('Replenishment_Lead_Time', 'mean'),
        stock_mean=('Stock_Level', 'mean'),
    )
    daily = daily.sort_values('Date').rename(columns={'Date': 'ds'})

    # Fill missing dates
    full = pd.DataFrame({'ds': pd.date_range(daily['ds'].min(), daily['ds'].max(), freq='D')})
    daily = full.merge(daily, on='ds', how='left')

    daily['y'] = daily['y'].fillna(0.0)
    for c in ['price_mean', 'supplier_cost_mean', 'lead_time_mean', 'stock_mean']:
        daily[c] = daily[c].ffill().bfill()
    daily['promo_sum'] = daily['promo_sum'].fillna(0.0)

    # calendar features
    daily['dow'] = daily['ds'].dt.dayofweek
    daily['month'] = daily['ds'].dt.month
    daily['year'] = daily['ds'].dt.year

    return daily


def make_supervised_features(daily: pd.DataFrame, max_lag: int = 14) -> pd.DataFrame:
    """Create lag and rolling features for ML models."""
    d = daily.copy().sort_values('ds')

    for lag in [1, 7, 14]:
        if lag <= max_lag:
            d[f'lag_{lag}'] = d['y'].shift(lag)

    d['roll_mean_7'] = d['y'].shift(1).rolling(7).mean()
    d['roll_std_7'] = d['y'].shift(1).rolling(7).std()

    # Avoid NA rows for training
    d = d.dropna().reset_index(drop=True)

    return d
