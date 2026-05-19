from pathlib import Path
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

IN = Path("data/analytical/business_load_hourly.parquet")
AREA_POLICY = Path("data/analytical/area_modeling_policy.parquet")
OUT = Path("data/features/business_features.parquet")

LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336] # past 1h, 2h, 3h, 6h, 12h, 1d, 2d, 3d, 7d, 14d
ROLLS = [3, 6, 12, 24, 72, 168] # past 3h, 6h, 12h, 1d, 3d, 7d
EWMS = [6, 24, 168] # smoothed history over short, daily, and weekly horizons

SIGNALS = [
    "orders_count",
    "driver_requests_count",
    "requests_per_order",
    "acceptance_rate",
    "rejection_rate",
    "cancellation_rate",
]

COUNT_COLUMNS = [
    "orders_count",
    "completed_orders",
    "canceled_orders",
    "failed_orders",
    "other_orders",
    "unique_merchants_count",
    "unique_branches_count",
    "driver_requests_count",
    "accepted_requests",
    "rejected_requests",
    "ignored_requests",
    "pending_requests",
    "unique_requested_drivers",
]

ZERO_FILL_COLUMNS = COUNT_COLUMNS

def safe_rate(num, den):
    return num.div(den).where(den > 0)

def prepare_hourly_area_frame(df):
    df = df.sort_values("window_start").copy()
    if df["window_start"].duplicated().any():
        area = df["area"].iloc[0]
        raise ValueError(f"duplicate hourly rows found for area: {area}")

    original_windows = df["window_start"]
    full_index = pd.date_range(
        df["window_start"].min(),
        df["window_start"].max(),
        freq="h",
        tz=df["window_start"].dt.tz,
        name="window_start",
    )

    df = df.set_index("window_start").reindex(full_index)
    df["area"] = df["area"].ffill().bfill()
    if "country" in df.columns:
        df["country"] = df["country"].ffill().bfill()

    fill_cols = [col for col in ZERO_FILL_COLUMNS if col in df.columns]
    df[fill_cols] = df[fill_cols].fillna(0)

    df["requests_per_order"] = safe_rate(df["driver_requests_count"], df["orders_count"])
    df["acceptance_rate"] = safe_rate(df["accepted_requests"], df["driver_requests_count"])
    df["rejection_rate"] = safe_rate(df["rejected_requests"], df["driver_requests_count"])
    df["cancellation_rate"] = safe_rate(df["canceled_orders"], df["orders_count"])

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.day_of_week
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"].isin([4, 5]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_observed_row"] = df.index.isin(original_windows)
    return df.reset_index()

def build_area_features(df):
    df=prepare_hourly_area_frame(df)
    features = {}
    
    for col in SIGNALS:
        if col not in df.columns:
            continue
        
        for lag in LAGS:
            features[f"{col}_lag_{lag}h"] = df[col].shift(lag)
            
        shifted=df[col].shift(1)
        for window in ROLLS:
            rolling = shifted.rolling(window=window, min_periods=1)
            features[f"{col}_roll_mean_{window}h"] = rolling.mean()
            features[f"{col}_roll_std_{window}h"] = rolling.std()
            features[f"{col}_roll_max_{window}h"] = rolling.max()
            features[f"{col}_roll_median_{window}h"] = rolling.median()
            features[f"{col}_roll_sum_{window}h"] = rolling.sum()

        for span in EWMS:
            features[f"{col}_ewm_mean_{span}h"] = shifted.ewm(span=span, adjust=False).mean()

        features[f"{col}_lag_24h_delta"] = df[col].shift(1) - df[col].shift(24)
        features[f"{col}_lag_168h_delta"] = df[col].shift(1) - df[col].shift(168)
            
    features["orders_velocity_1h"] = df["orders_count"].shift(1) - df["orders_count"].shift(2)
    features["requests_velocity_1h"] = df["driver_requests_count"].shift(1) - df["driver_requests_count"].shift(2)
    
    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    
    df=pd.read_parquet(IN)
    area_policy = pd.read_parquet(AREA_POLICY)
    kept_areas = area_policy.loc[area_policy["is_modeling_eligible"], "area"]
    dropped_areas = df["area"].nunique() - kept_areas.nunique()
    df = df[df["area"].isin(kept_areas)]
    log.info(
        "keeping %s modeling-eligible areas from %s; dropped %s sparse or noisy areas",
        kept_areas.nunique(),
        AREA_POLICY,
        dropped_areas,
    )

    result=pd.concat([build_area_features(group) for _,group in df.groupby("area")], ignore_index=True)
    
    required = ["orders_count_lag_168h", "driver_requests_count_lag_168h"]
    missing = [c for c in required if c not in result.columns]
    if missing:
        raise ValueError(f"missing required feature columns: {missing}")
    result = result.dropna(subset=required)
    history_feature_cols = [
        c for c in result.columns
        if "_lag_" in c or "_roll_" in c or "_ewm_" in c or "_velocity_" in c
    ]
    result[history_feature_cols] = result[history_feature_cols].fillna(0)
    result = result.drop(columns=["is_observed_row"])

    numeric_cols = result.select_dtypes(include=["number", "bool"]).columns
    result[numeric_cols] = result[numeric_cols].apply(pd.to_numeric, downcast="float")

    result.to_parquet(OUT, index=False)
    log.info("saved features: %s rows, %s columns", len(result), len(result.columns))
    
if __name__ == "__main__":
    main()
