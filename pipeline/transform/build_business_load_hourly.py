import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log=logging.getLogger(__name__)

ORDERS=Path("data/staging/orders_clean")
DELIVERY_REQUESTS=Path("data/staging/delivery_requests_clean")
OUTPUT=Path("data/analytical/business_load_hourly.parquet")

def safe_rate(num,den):
    return np.where(den>0,num/den,np.nan)

def main():
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    
    orders=pd.read_parquet(ORDERS)
    delivery_requests=pd.read_parquet(DELIVERY_REQUESTS)
    
    orders=orders.dropna(subset=["area"])
    orders["window_start"]=orders["createdAt"].dt.floor("h")
    orders["is_completed_order"] = (orders["orderStatus"] == "completed").astype("int8")
    orders["is_canceled_order"] = (orders["orderStatus"] == "canceled").astype("int8")
    orders["is_failed_order"] = (orders["orderStatus"] == "failed").astype("int8")
    order_context=orders[["order_id","window_start","area"]].drop_duplicates('order_id')
    
    requests_enriched=delivery_requests.merge(order_context,on="order_id",how="left")
    match_rate=requests_enriched["area"].notna().mean()
    log.info('delivery request area match rate: %.2f%%', match_rate*100)
    
    orders_group=orders.groupby(["window_start","area"], sort=False)
    orders_agg=(
        orders_group
        .agg(
            orders_count=("order_id","count"),
            completed_orders=("is_completed_order","sum"),
            canceled_orders=("is_canceled_order","sum"),
            failed_orders=("is_failed_order","sum"),
            unique_merchants_count=("merchant_id","nunique"),
            unique_branches_count=("branch_id","nunique"),
            country=("country", "first")
            
        )
        .reset_index()
    )

    requests_with_area = requests_enriched.dropna(subset=["area"]).copy()
    requests_with_area["accepted_request"] = (requests_with_area["status"] == "accepted").astype("int8")
    requests_with_area["rejected_request"] = (requests_with_area["status"] == "rejected").astype("int8")
    requests_with_area["ignored_request"] = (requests_with_area["status"] == "ignored").astype("int8")
    requests_with_area["pending_request"] = (requests_with_area["status"] == "pending").astype("int8")
    requests_agg=(
        requests_with_area
        .groupby(["window_start","area"], sort=False)
        .agg(
            driver_requests_count=("request_id","count"),
            accepted_requests=("accepted_request", "sum"),
            rejected_requests=("rejected_request", "sum"),
            ignored_requests=("ignored_request", "sum"),
            pending_requests=("pending_request", "sum"),
            unique_requested_drivers=("driver_id", "nunique"),
        )
        .reset_index()
    )
    df=orders_agg.merge(requests_agg,on=["window_start","area"],how="outer")
    count_cols = [
        "orders_count",
        "completed_orders",
        "canceled_orders",
        "failed_orders",
        "driver_requests_count",
        "accepted_requests",
        "rejected_requests",
        "ignored_requests",
        "pending_requests",
        "unique_requested_drivers",
        "unique_merchants_count",
        "unique_branches_count",
    ]
    for col in count_cols:
        df[col] = df[col].fillna(0).astype(int)

    df["other_orders"] = (
        df["orders_count"]
        - df["completed_orders"]
        - df["canceled_orders"]
        - df["failed_orders"]
    )
    
    df["requests_per_order"] = safe_rate(df["driver_requests_count"], df["orders_count"])
    df["acceptance_rate"] = safe_rate(df["accepted_requests"], df["driver_requests_count"])
    df["rejection_rate"] = safe_rate(df["rejected_requests"], df["driver_requests_count"])
    df["ignored_rate"] = safe_rate(df["ignored_requests"], df["driver_requests_count"])
    df["completion_rate"] = safe_rate(df["completed_orders"], df["orders_count"])
    df["cancellation_rate"] = safe_rate(df["canceled_orders"], df["orders_count"])
    df["failure_rate"] = safe_rate(df["failed_orders"], df["orders_count"])

    df["avg_requests_per_unique_driver"] = safe_rate(
        df["driver_requests_count"], df["unique_requested_drivers"])

    df["orders_per_unique_merchant"] = safe_rate(
        df["orders_count"], df["unique_merchants_count"])

    df["orders_per_unique_branch"] = safe_rate(
        df["orders_count"], df["unique_branches_count"])
    
    
    df["hour"]=df["window_start"].dt.hour
    df["day_of_week"]=df["window_start"].dt.day_of_week
    df["month"]=df["window_start"].dt.month
    df["is_weekend"]=df["day_of_week"].isin([4,5]).astype(int)
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    df=df.sort_values(["area","window_start"]).reset_index(drop=True)
    df.to_parquet(OUTPUT,index=False)
    
    log.info("saved business load: %s rows", len(df))
    log.info("areas: %s", df["area"].nunique())
    log.info("range: %s -> %s", df["window_start"].min(), df["window_start"].max())
    
if __name__ == "__main__":
    main()
    
