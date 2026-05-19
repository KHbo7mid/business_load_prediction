from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    return df


def dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "start_date": str(df["window_start"].min()),
        "end_date": str(df["window_start"].max()),
        "zones": int(df["area"].nunique()),
        "total_orders": int(df["orders_count"].sum()),
        "total_driver_requests": int(df["driver_requests_count"].sum()),
        "avg_requests_per_order": float(df["requests_per_order"].mean()),
    }


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )
    result["missing_rate"] = result["missing_count"] / len(df)
    return result.sort_values("missing_count", ascending=False)


def top_zones(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return (
        df.groupby("area")
        .agg(
            orders_count=("orders_count", "sum"),
            driver_requests_count=("driver_requests_count", "sum"),
            avg_requests_per_order=("requests_per_order", "mean"),
            avg_rejection_rate=("rejection_rate", "mean"),
        )
        .sort_values("orders_count", ascending=False)
        .head(n)
        .reset_index()
    )


def sparse_zones(df: pd.DataFrame, min_orders: int = 10) -> pd.DataFrame:
    zone_stats = (
        df.groupby("area")
        .agg(
            total_orders=("orders_count", "sum"),
            active_hours=("orders_count", lambda s: int((s > 0).sum())),
            avg_orders_per_hour=("orders_count", "mean"),
        )
        .reset_index()
    )

    zone_stats["is_sparse"] = zone_stats["total_orders"] < min_orders
    return zone_stats.sort_values("total_orders")


def daily_orders(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(df["window_start"].dt.date)
        .agg(
            orders_count=("orders_count", "sum"),
            driver_requests_count=("driver_requests_count", "sum"),
        )
        .reset_index()
        .rename(columns={"window_start": "date"})
    )


def orders_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("hour")
        .agg(orders_count=("orders_count", "sum"))
        .reset_index()
        .sort_values("hour")
    )


def orders_by_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("day_of_week")
        .agg(orders_count=("orders_count", "sum"))
        .reset_index()
        .sort_values("day_of_week")
    )


def seasonality_heatmap_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index="day_of_week",
        columns="hour",
        values="orders_count",
        aggfunc="sum",
        fill_value=0,
    )


def order_status_summary(df: pd.DataFrame) -> Dict[str, Any]:
    completed = int(df["completed_orders"].sum())
    canceled = int(df["canceled_orders"].sum())
    failed = int(df["failed_orders"].sum())
    other = int(df["other_orders"].sum()) if "other_orders" in df.columns else 0
    total = int(df["orders_count"].sum())

    return {
        "total_orders": total,
        "completed_orders": completed,
        "canceled_orders": canceled,
        "failed_orders": failed,
        "other_orders": other,
        "completion_rate": completed / total if total else None,
        "cancellation_rate": canceled / total if total else None,
        "failure_rate": failed / total if total else None,
    }


def driver_request_summary(df: pd.DataFrame) -> Dict[str, Any]:
    total = int(df["driver_requests_count"].sum())
    accepted = int(df["accepted_requests"].sum())
    rejected = int(df["rejected_requests"].sum())
    ignored = int(df["ignored_requests"].sum())
    pending = int(df["pending_requests"].sum())

    return {
        "total_driver_requests": total,
        "accepted_requests": accepted,
        "rejected_requests": rejected,
        "ignored_requests": ignored,
        "pending_requests": pending,
        "acceptance_rate": accepted / total if total else None,
        "rejection_rate": rejected / total if total else None,
        "ignored_rate": ignored / total if total else None,
        "pending_rate": pending / total if total else None,
    }


def high_pressure_zones(df: pd.DataFrame, min_orders: int = 100, n: int = 20) -> pd.DataFrame:
    result = (
        df.groupby("area")
        .agg(
            orders_count=("orders_count", "sum"),
            driver_requests_count=("driver_requests_count", "sum"),
            rejected_requests=("rejected_requests", "sum"),
            unique_requested_drivers=("unique_requested_drivers", "sum"),
        )
        .reset_index()
    )

    result = result[result["orders_count"] >= min_orders].copy()
    result["requests_per_order"] = (
        result["driver_requests_count"] / result["orders_count"].clip(lower=1)
    )
    result["rejection_rate"] = (
        result["rejected_requests"] / result["driver_requests_count"].clip(lower=1)
    )

    return result.sort_values("requests_per_order", ascending=False).head(n)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "orders_count",
        "driver_requests_count",
        "requests_per_order",
        "avg_driver_search_trials",
        "avg_delivery_fee",
        "p95_delivery_fee",
        "avg_amount",
    ]
    existing = [c for c in columns if c in df.columns]

    rows = []
    for col in existing:
        s = df[col].dropna()
        rows.append(
            {
                "column": col,
                "min": float(s.min()) if len(s) else None,
                "p50": float(s.quantile(0.50)) if len(s) else None,
                "p95": float(s.quantile(0.95)) if len(s) else None,
                "p99": float(s.quantile(0.99)) if len(s) else None,
                "max": float(s.max()) if len(s) else None,
            }
        )

    return pd.DataFrame(rows)


def business_technical_correlation(
    business_df: pd.DataFrame,
    technical_df: pd.DataFrame,
) -> pd.DataFrame:
    business_hourly = (
        business_df.groupby("window_start")
        .agg(
            orders_count=("orders_count", "sum"),
            driver_requests_count=("driver_requests_count", "sum"),
            avg_requests_per_order=("requests_per_order", "mean"),
        )
        .reset_index()
    )

    technical_hourly = (
        technical_df.groupby("window_start")
        .agg(
            api_requests=("request_count", "sum"),
            avg_response_time=("avg_response_time", "mean"),
            p95_response_time=("p95_response_time", "max"),
            error_count=("error_count", "sum"),
        )
        .reset_index()
    )

    merged = business_hourly.merge(technical_hourly, on="window_start", how="inner")
    cols = [
        "orders_count",
        "driver_requests_count",
        "avg_requests_per_order",
        "api_requests",
        "avg_response_time",
        "p95_response_time",
        "error_count",
    ]

    return merged[cols].corr(numeric_only=True)
