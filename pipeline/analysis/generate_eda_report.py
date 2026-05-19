import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pipeline.analysis.eda import (
    business_technical_correlation,
    daily_orders,
    dataset_overview,
    driver_request_summary,
    high_pressure_zones,
    load_dataset,
    missing_values,
    order_status_summary,
    orders_by_day_of_week,
    orders_by_hour,
    outlier_summary,
    seasonality_heatmap_table,
    sparse_zones,
    top_zones,
)

BUSINESS_PATH = Path("data/analytical/business_load_hourly.parquet")
TECHNICAL_PATH = Path("data/analytical/technical_pressure_hourly.parquet")

REPORT_DIR = Path("reports/EDA")
TABLES_DIR = REPORT_DIR / "tables"
PLOTS_DIR = REPORT_DIR / "plots"


def save_table(df: pd.DataFrame, name: str):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{name}.csv", index=True)


def save_plot(name: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_daily_orders(df: pd.DataFrame):
    daily = daily_orders(df)

    plt.figure(figsize=(12, 5))
    plt.plot(daily["date"], daily["orders_count"])
    plt.title("Daily Order Volume")
    plt.xlabel("Date")
    plt.ylabel("Orders")
    plt.xticks(rotation=45)
    save_plot("daily_order_volume")


def plot_top_zones(df: pd.DataFrame):
    top = top_zones(df, n=20)

    plt.figure(figsize=(10, 7))
    plt.barh(top["area"][::-1], top["orders_count"][::-1])
    plt.title("Top 20 Zones by Order Volume")
    plt.xlabel("Orders")
    plt.ylabel("Zone")
    save_plot("top_20_zones")


def plot_orders_by_hour(df: pd.DataFrame):
    hourly = orders_by_hour(df)

    plt.figure(figsize=(10, 5))
    plt.bar(hourly["hour"], hourly["orders_count"])
    plt.title("Orders by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Orders")
    save_plot("orders_by_hour")


def plot_orders_by_day_of_week(df: pd.DataFrame):
    dow = orders_by_day_of_week(df)

    plt.figure(figsize=(8, 5))
    plt.bar(dow["day_of_week"], dow["orders_count"])
    plt.title("Orders by Day of Week")
    plt.xlabel("Day of Week, 0=Monday")
    plt.ylabel("Orders")
    save_plot("orders_by_day_of_week")


def plot_seasonality_heatmap(df: pd.DataFrame):
    pivot = seasonality_heatmap_table(df)

    plt.figure(figsize=(13, 5))
    plt.imshow(pivot, aspect="auto")
    plt.colorbar(label="Orders")
    plt.title("Order Volume Heatmap: Day of Week x Hour")
    plt.xlabel("Hour")
    plt.ylabel("Day of Week")
    plt.xticks(range(24))
    plt.yticks(range(7))
    save_plot("seasonality_heatmap")


def plot_driver_pressure(df: pd.DataFrame):
    daily = daily_orders(df)
    daily["requests_per_order"] = (
        daily["driver_requests_count"] / daily["orders_count"].clip(lower=1)
    )

    plt.figure(figsize=(12, 5))
    plt.plot(daily["date"], daily["requests_per_order"])
    plt.title("Daily Driver Requests per Order")
    plt.xlabel("Date")
    plt.ylabel("Requests per Order")
    plt.xticks(rotation=45)
    save_plot("daily_requests_per_order")


def plot_business_technical_scatter(business_df: pd.DataFrame, technical_df: pd.DataFrame):
    business_hourly = (
        business_df.groupby("window_start")
        .agg(
            orders_count=("orders_count", "sum"),
            driver_requests_count=("driver_requests_count", "sum"),
        )
        .reset_index()
    )

    technical_hourly = (
        technical_df.groupby("window_start")
        .agg(
            api_requests=("request_count", "sum"),
            p95_response_time=("p95_response_time", "max"),
            error_count=("error_count", "sum"),
        )
        .reset_index()
    )

    merged = business_hourly.merge(technical_hourly, on="window_start", how="inner")

    plt.figure(figsize=(8, 6))
    plt.scatter(merged["orders_count"], merged["api_requests"], alpha=0.35)
    plt.title("Orders vs API Requests")
    plt.xlabel("Orders")
    plt.ylabel("API Requests")
    save_plot("orders_vs_api_requests")

    plt.figure(figsize=(8, 6))
    plt.scatter(merged["driver_requests_count"], merged["p95_response_time"], alpha=0.35)
    plt.title("Driver Requests vs P95 Response Time")
    plt.xlabel("Driver Requests")
    plt.ylabel("P95 Response Time")
    save_plot("driver_requests_vs_p95_response_time")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    business_df = load_dataset(BUSINESS_PATH)
    technical_df = load_dataset(TECHNICAL_PATH)

    summary = {
        "business_overview": dataset_overview(business_df),
        "technical_overview": {
            "rows": int(len(technical_df)),
            "endpoints": int(technical_df["endpoint"].nunique()),
            "start_date": str(technical_df["window_start"].min()),
            "end_date": str(technical_df["window_start"].max()),
            "total_api_requests": int(technical_df["request_count"].sum()),
            "total_errors": int(technical_df["error_count"].sum()),
        },
        "order_status_summary": order_status_summary(business_df),
        "driver_request_summary": driver_request_summary(business_df),
    }

    with (REPORT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_table(missing_values(business_df), "missing_values")
    save_table(top_zones(business_df), "top_zones")
    save_table(sparse_zones(business_df), "sparse_zones")
    save_table(high_pressure_zones(business_df), "high_pressure_zones")
    save_table(outlier_summary(business_df), "outlier_summary")
    save_table(
        business_technical_correlation(business_df, technical_df),
        "business_technical_correlation",
    )

    plot_daily_orders(business_df)
    plot_top_zones(business_df)
    plot_orders_by_hour(business_df)
    plot_orders_by_day_of_week(business_df)
    plot_seasonality_heatmap(business_df)
    plot_driver_pressure(business_df)
    plot_business_technical_scatter(business_df, technical_df)

    print(f"EDA report generated in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
