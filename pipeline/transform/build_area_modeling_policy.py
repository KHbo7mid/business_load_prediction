import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

BUSINESS = Path("data/analytical/business_load_hourly.parquet")
OUT = Path("data/analytical/area_modeling_policy.parquet")
REVIEW_OUT = Path("reports/EDA/area_modeling_policy.csv")

MIN_TOTAL_ORDERS = 500
MIN_OBSERVED_HOURS = 24 * 30
MIN_AVG_ORDERS_PER_OBSERVED_HOUR = 0.05

BROAD_AREA_NAMES_REQUIRING_REVIEW = {
    "Bahrain",
    "Kuwait",
    "Kuwait City",
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_OUT.parent.mkdir(parents=True, exist_ok=True)

    business = pd.read_parquet(BUSINESS)

    area_policy = (
        business
        .groupby("area", as_index=False)
        .agg(
            total_orders=("orders_count", "sum"),
            total_driver_requests=("driver_requests_count", "sum"),
            observed_hours=("window_start", "nunique"),
            first_hour=("window_start", "min"),
            last_hour=("window_start", "max"),
        )
    )

    area_policy["avg_orders_per_observed_hour"] = (
        area_policy["total_orders"] / area_policy["observed_hours"]
    )
    area_policy["requests_per_order"] = (
        area_policy["total_driver_requests"] / area_policy["total_orders"].where(area_policy["total_orders"] > 0)
    )

    area_policy["is_modeling_eligible"] = (
        (area_policy["total_orders"] >= MIN_TOTAL_ORDERS)
        & (area_policy["observed_hours"] >= MIN_OBSERVED_HOURS)
        & (area_policy["avg_orders_per_observed_hour"] >= MIN_AVG_ORDERS_PER_OBSERVED_HOUR)
    )
    area_policy["needs_area_review"] = area_policy["area"].isin(BROAD_AREA_NAMES_REQUIRING_REVIEW)
    area_policy["is_modeling_eligible"] = (
        area_policy["is_modeling_eligible"]
        & ~area_policy["needs_area_review"]
    )

    area_policy["modeling_area"] = area_policy["area"].where(
        area_policy["is_modeling_eligible"],
        "OTHER_AREA",
    )

    area_policy = area_policy.sort_values(
        ["is_modeling_eligible", "total_orders"],
        ascending=[False, False],
    ).reset_index(drop=True)

    area_policy.to_parquet(OUT, index=False)
    area_policy.to_csv(REVIEW_OUT, index=False)

    eligible = area_policy["is_modeling_eligible"].sum()
    total = len(area_policy)
    eligible_order_share = (
        area_policy.loc[area_policy["is_modeling_eligible"], "total_orders"].sum()
        / area_policy["total_orders"].sum()
    )

    log.info("saved area modeling policy to %s", OUT)
    log.info("eligible areas: %s / %s", eligible, total)
    log.info("eligible order share: %.2f%%", eligible_order_share * 100)


if __name__ == "__main__":
    main()
