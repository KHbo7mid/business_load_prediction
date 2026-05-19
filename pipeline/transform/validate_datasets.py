import sys
from pathlib import Path

import pandas as pd

BUSINESS = Path("data/analytical/business_load_hourly.parquet")
TECH = Path("data/analytical/technical_pressure_hourly.parquet")


def assert_check(name, condition, failures):
    if condition:
        print(f"OK: {name}")
    else:
        print(f"FAIL: {name}")
        failures.append(name)


def main():
    failures = []

    business = pd.read_parquet(BUSINESS)

    assert_check("Business has rows", len(business) > 0, failures)
    assert_check("no null window_start", business["window_start"].isna().sum() == 0, failures)
    assert_check("no null area", business["area"].isna().sum() == 0, failures)
    assert_check("orders non-negative", (business["orders_count"] >= 0).all(), failures)
    assert_check(
        "order subcounts <= total",
        (
            business["completed_orders"]
            + business["canceled_orders"]
            + business["failed_orders"]
            <= business["orders_count"]
        ).all(),
        failures,
    )
    assert_check("other orders non-negative", (business["other_orders"] >= 0).all(), failures)
    assert_check(
        "order subcounts plus other orders equal total",
        (
            business["completed_orders"]
            + business["canceled_orders"]
            + business["failed_orders"]
            + business["other_orders"]
            == business["orders_count"]
        ).all(),
        failures,
    )
    assert_check(
        "request subcounts <= total",
        (
            business["accepted_requests"]
            + business["rejected_requests"]
            + business["ignored_requests"]
            + business["pending_requests"]
            <= business["driver_requests_count"]
        ).all(),
        failures,
    )

    print("\nBusiness summary")
    print("rows:", len(business))
    print("areas:", business["area"].nunique())
    print("date:", business["window_start"].min(), "->", business["window_start"].max())
    print("orders:", int(business["orders_count"].sum()))
    print("requests:", int(business["driver_requests_count"].sum()))
    print("top areas:")
    print(business.groupby("area")["orders_count"].sum().sort_values(ascending=False).head(10))

    if TECH.exists():
        tech = pd.read_parquet(TECH)
        assert_check("technical has rows", len(tech) > 0, failures)
        assert_check("technical response time non-negative", (tech["avg_response_time"] >= 0).all(), failures)

        print("\nTechnical summary")
        print("rows:", len(tech))
        print("endpoints:", tech["endpoint"].nunique())
        print("date:", tech["window_start"].min(), "->", tech["window_start"].max())

    if failures:
        print("\nFAILED:", failures)
        sys.exit(1)

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
