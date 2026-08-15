"""Generate a demo retail source: a SQLite database and an Excel workbook.

The data is synthetic but deliberately not clean — it carries a growth trend,
December seasonality, a pricing changepoint, right-skewed revenue, missing costs,
a 30:1 imbalanced return flag, and a leaking column. Those are exactly the things
the pattern layer should find, so the demo is a test of it rather than a decoration.

    python -m dataplatform.examples.make_sample_data
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("sample")
START = date(2022, 1, 1)
DAYS = 1095  # three years
RNG = np.random.default_rng(20240513)

REGIONS = ["North", "South", "East", "West", "Central"]
REGION_WEIGHT = [0.28, 0.22, 0.18, 0.24, 0.08]
CHANNELS = ["Online", "Retail", "Partner"]
CATEGORIES = ["Electronics", "Apparel", "Home", "Grocery", "Sports"]
CATEGORY_PRICE = {"Electronics": 320, "Apparel": 48, "Home": 90, "Grocery": 18, "Sports": 72}

CHANGEPOINT_DAY = 700  # a price rise partway through, for the changepoint detector


def build_orders() -> pd.DataFrame:
    rows = []
    order_id = 100_000

    for day_index in range(DAYS):
        current = START + timedelta(days=day_index)

        # volume: growth trend + yearly seasonality + weekly cycle + noise
        trend = 40 + 0.035 * day_index
        yearly = 1 + 0.45 * np.sin(2 * np.pi * (day_index - 60) / 365.25)
        december = 1.6 if current.month == 12 else 1.0
        weekly = 1.25 if current.weekday() >= 5 else 0.95
        expected = trend * yearly * december * weekly
        n_orders = max(1, int(RNG.poisson(expected)))

        for _ in range(n_orders):
            order_id += 1
            region = RNG.choice(REGIONS, p=REGION_WEIGHT)
            channel = RNG.choice(CHANNELS, p=[0.55, 0.33, 0.12])
            category = RNG.choice(CATEGORIES, p=[0.15, 0.25, 0.2, 0.3, 0.1])

            base_price = CATEGORY_PRICE[category]
            price_shock = 1.18 if day_index >= CHANGEPOINT_DAY else 1.0
            unit_price = round(float(RNG.lognormal(np.log(base_price), 0.35)) * price_shock, 2)
            units = int(RNG.integers(1, 6)) if category != "Grocery" else int(RNG.integers(1, 15))
            discount = float(np.round(RNG.choice([0, 0, 0, 0.05, 0.1, 0.2]), 2))
            revenue = round(unit_price * units * (1 - discount), 2)

            # ~3% of rows are genuine outliers: bulk orders
            if RNG.random() < 0.03:
                revenue = round(revenue * float(RNG.uniform(8, 25)), 2)
                units *= int(RNG.integers(10, 40))

            returned = int(RNG.random() < (0.055 if channel == "Online" else 0.012))
            cost = round(revenue * float(RNG.uniform(0.55, 0.78)), 2)

            rows.append(
                {
                    "order_id": order_id,
                    "order_date": current.isoformat(),
                    "customer_id": int(RNG.integers(1, 9000)),
                    "region": region,
                    "channel": channel,
                    "product_category": category,
                    "units": units,
                    "unit_price": unit_price,
                    "discount_pct": discount,
                    "revenue": revenue,
                    "cost": cost,
                    # gross_margin is revenue-derived: the leakage detector should catch it
                    "gross_margin": round(revenue - cost, 2),
                    "returned": returned,
                    "delivery_days": int(max(1, RNG.normal(4.2, 1.8))),
                }
            )

    frame = pd.DataFrame(rows)

    # ~12% of costs unrecorded, and missing more often on the Partner channel,
    # so the missingness is structured rather than random.
    missing = (RNG.random(len(frame)) < 0.09) | (
        (frame["channel"] == "Partner") & (RNG.random(len(frame)) < 0.35)
    )
    frame.loc[missing, "cost"] = np.nan
    frame.loc[missing, "gross_margin"] = np.nan

    frame["notes"] = np.where(
        RNG.random(len(frame)) < 0.06,
        "Customer requested expedited handling and a follow-up call from support.",
        "",
    )
    return frame


def build_targets() -> pd.DataFrame:
    """A small Excel-shaped planning sheet, with the messy headers those have."""
    months = pd.date_range("2022-01-01", periods=36, freq="MS")
    rows = []
    for month in months:
        for region in REGIONS:
            rows.append(
                {
                    "Month ": month.date().isoformat(),
                    "Region": region,
                    "Revenue Target": int(RNG.normal(120_000, 25_000)),
                    "Headcount": int(RNG.integers(4, 20)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    orders = build_orders()
    db_path = OUT / "retail.db"
    with sqlite3.connect(db_path) as conn:
        orders.to_sql("orders", conn, if_exists="replace", index=False)
        customers = (
            orders.groupby("customer_id")
            .agg(
                first_order=("order_date", "min"),
                last_order=("order_date", "max"),
                lifetime_orders=("order_id", "count"),
                lifetime_revenue=("revenue", "sum"),
            )
            .reset_index()
        )
        customers.to_sql("customers", conn, if_exists="replace", index=False)

    excel_path = OUT / "targets.xlsx"
    try:
        build_targets().to_excel(excel_path, sheet_name="Regional Targets", index=False)
        excel_note = f"  {excel_path}  (1 sheet)"
    except ImportError:
        excel_note = "  (openpyxl not installed — skipped targets.xlsx)"

    print("wrote:")
    print(f"  {db_path}  (orders: {len(orders):,} rows, customers: {orders.customer_id.nunique():,} rows)")
    print(excel_note)
    print()
    print("next:")
    print(f'  insight source add --name sales_db --type sql --uri sqlite:///{db_path.as_posix()}')
    print("  insight ingest --source sales_db")
    print('  insight ask "monthly revenue by region"')
    print("  insight analyze --dataset sales_db_orders --target revenue")


if __name__ == "__main__":
    main()
