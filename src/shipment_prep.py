
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

PROMISED_HOURS = {
    "Express": 24,
    "Standard": 48,
    "Economy": 72
}


def clean_shipments(raw, medians=None):
    df = raw.copy()

    df = df.drop_duplicates(subset="shipment_id").copy()

    df["service_type"] = (
        df["service_type"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    df["payment_mode"] = (
        df["payment_mode"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.title()
    )

    df["dispatch_time"] = pd.to_datetime(
        df["dispatch_time"],
        errors="coerce"
    )

    df["distance_km"] = pd.to_numeric(
        df["distance_km"], errors="coerce"
    )

    df["parcel_weight_kg"] = pd.to_numeric(
        df["parcel_weight_kg"], errors="coerce"
    )

    df["parcel_weight_kg"] = df["parcel_weight_kg"].replace(
        999.0, np.nan
    )

    df["declared_value"] = (
        df["declared_value"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    df["declared_value"] = pd.to_numeric(
        df["declared_value"], errors="coerce"
    )

    numeric_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value"
    ]

    if medians is None:
        medians = {
            col: df[col].median()
            for col in numeric_columns
        }

    for col in numeric_columns:
        df[col] = df[col].fillna(medians[col])

    df["payment_mode"] = df["payment_mode"].replace(
        {"Nan": "Unknown", "None": "Unknown"}
    )

    df = df.dropna(
        subset=["dispatch_time", "hub_id", "service_type"]
    )

    return df, medians


def build_features(clean, hubs):
    df = clean.copy()

    df = df.merge(
        hubs[["hub_id", "hub_tier", "hub_daily_capacity", "region"]],
        on="hub_id",
        how="left"
    )

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)

    df["log_declared_value"] = np.log1p(
        df["declared_value"].clip(lower=0)
    )

    df["is_cod"] = (
        df["payment_mode"].str.upper() == "COD"
    ).astype(int)

    df["is_weekend"] = (
        df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    df["dispatch_hour"] = df["dispatch_time"].dt.hour

    for col in NOISE_COLUMNS:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0)

    df["rain_flag"] = pd.to_numeric(
        df["rain_flag"], errors="coerce"
    ).fillna(0)

    df["sale_event_flag"] = pd.to_numeric(
        df["sale_event_flag"], errors="coerce"
    ).fillna(0)

    df["hub_tier"] = df["hub_tier"].fillna(
        df["hub_tier"].median()
    )

    df["hub_daily_capacity"] = df["hub_daily_capacity"].fillna(
        df["hub_daily_capacity"].median()
    )

    if "delivery_hours" in df.columns:
    df["delay_hours"] = df["delivery_hours"] - df["promised_hours"]
    df["is_delayed"] = (df["delay_hours"] > 0).astype(int)

return df
