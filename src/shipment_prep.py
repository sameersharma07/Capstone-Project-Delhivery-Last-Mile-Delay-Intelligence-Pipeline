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


def clean_shipments(df):
    df = df.copy()

    # Remove duplicate shipment records
    df = df.drop_duplicates()

    # Handle missing numerical values
    numeric_columns = df.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    # Handle missing categorical values
    categorical_columns = df.select_dtypes(include="object").columns

    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown")

    return df


def build_features(df):
    df = df.copy()

    # Log transformation of declared value
    if "declared_value" in df.columns:
        df["log_declared_value"] = np.log1p(
            df["declared_value"].clip(lower=0)
        )

    # Weekend indicator
    if "dispatch_day" in df.columns:
        df["is_weekend"] = (
            df["dispatch_day"].isin(["Saturday", "Sunday"])
        ).astype(int)

    # Delay-related target columns are created only
    # when actual delivery time is available.
    if "delivery_hours" in df.columns:
        df["delay_hours"] = (
            df["delivery_hours"] - df["promised_hours"]
        )

        df["is_delayed"] = (
            df["delay_hours"] > 0
        ).astype(int)

    return df
