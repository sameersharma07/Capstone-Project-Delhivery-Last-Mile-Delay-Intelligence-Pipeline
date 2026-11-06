import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/modelling_table.csv")

hub_summary = (
    df.groupby(
        ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    )
    .agg(
        shipment_count=("shipment_id", "count"),
        average_delivery_hours=("delivery_hours", "mean"),
        average_distance_km=("distance_km", "mean"),
        average_stops=("n_stops", "mean"),
        delay_rate=("is_delayed", "mean"),
        average_weight_kg=("parcel_weight_kg", "mean")
    )
    .reset_index()
)

segment_features = [
    "shipment_count",
    "average_delivery_hours",
    "average_distance_km",
    "average_stops",
    "delay_rate",
    "average_weight_kg",
    "hub_tier",
    "hub_daily_capacity"
]

X = hub_summary[segment_features]

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_scaled = pipeline.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

hub_summary["segment"] = kmeans.fit_predict(X_scaled)

segment_summary = (
    hub_summary.groupby("segment")
    .agg(
        hub_count=("hub_id", "count"),
        average_delivery_hours=("average_delivery_hours", "mean"),
        average_distance_km=("average_distance_km", "mean"),
        average_delay_rate=("delay_rate", "mean"),
        average_capacity=("hub_daily_capacity", "mean")
    )
    .reset_index()
)

hub_summary.to_csv("outputs/hub_segments.csv", index=False)
segment_summary.to_csv("outputs/segment_summary.csv", index=False)

print("Part D completed.")
print(segment_summary)
