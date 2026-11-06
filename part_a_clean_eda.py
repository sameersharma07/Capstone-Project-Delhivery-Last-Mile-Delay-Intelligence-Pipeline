import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.shipment_prep import clean_shipments, build_features

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

clean, medians = clean_shipments(raw)
model_data = build_features(clean, hubs)

model_data.to_csv("data/modelling_table.csv", index=False)

summary = model_data.describe(include="all").transpose()
summary.to_csv("outputs/eda_summary.csv")

delay_by_service = (
    model_data.groupby("service_type")["is_delayed"]
    .agg(["count", "sum", "mean"])
    .reset_index()
)

delay_by_service.to_csv(
    "outputs/delay_by_service.csv",
    index=False
)

delay_by_region = (
    model_data.groupby("region")["is_delayed"]
    .agg(["count", "sum", "mean"])
    .reset_index()
)

delay_by_region.to_csv(
    "outputs/delay_by_region.csv",
    index=False
)

plt.figure(figsize=(8, 5))
sns.countplot(data=model_data, x="service_type")
plt.title("Shipment Count by Service Type")
plt.tight_layout()
plt.savefig("outputs/service_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(model_data["delivery_hours"], bins=30, kde=True)
plt.title("Distribution of Delivery Hours")
plt.tight_layout()
plt.savefig("outputs/delivery_hours_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=model_data, x="is_delayed")
plt.title("Delayed vs On-Time Shipments")
plt.tight_layout()
plt.savefig("outputs/delay_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=model_data,
    x="service_type",
    y="delivery_hours"
)
plt.title("Delivery Hours by Service Type")
plt.tight_layout()
plt.savefig("outputs/delivery_by_service.png")
plt.close()

print("Part A completed.")
print("Raw rows:", len(raw))
print("Clean rows:", len(clean))
print("Modelling rows:", len(model_data))
print("Delayed shipments:", int(model_data["is_delayed"].sum()))
print("On-time shipments:", int((model_data["is_delayed"] == 0).sum()))
