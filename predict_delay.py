import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, build_features

package = joblib.load(
    "models/delay_intelligence_model.pkl"
)

raw_new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

clean_new, _ = clean_shipments(
    raw_new,
    medians=package["medians"]
)

features = build_features(clean_new, hubs)

X_new = features[package["feature_columns"]]

predicted_hours = package["regression_model"].predict(X_new)

delay_probabilities = package["classifier"].predict_proba(X_new)[:, 1]

delay_predictions = (
    delay_probabilities >= package["threshold"]
).astype(int)

results = pd.DataFrame({
    "shipment_id": features["shipment_id"],
    "hub_id": features["hub_id"],
    "service_type": features["service_type"],
    "predicted_delivery_hours": predicted_hours.round(2),
    "delay_probability": delay_probabilities.round(4),
    "predicted_delay": delay_predictions
})

results["predicted_status"] = results["predicted_delay"].map({
    0: "On Time",
    1: "Delayed"
})

results.to_csv(
    "outputs/new_shipment_predictions.csv",
    index=False
)

print("Prediction completed.")
print(results)
