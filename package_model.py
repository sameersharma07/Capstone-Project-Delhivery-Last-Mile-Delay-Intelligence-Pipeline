import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, FEATURE_COLUMNS

raw = pd.read_csv("data/raw_shipments.csv")

_, medians = clean_shipments(raw)

regression_model = joblib.load(
    "models/regression_model.pkl"
)

classifier = joblib.load(
    "models/classifier_candidate.pkl"
)

threshold = joblib.load(
    "models/classification_threshold.pkl"
)

package = {
    "regression_model": regression_model,
    "classifier": classifier,
    "threshold": threshold,
    "feature_columns": FEATURE_COLUMNS,
    "medians": medians
}

joblib.dump(
    package,
    "models/delay_intelligence_model.pkl"
)

print("Model packaging completed.")
print("Saved: models/delay_intelligence_model.pkl")
