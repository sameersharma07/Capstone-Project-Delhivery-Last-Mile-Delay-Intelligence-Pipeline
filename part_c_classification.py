
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.shipment_prep import FEATURE_COLUMNS


df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURE_COLUMNS]
y = df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

classifier = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

classifier.fit(X_train, y_train)

valid_probabilities = classifier.predict_proba(X_valid)[:, 1]

thresholds = [i / 100 for i in range(10, 91, 5)]

best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:
    valid_predictions = (
        valid_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_valid,
        valid_predictions,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

test_probabilities = classifier.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

metrics = {
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "F1": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "Threshold": best_threshold
}

pd.DataFrame([metrics]).to_csv(
    "outputs/classification_metrics.csv",
    index=False
)

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

pd.DataFrame(
    matrix,
    index=["Actual_On_Time", "Actual_Delayed"],
    columns=["Predicted_On_Time", "Predicted_Delayed"]
).to_csv("outputs/confusion_matrix.csv")

report = classification_report(
    y_test,
    test_predictions,
    zero_division=0,
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    "outputs/classification_report.csv"
)

prediction_results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": test_predictions,
    "Delay_Probability": test_probabilities
})

prediction_results.to_csv(
    "outputs/classification_test_predictions.csv",
    index=False
)

joblib.dump(classifier, "models/classifier_candidate.pkl")
joblib.dump(best_threshold, "models/classification_threshold.pkl")

print("Part C completed.")
print("Selected threshold:", best_threshold)
print(metrics)
print(matrix)
