
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURE_COLUMNS


df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURE_COLUMNS]
y = df["delivery_hours"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])
}

results = []

best_name = None
best_model = None
best_mae = float("inf")

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, predictions)
    rmse = mean_squared_error(y_valid, predictions) ** 0.5
    r2 = r2_score(y_valid, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    if mae < best_mae:
        best_mae = mae
        best_name = name
        best_model = model

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/regression_model_comparison.csv",
    index=False
)

best_model.fit(
    pd.concat([X_train, X_valid]),
    pd.concat([y_train, y_valid])
)

test_predictions = best_model.predict(X_test)

test_results = pd.DataFrame({
    "Actual_Delivery_Hours": y_test.values,
    "Predicted_Delivery_Hours": test_predictions
})

test_results.to_csv(
    "outputs/regression_test_predictions.csv",
    index=False
)

test_metrics = {
    "Model": best_name,
    "MAE": mean_absolute_error(y_test, test_predictions),
    "RMSE": mean_squared_error(y_test, test_predictions) ** 0.5,
    "R2": r2_score(y_test, test_predictions)
}

pd.DataFrame([test_metrics]).to_csv(
    "outputs/regression_test_metrics.csv",
    index=False
)

joblib.dump(best_model, "models/regression_model.pkl")

print("Part B completed.")
print(results_df)
print(test_metrics)
