# Capstone Project — Delhivery Last-Mile Delay Intelligence Pipeline

**Project Duration:** 16 October 2026 – 6 November 2026  
**Programming Language:** Python  
**Development Environments:** Google Colab and Cursor

---

## 1. Project Overview

The Delhivery Last-Mile Delay Intelligence Pipeline is an end-to-end data analytics and machine learning project designed to analyze shipment delivery performance, predict delivery durations, identify potentially delayed shipments, and segment logistics hubs based on operational performance.

The project uses simulated shipment and hub datasets to demonstrate how data preprocessing, exploratory data analysis, regression, classification, and clustering can support logistics decision-making.

The pipeline also includes model packaging and a prediction workflow for evaluating new shipment records.

## 2. Project Objectives

- Generate and preprocess shipment and logistics hub datasets.
- Perform exploratory data analysis to identify delivery patterns.
- Identify shipment and operational characteristics associated with delays.
- Develop regression models to predict delivery duration.
- Develop classification models to identify potentially delayed shipments.
- Apply clustering to segment logistics hubs.
- Package trained models for reuse.
- Generate delivery-time and delay predictions for new shipments.

## 3. Technologies and Libraries

| Technology | Application |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and model evaluation |
| Joblib | Model serialization |
| Google Colab | Model development and experimentation |
| Cursor | Code editing and project organization |

## 4. Repository Structure

```text
Capstone-Project-Delhivery-Last-Mile-Delay-Intelligence-Pipeline/
│
├── data/
│   ├── hubs.csv
│   ├── raw_shipments.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
│
├── src/
│   └── shipment_prep.py
│
├── models/
│   ├── classification_threshold.pkl
│   ├── classifier_candidate.pkl
│   ├── delay_classifier.pkl
│   ├── delay_intelligence_model.pkl
│   ├── delivery_hours_model.pkl
│   ├── hub_kmeans.pkl
│   ├── hub_scaler.pkl
│   ├── model_config.pkl
│   └── regression_model.pkl
│
├── outputs/
│   ├── EDA reports and visualizations
│   ├── Regression evaluation results
│   ├── Classification evaluation results
│   ├── Hub segmentation results
│   └── New shipment predictions
│
├── generate_data.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
├── requirements.txt
└── README.md
```

## 5. Dataset Description

The project uses simulated shipment and logistics hub data.

| Dataset | Description |
|---|---|
| `hubs.csv` | Logistics hub information and operational attributes |
| `raw_shipments.csv` | Shipment records used for analysis and model training |
| `new_shipments.csv` | New shipment records used for prediction |
| `modelling_table.csv` | Cleaned and feature-engineered dataset prepared for modelling |

The data generation script creates:
- 24 logistics hubs.
- 1,200 shipment records.
- 48 new shipment records for prediction.

The generated datasets provide a consistent environment for testing the complete analytical workflow.

## 6. Project Workflow

### Part A — Data Generation, Cleaning and Exploratory Data Analysis

The first stage prepares the datasets for analysis and modelling.

Activities include:
- Generating shipment and hub datasets.
- Inspecting missing values, duplicates, and data inconsistencies.
- Cleaning and preprocessing shipment information.
- Performing feature engineering.
- Preparing the modelling dataset.
- Conducting exploratory data analysis.
- Generating summary reports and visualizations.

**Key data findings:**

| Finding | Observed Delay Rate |
|---|---:|
| Express shipments | 48.97% |
| Standard shipments | 20.56% |
| Economy shipments | 8.57% |
| East region | 43.25% |

Express shipments show the highest delay rate among the service categories, while the East region records a comparatively high delay rate.

### Part B — Delivery Time Prediction Using Regression

Regression models were developed to estimate shipment delivery duration in hours.

The models compared include:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).

**Final regression test results:**

| Evaluation Metric | Result |
|---|---:|
| MAE | 3.4711 |
| RMSE | 4.5125 |
| R² | 0.9495 |

The final regression model achieved an R² score of approximately 0.95, indicating a strong fit on the test dataset.

### Part C — Shipment Delay Classification

Classification models were developed to predict whether a shipment is likely to be delayed.

The workflow includes:
- Model comparison.
- Feature evaluation.
- Classification threshold selection.
- Model evaluation using a confusion matrix and classification metrics.

**Classification results:**

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 85.56% |
| Precision | 65.96% |
| Recall | 75.61% |
| F1-score | 70.45% |
| Selected Threshold | 0.45 |

The selected threshold determines how predicted delay probabilities are converted into delayed and on-time classifications.

### Part D — Logistics Hub Segmentation

K-Means clustering was used to group logistics hubs based on their operational characteristics and delivery performance.

Feature scaling was applied before clustering.

**Hub segmentation results:**

| Segment | Number of Hubs | Average Delivery Time | Delay Rate |
|---|---:|---:|---:|
| Segment 0 | 8 | 56.81 hours | 15.50% |
| Segment 1 | 9 | 64.65 hours | 46.00% |
| Segment 2 | 7 | 47.24 hours | 2.57% |

The segmentation identifies different operational performance groups. Segment 1 has the highest observed delay rate, while Segment 2 has the lowest.

## 7. Model Packaging and Prediction

### Model Packaging

The `package_model.py` script combines the trained regression model, classification model, classification threshold, feature column information, and preprocessing medians into a single reusable model package.

The packaged model is saved as:

```text
models/delay_intelligence_model.pkl
```

### New Shipment Prediction

The `predict_delay.py` script uses the packaged model to process new shipment records.

The prediction workflow:
1. Loads the packaged model.
2. Reads new shipment and hub datasets.
3. Applies the required preprocessing.
4. Predicts delivery duration in hours.
5. Calculates the probability of delay.
6. Classifies shipments as delayed or on time.
7. Saves the prediction results.

The generated output is:

```text
outputs/new_shipment_predictions.csv
```

The prediction file contains shipment identifiers, hub information, predicted delivery hours, delay probabilities, predicted delay classifications, and predicted shipment status.

## 8. How to Run

### Prerequisites

- Python installed on the system.
- Git installed for repository access.
- Required Python libraries listed in `requirements.txt`.

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Generate the Datasets

```bash
python generate_data.py
```

### Step 3 — Run Part A: Data Cleaning and EDA

```bash
python part_a_clean_eda.py
```

### Step 4 — Run Part B: Regression Modelling

```bash
python part_b_regression.py
```

### Step 5 — Run Part C: Classification

```bash
python part_c_classification.py
```

### Step 6 — Run Part D: Hub Segmentation

```bash
python part_d_segments.py
```

### Step 7 — Package the Models

```bash
python package_model.py
```

### Step 8 — Generate New Shipment Predictions

```bash
python predict_delay.py
```

Run the scripts from the project root directory and in the specified order to ensure that the required datasets and model artifacts are available.

## 9. Operations Recommendations

Based on the observed data findings and model outputs, the following operational recommendations can be considered:

1. **Prioritize Express shipments:** Their observed delay rate is the highest among the three service categories. Review dispatch scheduling, handling processes, and delivery capacity for these shipments.

2. **Investigate regional performance:** The East region has a comparatively high delay rate. Further analysis of hub-level operations, shipment volumes, and delivery conditions may help identify contributing factors.

3. **Review underperforming hub segments:** Segment 1 has the highest average delivery time and delay rate. These hubs should be prioritized for operational assessment and process improvement.

4. **Use delay predictions for proactive monitoring:** Shipments with high predicted delay probabilities can be flagged for closer monitoring and early intervention.

5. **Use hub segmentation for resource planning:** Hub clusters can help identify differences in operational performance and support targeted resource allocation.

These recommendations are based on the simulated dataset and should be validated against real operational information before implementation.

## 10. Production Plan

The current project is an analytical prototype. The following stages outline a possible path toward production deployment.

| Phase | Proposed Activity |
|---|---|
| Data Integration | Connect real shipment, hub, traffic, and weather data sources |
| Data Validation | Implement automated checks for missing, inconsistent, and invalid records |
| Model Deployment | Expose trained models through a REST API or cloud-based service |
| Monitoring | Track prediction accuracy, delay trends, and model performance |
| Model Maintenance | Retrain models periodically using updated operational data |
| Dashboard Integration | Develop an interactive dashboard for shipment and hub monitoring |
| Operational Integration | Integrate predictions into logistics monitoring and decision-making workflows |

A production implementation would require real-world data validation, deployment infrastructure, security controls, and continuous model monitoring.

## 11. Limitations

- The project uses simulated shipment data rather than live operational data.
- Model performance on synthetic data may differ from performance on real logistics datasets.
- Predictions depend on the quality and availability of shipment and hub information.
- The project is intended as an analytical and predictive prototype, not a production logistics deployment.
- The operational recommendations require validation using actual logistics data before implementation.

## 12. Future Scope

Potential improvements include:

- Integration with real-time shipment tracking systems.
- Incorporation of live traffic and weather information.
- Development of an interactive logistics monitoring dashboard.
- Automated model retraining using updated shipment data.
- Deployment through a REST API or cloud-based platform.
- Advanced optimization of hub operations and delivery routes.

## 13. Project Timeline

| Dates | Activities |
|---|---|
| 16–19 October 2026 | Data generation, cleaning, preprocessing and EDA |
| 20–23 October 2026 | Regression modelling and evaluation |
| 24–28 October 2026 | Classification modelling and threshold selection |
| 29–31 October 2026 | Hub segmentation and clustering |
| 1–4 November 2026 | Model packaging, prediction integration and testing |
| 5–6 November 2026 | Documentation, repository organization and submission |

## 14. Conclusion

The Delhivery Last-Mile Delay Intelligence Pipeline demonstrates the application of data analytics and machine learning techniques to last-mile logistics operations.

By combining exploratory data analysis, regression, classification, and clustering, the project provides insights into shipment delivery performance, estimates delivery durations, identifies potentially delayed shipments, and groups logistics hubs based on operational characteristics.

The model packaging and prediction workflow further demonstrates how trained models can be reused to evaluate new shipment records without repeating the entire training process.

Overall, this project presents an end-to-end prototype for shipment delay intelligence and data-driven logistics analysis.
