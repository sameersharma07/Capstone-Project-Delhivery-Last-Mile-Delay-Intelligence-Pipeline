# Capstone Project — Delhivery Last-Mile Delay Intelligence Pipeline

**Project Period:** 16 October 2026 – 6 November 2026  
**Language:** Python  
**Development Environments:** Google Colab and Cursor  
**Repository:** [Capstone Project — Delhivery Last-Mile Delay Intelligence Pipeline](https://github.com/sameersharma07/Capstone-Project-Delhivery-Last-Mile-Delay-Intelligence-Pipeline)

---

## 1. Project Overview

The **Delhivery Last-Mile Delay Intelligence Pipeline** is a data analytics and machine learning project designed to analyze shipment delivery performance, identify factors contributing to delivery delays, predict delivery times, classify delayed shipments, and group logistics hubs based on their operational performance.

The project uses a simulated shipment dataset representing last-mile delivery operations. It combines data preprocessing, exploratory data analysis, regression, classification, and clustering into a single end-to-end analytical pipeline.

The final objective is to develop a reusable model package that can generate delivery-time estimates and delay predictions for new shipments.

## 2. Project Objectives

- Generate and prepare shipment and logistics hub datasets.
- Clean and preprocess raw shipment data.
- Perform exploratory data analysis (EDA) to identify delivery patterns.
- Analyze the factors influencing delivery duration and delays.
- Build regression models to predict delivery hours.
- Build classification models to identify potentially delayed shipments.
- Apply clustering techniques to identify operational hub segments.
- Package trained models and preprocessing information for reuse.
- Generate predictions for previously unseen shipments.

## 3. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Google Colab | Development and model experimentation |
| Cursor | Code editing and project organization |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and model evaluation |
| Joblib | Model serialization and packaging |

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
│   ├── EDA outputs
│   ├── Regression outputs
│   ├── Classification outputs
│   ├── Hub segmentation outputs
│   └── Prediction outputs
│
├── generate_data.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
│
├── requirements.txt
└── README.md
```

*Note: The outputs directory contains the generated CSV reports and visualization files. The models directory contains the serialized model artifacts.*

## 5. Dataset Description

The project uses three primary input datasets and one processed dataset.

| Dataset | Description |
|---|---|
| `hubs.csv` | Contains logistics hub information and operational attributes |
| `raw_shipments.csv` | Contains the generated shipment records used for analysis and training |
| `new_shipments.csv` | Contains new shipment records used for prediction |
| `modelling_table.csv` | Contains the cleaned and feature-engineered dataset prepared for modelling |

The synthetic dataset generator creates 24 logistics hubs, 1,200 shipment records, and 48 new shipment records for prediction.

## 6. Project Workflow

### Part A — Data Generation, Cleaning and Exploratory Data Analysis

- Generate shipment and hub datasets.
- Inspect the raw data for missing values, inconsistencies, and duplicate records.
- Clean and preprocess shipment information.
- Perform feature engineering and prepare the modelling dataset.
- Conduct exploratory data analysis to understand delivery patterns.
- Generate summary tables and visualizations.

**Key observations:**

- Express shipments recorded a delay rate of approximately 48.97%.
- Standard shipments recorded a delay rate of approximately 20.56%.
- Economy shipments recorded a delay rate of approximately 8.57%.
- The East region recorded a delay rate of approximately 43.25%.

### Part B — Delivery Time Prediction Using Regression

Regression models were developed to estimate shipment delivery duration in hours.

The project compares multiple regression approaches, including:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

**Final regression test results:**

| Metric | Result |
|---|---:|
| MAE | 3.4711 |
| RMSE | 4.5125 |
| R² | 0.9495 |

The final regression model achieved an R² score of approximately 0.95 on the test dataset.

### Part C — Shipment Delay Classification

Classification models were developed to predict whether a shipment is likely to be delayed.

The classification workflow includes model comparison, evaluation, threshold selection, and final prediction.

The evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

**Classification results:**

| Metric | Result |
|---|---:|
| Accuracy | 85.56% |
| Precision | 65.96% |
| Recall | 75.61% |
| F1-score | 70.45% |
| Classification Threshold | 0.45 |

The selected classification threshold provides a balance between identifying delayed shipments and limiting false delay predictions.

### Part D — Logistics Hub Segmentation

Clustering was applied to group logistics hubs based on their operational characteristics and delivery performance.

The segmentation process uses feature scaling and K-Means clustering.

**Hub segment summary:**

| Segment | Number of Hubs | Average Delivery Time | Delay Rate |
|---|---:|---:|---:|
| Segment 0 | 8 | 56.81 hours | 15.50% |
| Segment 1 | 9 | 64.65 hours | 46.00% |
| Segment 2 | 7 | 47.24 hours | 2.57% |

The results identify different operational performance groups, helping highlight hubs that may require further investigation or process improvement.

## 7. Model Packaging and Prediction

The project includes two additional scripts for model reuse and prediction.

### Model Packaging — `package_model.py`

This script combines the trained regression model, classification model, selected classification threshold, feature column information, and preprocessing medians into a single serialized model package.

The packaged model is saved as:

```text
models/delay_intelligence_model.pkl
```

### New Shipment Prediction — `predict_delay.py`

The prediction script:

1. Loads the packaged model.
2. Reads new shipment records and hub information.
3. Applies the same preprocessing logic used during training.
4. Predicts delivery duration in hours.
5. Calculates the probability of shipment delay.
6. Classifies shipments as delayed or on time using the selected threshold.
7. Saves the prediction results.

The generated prediction file is:

```text
outputs/new_shipment_predictions.csv
```

The output includes shipment identifiers, hub information, predicted delivery hours, delay probability, predicted delay classification, and predicted status.

## 8. Installation and Setup

Clone the repository:

```bash
git clone https://github.com/sameersharma07/Capstone-Project-Delhivery-Last-Mile-Delay-Intelligence-Pipeline.git
```

Navigate to the project directory:

```bash
cd Capstone-Project-Delhivery-Last-Mile-Delay-Intelligence-Pipeline
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 9. Running the Project

Run the scripts from the repository root directory.

**Step 1 — Generate the datasets**

```bash
python generate_data.py
```

**Step 2 — Run Part A: Data Cleaning and EDA**

```bash
python part_a_clean_eda.py
```

**Step 3 — Run Part B: Regression Modelling**

```bash
python part_b_regression.py
```

**Step 4 — Run Part C: Classification**

```bash
python part_c_classification.py
```

**Step 5 — Run Part D: Hub Segmentation**

```bash
python part_d_segments.py
```

**Step 6 — Package the trained models**

```bash
python package_model.py
```

**Step 7 — Predict delays for new shipments**

```bash
python predict_delay.py
```

*Run the scripts in the specified order so that the required datasets, model artifacts, and output directories are available.*

## 10. Expected Outputs

The project generates:

- Cleaned and feature-engineered shipment datasets.
- Exploratory data analysis summaries and visualizations.
- Regression model comparison and evaluation reports.
- Classification metrics, reports, and confusion matrices.
- Hub segmentation and cluster analysis results.
- Serialized machine learning models.
- A consolidated model package.
- New shipment delivery-time and delay predictions.

Generated CSV reports and visualizations are stored in `outputs/`, while serialized models are stored in `models/`.

## 11. Limitations

- The project uses simulated shipment data rather than live operational data.
- Model performance on synthetic data may differ from performance on real logistics datasets.
- Predictions depend on the quality and availability of shipment and hub information.
- The project is intended as an analytical and predictive prototype, not a production logistics deployment.

## 12. Future Scope

Potential improvements include:

- Integration with real-time shipment tracking systems.
- Incorporation of live traffic and weather information.
- Development of an interactive dashboard for logistics monitoring.
- Automated model retraining using updated shipment data.
- Deployment through a REST API or cloud-based service.
- Advanced optimization of hub operations and delivery routes.

## 13. Project Timeline

| Dates | Work Completed |
|---|---|
| 16–19 October 2026 | Data generation, cleaning, preprocessing and EDA |
| 20–23 October 2026 | Regression modelling and evaluation |
| 24–28 October 2026 | Classification modelling and threshold selection |
| 29–31 October 2026 | Hub segmentation and clustering |
| 1–4 November 2026 | Model packaging, prediction integration and testing |
| 5–6 November 2026 | Documentation, repository organization and submission |

## 14. Conclusion

The Delhivery Last-Mile Delay Intelligence Pipeline demonstrates how data analytics and machine learning can be applied to last-mile logistics operations.

The project combines exploratory analysis, regression, classification, and clustering to examine shipment delivery performance, estimate delivery duration, identify potentially delayed shipments, and segment logistics hubs.

The final model packaging and prediction workflow makes it possible to reuse the trained models for new shipment records without repeating the entire training process.

Overall, the project provides an end-to-end prototype demonstrating the application of Python-based data science and machine learning techniques to logistics delay intelligence.

---

**Author:** Sameer Sharma  
**Project:** Delhivery Last-Mile Delay Intelligence Pipeline  
**Repository:** [GitHub Repository](https://github.com/sameersharma07/Capstone-Project-Delhivery-Last-Mile-Delay-Intelligence-Pipeline)
