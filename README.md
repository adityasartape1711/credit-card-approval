# credit-card-approval
An end-to-end machine learning project designed to identify high-risk credit card applicants. The system utilizes personal and financial demographic data alongside past payment histories to predict whether an applicant should be approved or rejected.
igned to identify high-risk credit card applicants. The system utilizes personal and financial demographic data alongside past payment histories to predict whether an applicant should be approved or rejected.

##  Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Pipeline & Workflow](#pipeline--workflow)
- [Model Performance](#model-performance)
- [Feature Importances](#feature-importances)
- [How to Run & Deploy](#how-to-run--deploy)
- [Requirements](#requirements)

---

##  Project Overview
Financial institutions need robust systems to assess creditworthiness and control default rates. This repository builds a machine learning pipeline that:
1. Consolidates application details with historical payment records.
2. Defines a target label mapping risk profile based on historical payment behavior.
3. Handles extreme class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
4. Trains a **Random Forest Classifier** to flag risky applicants.
5. Saves the model and preprocessing artifacts for real-time inference.

##  Dataset Description
The analysis utilizes two main relational datasets:
1. **`application_record.csv`**: Contains client demographic attributes (e.g., gender, income, assets, education, occupation, age, employment length).
2. **`credit_record.csv`**: Contains monthly payment tracking history with status indicators:
   - `C`: Paid off that month
   - `X`: No debt for that month
   - `0`: 1 to 29 days overdue
   - `1` to `5`: Overdue ranging from 30 days to over 150 days.

### Label Definition
Applicants with payment delays of 30 days or more (`STATUS` 1-5) at any point in their record are defined as **High Risk/Bad (1)**. Others are defined as **Approved/Good (0)**.

## 🛠️ Pipeline & Workflow
- **Step 1 - Data Integration**: Datasets are loaded and joined using the unique applicant identifier (`ID`).
- **Step 2 - Cleaning & De-duplication**: Features-based duplicates are removed, and missing categorical attributes (`OCCUPATION_TYPE`) are filled.
- **Step 3 - Feature Engineering**: Date-based records (`DAYS_BIRTH`, `DAYS_EMPLOYED`) are converted into human-readable years. Categorical variables are converted into integers via Label Encoding.
- **Step 4 - Handling Imbalance**: Implemented **SMOTE** on scaled features to re-balance the highly skewed credit classification target.
- **Step 5 - Model Training**: Developed and evaluated a Random Forest Classifier.
- **Step 6 - Export**: Serialized both the trained estimator and the standard scaler object into `.pkl` format for production.

##  Model Performance

### Without Class Balancing
- **Accuracy**: ~67%
- **Recall (Class 1 - Rejected)**: Very low (~3%) due to heavy class skew.

### With SMOTE Resampling
- **Accuracy**: ~65%
- **Overall f1-score** and predictive stability on minority class instances were boosted, establishing a more risk-averse model suitable for deployment.

##  Feature Importances
The top predictors identified by the model during analysis:
1. **Total Income Amount (`AMT_INCOME_TOTAL`)**
2. **Applicant Age (`AGE`)**
3. **Years of Employment (`YEARS_EMPLOYED`)**
4. **Occupation Type (`OCCUPATION_TYPE`)**

##  How to Run & Deploy

### 1. Project Directory Structure
```text
├── application_record.csv
├── credit_record.csv
├── credit_card_model.pkl
├── scaler.pkl
├── main.ipynb
└── README.md
```

### 2. Loading Artifacts & In-memory Testing
```python
import joblib
import pandas as pd

# Load saved model and scaler
loaded_model = joblib.load('credit_card_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Input applicant features as a DataFrame and predict:
# scaled_features = loaded_scaler.transform(new_data_df)
# prediction = loaded_model.predict(scaled_features)
```

##  Requirements
To run the notebook and scripts locally, install:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn joblib
