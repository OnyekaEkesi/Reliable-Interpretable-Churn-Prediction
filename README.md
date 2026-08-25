# Reliable and Interpretable Bank Customer Churn Prediction

An end-to-end machine learning project for predicting **bank customer churn**, evaluating competing models, interpreting their predictions, and providing uncertainty-aware outputs through an interactive application.

The study compares three tree-based machine learning models — **Random Forest, LightGBM, and XGBoost** — with **XGBoost achieving the best overall performance and being selected for deployment**.

## Project Overview

The project follows an end-to-end machine learning workflow:

**Data Preparation → Exploratory Data Analysis → Model Development → Model Comparison → Evaluation → Interpretability → Uncertainty Analysis → Deployment**

The analysis includes:

- Data preprocessing and exploratory analysis
- Class imbalance handling using SMOTE
- Random Forest, LightGBM and XGBoost modelling
- Hyperparameter tuning
- Model performance evaluation
- SHAP-based model interpretation
- SHAP dependence and Partial Dependence analysis
- Conformal prediction for uncertainty
- Interactive Streamlit deployment

## Application

The final XGBoost model is deployed through a **Streamlit application** designed as a decision-support tool for **bank customer representatives and customer-care teams**.

The application provides:

- **Single Customer Prediction**
- **Batch Customer Prediction**
- Churn probability and risk classification
- Conformal prediction sets
- Local SHAP explanations
- Global SHAP analysis
- SHAP dependence plots
- Partial Dependence Plots (PDP)
- Downloadable prediction results

The application is intended to help customer-care teams identify customers who may be at risk of churn and understand the factors influencing the model's predictions.

> **Note:** Model explanations describe the behaviour of the trained model and should not be interpreted as evidence that individual customer characteristics directly cause churn.

## Model Interpretability

SHAP is used to provide both **local and global explanations** of the XGBoost model.

Local explanations identify the customer characteristics that contribute most strongly to an individual churn prediction.

Global explanations examine the features that have the greatest overall influence on predictions across customers.

SHAP dependence plots and Partial Dependence Plots provide additional analysis of how selected features relate to model behaviour.

## Uncertainty Analysis

**Conformal prediction** is used alongside the model's predicted churn probability to provide an additional indication of prediction uncertainty.

At the configured confidence level, prediction sets may contain:

```text
{Churn}
{No Churn}
{No Churn, Churn}
```

A prediction set containing both classes indicates uncertainty between the two possible outcomes.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/OnyekaEkesi/Reliable-Interpretable-Churn-Prediction.git
cd Reliable-Interpretable-Churn-Prediction
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
```

**macOS/Linux**

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```cmd
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application should open automatically in your browser.

## Input Features

The model uses the following customer characteristics:

- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`

These features are also required for batch prediction files.

## Project Structure

```text
Reliable-Interpretable-Churn-Prediction/
│
├── notebooks/                  # Data analysis, modelling and evaluation
│
├── models/                     # Trained model and supporting artefacts
│   ├── xgb_model.pkl
│   ├── feature_columns.pkl
│   ├── X_calib.pkl
│   └── y_calib.pkl
│
├── assets/                     # Application assets
│   └── logo.png
│
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Deployment runtime
└── README.md
```

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Random Forest
- XGBoost
- LightGBM
- SHAP
- MAPIE
- Matplotlib
- Seaborn
- Streamlit

## Deployment

The final application is deployed using **Streamlit Community Cloud**.

The deployed application uses the trained XGBoost model and supporting artefacts stored in the `models/` directory.

## Author

**Onyekachukwu Ekesi**  
MSc Data Science and Business Analytics

This project was developed as part of an MSc research project on **reliable and interpretable machine learning for bank customer churn prediction**.
