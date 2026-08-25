# Bank Customer Churn Prediction

An interpretable machine learning application for predicting **bank customer churn** and supporting customer-care decision-making.

The project uses a trained **LightGBM classifier** to estimate customer churn risk, **SHAP** to explain model predictions, and **conformal prediction** to provide additional information about prediction uncertainty. The model is deployed through an interactive **Streamlit application**.

## Key Features

- **Single Customer Prediction** — predict churn risk for an individual customer.
- **Batch Prediction** — upload a CSV file and analyse multiple customers.
- **Churn Probability** — view the estimated probability of customer churn.
- **Conformal Prediction** — communicate uncertainty through prediction sets.
- **Local SHAP Explanations** — understand the factors influencing an individual prediction.
- **Global SHAP Analysis** — identify influential features across multiple customers.
- **SHAP Dependence Plots** — examine how individual features influence model predictions.
- **Partial Dependence Plots (PDP)** — examine the model's average response to changes in selected features.
- **Downloadable Results** — export batch predictions as a CSV file.

## Application Purpose

The application is designed primarily as a **decision-support tool for bank customer representatives and customer-care teams**.

It helps users answer three practical questions:

1. **Is this customer at risk of churn?**
2. **How high is the estimated churn risk?**
3. **Which customer characteristics influenced the prediction?**

Model explanations describe the behaviour of the predictive model and should not be interpreted as evidence that individual customer characteristics directly cause churn.

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

### 3. Activate the environment

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

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application should open automatically in your browser.

## Required Input Features

The model uses the following customer information:

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

Batch prediction files must contain these columns.

## Model Interpretability

The application provides both **local and global explanations**.

**Local explanations** use SHAP to show which features increase or reduce the estimated churn risk for an individual customer.

**Global explanations** use SHAP feature importance, beeswarm plots, dependence plots, and partial dependence plots to examine the model's behaviour across multiple customers.

## Uncertainty Analysis

Conformal prediction is used alongside the model's churn probability.

At the configured **95% confidence level**, the prediction set may contain:

```text
{Churn}
{No Churn}
{No Churn, Churn}
```

A set containing both classes indicates greater uncertainty between the two possible outcomes.

## Project Structure

```text
Reliable-Interpretable-Churn-Prediction/
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── models/
│   ├── lgbm_model.pkl
│   ├── feature_columns.pkl
│   ├── X_calib.pkl
│   └── y_calib.pkl
│
├── assets/
│   └── logo.png
│
└── README.md
```

## Technology Stack

- Python
- Streamlit
- LightGBM
- SHAP
- MAPIE
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Deployment

The application is deployed using **Streamlit Community Cloud**.

The main application entry point is:

```text
app.py
```

Dependencies required for deployment are specified in:

```text
requirements.txt
```

## Author

**Onyekachukwu Ekesi**  
MSc Data Science and Business Analytics

This project was developed as part of an MSc research project on **bank customer churn prediction, model interpretability, uncertainty-aware prediction, and decision support**.
