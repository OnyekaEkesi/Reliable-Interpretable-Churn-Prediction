# Bank Customer Churn Prediction

## Introduction

This project develops an interpretable machine learning system for predicting **bank customer churn** and presenting the predictions through an interactive Streamlit application.

The project combines machine learning-based churn prediction with model interpretability and uncertainty-aware outputs. The application is designed primarily for **bank customer representatives and customer-care teams**, providing both individual and batch prediction capabilities.

For individual customers, the application provides a churn prediction, estimated churn probability, conformal prediction set, and local SHAP explanations showing which customer characteristics most influenced the prediction.

For multiple customers, the application provides batch predictions, conformal prediction results, downloadable outputs, and global model interpretation using SHAP feature importance, beeswarm plots, SHAP dependence plots, and partial dependence plots.

The overall objective is to provide a practical decision-support tool that can help customer-care teams identify customers who may be at elevated risk of churn and better understand the factors underlying model predictions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running the Application](#running-the-application)
4. [Using the Streamlit Application](#using-the-streamlit-application)
   - [Single Customer Prediction](#single-customer-prediction)
   - [Batch Customer Prediction](#batch-customer-prediction)
5. [Model Interpretability](#model-interpretability)
   - [Local SHAP Explanations](#local-shap-explanations)
   - [Global SHAP Analysis](#global-shap-analysis)
   - [SHAP Dependence Analysis](#shap-dependence-analysis)
   - [Partial Dependence Analysis](#partial-dependence-analysis)
6. [Uncertainty and Conformal Prediction](#uncertainty-and-conformal-prediction)
7. [Input Data](#input-data)
8. [Project Organisation](#project-organisation)
9. [Project Lifecycle](#project-lifecycle)
10. [Deployment](#deployment)
11. [Contributing](#contributing)
12. [License](#license)
13. [Author](#author)

---

# Project Overview

The project addresses the problem of predicting whether bank customers are likely to leave the institution.

The machine learning model uses customer-level information to estimate churn risk. The resulting predictions are presented through an interactive web application that allows users to analyse individual customers or process a larger customer dataset.

The application is designed around four main objectives:

- **Prediction** — estimate the likelihood that a customer will churn.
- **Interpretability** — explain the factors influencing individual and overall model predictions.
- **Uncertainty awareness** — provide conformal prediction sets alongside model probabilities.
- **Decision support** — help customer-care teams identify customers who may require closer attention.

The application is not intended to establish that a particular customer characteristic directly causes churn. Model explanations describe how the trained model uses the available customer characteristics when generating predictions.

---

# Key Features

## Individual Customer Prediction

Users can enter information for an individual bank customer and obtain:

- Churn prediction
- Estimated churn probability
- Churn risk classification
- Conformal prediction set
- Local SHAP explanation
- Key factors influencing the prediction
- Detailed SHAP waterfall plot
- Technical feature contribution information
- Factors increasing and reducing estimated churn risk

---

## Batch Customer Prediction

Users can upload a CSV file containing multiple customer records and obtain:

- Predictions for each customer
- Churn probabilities
- Conformal prediction sets
- Prediction summary statistics
- Customer filtering
- Downloadable prediction results

The batch functionality is particularly useful when analysing a larger customer population and identifying customers who may require closer attention from customer-care teams.

---

# Model Interpretability

Interpretability is a central component of the application because the system is intended to support customer-care decision-making rather than simply produce a binary prediction.

The application therefore provides both **local** and **global** model explanations.

---

## Local SHAP Explanations

For an individual customer, SHAP is used to identify the customer characteristics that contributed most strongly to the model's prediction.

The application provides:

- Key factors influencing the prediction
- Factors increasing estimated churn risk
- Factors reducing estimated churn risk
- A SHAP waterfall plot
- A technical feature contribution table

### SHAP Waterfall Plot

The waterfall plot provides a detailed view of how individual feature contributions move the prediction away from the model's baseline.

Positive contributions indicate movement toward the churn prediction, while negative contributions indicate movement away from churn.

These explanations describe the behaviour of the model and should not be interpreted as evidence that a particular characteristic directly causes customer churn.

---

## Global SHAP Analysis

For batch predictions, SHAP is calculated across the uploaded customer population.

The application provides a global feature importance analysis based on mean absolute SHAP values.

This identifies the characteristics that have the greatest overall influence on the model's predictions within the analysed customer dataset.

### SHAP Beeswarm Plot

The beeswarm plot provides a more detailed view of how individual feature values are associated with changes in model output across customers.

It is intended primarily for users who require a more technical understanding of model behaviour.

---

## SHAP Dependence Analysis

The SHAP dependence plot allows a selected feature to be examined across the customer population.

It shows how the value of an individual feature is associated with its SHAP contribution to the model's predictions.

This provides a more detailed view of how the model's use of a particular feature varies across customers.

---

## Partial Dependence Analysis

The Partial Dependence Plot (PDP) provides a complementary view of model behaviour.

It shows how the model's average predicted response changes as a selected feature varies while averaging over the other features in the analysed dataset.

PDP analysis describes model behaviour and should not be interpreted as evidence that changing a particular customer characteristic will necessarily cause churn or prevent it.

---

# Uncertainty and Conformal Prediction

The application uses conformal prediction to complement the model's churn probability.

The conformal prediction component is calibrated using a separate calibration dataset and configured with a **95% confidence level**.

For an individual customer, the prediction set may contain:

```text
{Churn}

{No Churn}

or:

{No Churn, Churn}

A prediction set containing both classes indicates uncertainty between the two possible outcomes at the selected confidence level.

For batch predictions, the conformal prediction set is provided for every customer and summarised in the application. This provides an additional uncertainty-aware output alongside the model's predicted probability.

---

## Input Data

For batch prediction, the application accepts CSV files.

The following columns are required:

```text
CreditScore
Geography
Gender
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary

After entering the customer's information, select:

**Predict Customer Churn**

The application processes the customer's information using the same feature structure used by the trained model and generates the prediction.

---

### Prediction Result

The application displays:

- Whether the customer is predicted to churn
- Estimated probability of churn
- Risk classification
- Conformal prediction set

The probability represents the model's estimated likelihood of churn and should not be interpreted as certainty that the customer will leave the bank.

---

## Batch Customer Prediction

The Batch Prediction interface allows users to upload a CSV file containing multiple customer records.

After the file is uploaded, the application:

1. Reads and validates the dataset.
2. Checks that all required columns are present.
3. Applies the required preprocessing.
4. Aligns the input features with the trained model.
5. Generates churn predictions.
6. Calculates churn probabilities.
7. Generates conformal prediction sets.
8. Displays summary statistics.
9. Provides filtering options.
10. Allows the results to be downloaded as a CSV file.
11. Provides global model interpretation for the uploaded customer population.

---

### Batch Prediction Summary

The application reports:

- Total number of customers
- Number predicted to churn
- Number predicted not to churn
- Number receiving a conformal churn prediction
