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

## Project Overview

The project addresses the problem of predicting whether bank customers are likely to leave the institution.

The machine learning model uses customer-level information to estimate churn risk. The resulting predictions are presented through an interactive web application that allows users to analyse individual customers or process a larger customer dataset.

The application is designed around four main objectives:

- **Prediction** — estimate the likelihood that a customer will churn.
- **Interpretability** — explain the factors influencing individual and overall model predictions.
- **Uncertainty awareness** — provide conformal prediction sets alongside model probabilities.
- **Decision support** — help customer-care teams identify customers who may require closer attention.

The application is not intended to establish that a particular customer characteristic directly causes churn. Model explanations describe how the trained model uses the available customer characteristics when generating predictions.

---

## Key Features

### Individual Customer Prediction

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

### Batch Customer Prediction

Users can upload a CSV file containing multiple customer records and obtain:

- Predictions for each customer
- Churn probabilities
- Conformal prediction sets
- Prediction summary statistics
- Customer filtering
- Downloadable prediction results

### Model Interpretability

The application provides several interpretability methods:

- Local SHAP explanations
- Global SHAP feature importance
- SHAP beeswarm analysis
- SHAP dependence analysis
- Partial dependence analysis

These provide complementary views of how the model uses customer characteristics when predicting churn.

### Uncertainty Analysis

The application uses conformal prediction to provide prediction sets at a 95% confidence level.

A prediction set may contain:

- `{Churn}`
- `{No Churn}`
- `{No Churn, Churn}`

A set containing both classes indicates that the conformal prediction does not provide a single-class prediction at the specified confidence level.

---

## Getting Started

These instructions explain how to set up and run the project locally.

## Prerequisites

Before running the project, ensure that Python is installed on your computer.

The project is developed using Python 3.12.

Python can be downloaded from:

https://www.python.org/

A virtual environment is recommended so that project dependencies remain isolated from other Python projects.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/OnyekaEkesi/Reliable-Interpretable-Churn-Prediction.git
