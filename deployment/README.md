# Healthletics Step 8 — Local Deployment

## Overview

This folder contains the local deployment component of the capstone project:

**Healthletics: AI-Powered Member Retention and Churn Prediction System**

The best-performing model, Random Forest, was deployed as a local web application using Flask.

## Deployment Technology

- Python 3
- Flask
- Pandas
- Joblib
- Scikit-learn
- Random Forest Classifier

## Deployment Architecture

User Input → Flask Web Application → Random Forest Model → Churn Probability → Risk Classification

The application accepts member and activity information through a web form and generates a predicted churn probability.

## Model

The deployed model is the Random Forest classifier selected during Step 4 based on its overall predictive performance.

Model configuration:

- Model: Random Forest Classifier
- Number of estimators: 200
- Random state: 42
- Predictor variables: 14

The 14 predictors include membership, demographic, and activity-related variables. The engineered feature `frequency_change` is calculated automatically by the application from current-month and historical class frequency.

## Running the Application

From the deployment folder:

```bash
python3 app.py
