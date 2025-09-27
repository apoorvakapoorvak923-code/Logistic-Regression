# Logistic-Regression
# Logistic Regression - Breast Cancer Classification

## 📝 About
This project demonstrates the implementation of **Logistic Regression** for a **binary classification problem** using the **Breast Cancer Wisconsin dataset** (`sklearn.datasets`).  

The goal is to classify whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)** based on 30 numerical medical features (e.g., radius, texture, smoothness).  

Logistic Regression is a supervised learning algorithm widely used for classification. 
It uses the **sigmoid function** to estimate probabilities and helps evaluate performance through metrics like **Precision, Recall, F1-Score, ROC Curve, and AUC**.  
This repository provides a **complete end-to-end ML pipeline** — from data preparation to model training, evaluation, visualization, and threshold tuning.  

## ✨ Features
- **Data Handling**
  - Loads Breast Cancer dataset (569 samples, 30 features).
  - Splits dataset into **train/test (80/20)**.
  - Standardizes features with `StandardScaler`.

- **Model Training**
  - Logistic Regression model (`max_iter=1000`).
  - Saves trained **model** and **scaler** (`joblib`).

- **Evaluation Metrics**
  - Confusion Matrix (CSV).
  - Classification Report (Precision, Recall, F1, Accuracy).
  - ROC Curve + AUC (PNG).
  - Precision-Recall Curve (PNG).
  - ROC Threshold values (CSV).

- **Visualization**
  - Sigmoid function plot.
  - ROC and Precision-Recall curves.

- **Threshold Tuning**
  - Compares performance at thresholds `[0.3, 0.4, 0.5, 0.6, 0.7]`.
  - Shows changes in confusion matrix and Precision/Recall.

- **Outputs Saved Automatically**
  - All results (plots, CSVs, models) saved to the `outputs/` folder.

## 📂 Project Structure
