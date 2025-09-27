"""
logistic_regression.py

Full example: Binary classification using Logistic Regression (Breast Cancer dataset).
Saves results (confusion matrix, ROC plot, sigmoid curve) to files.

How to run (PowerShell):
  python logistic_regression.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import joblib
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def plot_and_save_roc(y_true, y_prob, filename):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"ROC plot saved to: {filename}")
    plt.close()


def plot_and_save_precision_recall(y_true, y_prob, filename):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Precision-Recall plot saved to: {filename}")
    plt.close()


def sigmoid(x):
    # Numerically-stable sigmoid
    return 1.0 / (1.0 + np.exp(-x))


def plot_and_save_sigmoid(filename):
    x = np.linspace(-10, 10, 400)
    y = sigmoid(x)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.title("Sigmoid Function")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Sigmoid plot saved to: {filename}")
    plt.close()


def evaluate_and_save_reports(y_test, y_pred, y_prob, base_filename):
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(OUTPUT_DIR, f"{base_filename}_confusion_matrix.csv")
    pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_csv(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    print("Confusion Matrix:")
    print(cm)

    # Classification report
    cr = classification_report(y_test, y_pred, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_path = os.path.join(OUTPUT_DIR, f"{base_filename}_classification_report.csv")
    cr_df.to_csv(cr_path)
    print(f"Classification report saved to: {cr_path}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ROC and PR plots
    plot_and_save_roc(y_test, y_prob, os.path.join(OUTPUT_DIR, f"{base_filename}_roc.png"))
    plot_and_save_precision_recall(y_test, y_prob, os.path.join(OUTPUT_DIR, f"{base_filename}_precision_recall.png"))

    # Example: show thresholds for ROC
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    thresh_path = os.path.join(OUTPUT_DIR, f"{base_filename}_roc_thresholds.csv")
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": np.append(roc_thresholds, np.nan)[:len(fpr)]}).to_csv(thresh_path, index=False)
    print(f"ROC thresholds saved to: {thresh_path}")


def tune_threshold_and_show(y_test, y_prob, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Demonstrates how different thresholds change precision/recall and confusion matrix.
    Returns a dict of threshold -> (cm, report)
    """
    results = {}
    for t in thresholds:
        pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, pred_t)
        report = classification_report(y_test, pred_t, output_dict=True)
        print(f"\nThreshold = {t}")
        print("Confusion Matrix:\n", cm)
        print("Precision / Recall / F1 (class 1):",
              report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"])
        results[t] = {"cm": cm, "report": report}
    return results


def main():
    print("Loading dataset...")
    df, skdata = load_data()
    print("Dataset shape:", df.shape)
    print("Features:", len(skdata.feature_names))

    # Quick peek
    print(df.head())

    # Split features/target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = train_model(X_train_scaled, y_train)

    # Save scaler and model
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    model_path = os.path.join(OUTPUT_DIR, "logistic_model.joblib")
    joblib.dump(scaler, scaler_path)
    joblib.dump(model, model_path)
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved model to {model_path}")

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate and save
    evaluate_and_save_reports(y_test, y_pred, y_prob, base_filename="breast_cancer")

    # Sigmoid plot
    plot_and_save_sigmoid(os.path.join(OUTPUT_DIR, "sigmoid.png"))

    # Threshold tuning example
    _ = tune_threshold_and_show(y_test, y_prob, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7])

    print("\nDone. All outputs are in the 'outputs' folder.")


if __name__ == "__main__":
    main()
