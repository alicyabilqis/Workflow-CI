import argparse
import os
import urllib.request
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="Forest cover_preprocessing_dataset.csv")
args = parser.parse_args()

# Download dataset jika belum ada
if not os.path.exists(args.data_path):
    print(f"{args.data_path} not found. Downloading...")
    url = "https://drive.google.com/uc?export=download&id=1u4eL5GYTfv5AWZ0PUWNP2N9EEpn6n-NS"
    urllib.request.urlretrieve(url, args.data_path)
    print(f"Downloaded to {args.data_path}")

# Load data
data = pd.read_csv(args.data_path)
X = data.drop("Cover_Type", axis=1)
y = data["Cover_Type"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter ranges
n_estimators_range = [50, 100, 150]
max_depth_range = [10, 20, 30]  # ❗️fixed typo: 3 → 30

best_accuracy = 0
best_params = {}
best_model = None

# Input example untuk model signature
input_example = X_test.iloc[:1]

# Set experiment name
mlflow.set_experiment("Forest_Cover_Tuning")

# ✅ Start a single run
with mlflow.start_run(run_name="Hyperparameter Tuning"):
    for n in n_estimators_range:
        for d in max_depth_range:
            model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)

            # Log semua kombinasi sebagai metric, bukan param
            mlflow.log_metric(f"acc_n{n}_d{d}", acc)

            print(f"Tuning - n_estimators={n}, max_depth={d}, accuracy={acc:.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {"n_estimators": n, "max_depth": d}
                best_model = model

    # Log best result
    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", best_accuracy)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model_tuned",
        input_example=input_example
    )

    print(f"\n✅ Model terbaik: {best_params} - Accuracy: {best_accuracy:.4f}")
