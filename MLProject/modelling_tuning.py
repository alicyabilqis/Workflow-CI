import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
import os
import urllib.request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def download_if_needed(data_path: str, local_filename: str = "dataset.csv") -> str:
    if os.path.exists(data_path):
        print(f"✅ Local file found: {data_path}")
        return data_path
    elif data_path.startswith("http"):
        print(f"⬇️ Downloading dataset from {data_path} ...")
        urllib.request.urlretrieve(data_path, local_filename)
        print(f"✅ Downloaded to: {local_filename}")
        return local_filename
    else:
        raise FileNotFoundError(f"❌ Data path {data_path} is not valid and doesn't exist.")

def main(data_path):
    if not os.path.exists(data_path):
        download_dataset(data_path)

    df = pd.read_csv(data_path)
    X = df.drop("Cover_Type", axis=1)
    y = df["Cover_Type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators_range = [50, 100, 150]
    max_depth_range = [10, 20, 30]
    best_accuracy = 0
    best_params = {}
    best_model = None
    input_example = X_test.iloc[:1]

    # MULAI RUN SECARA MANUAL (WAJIB AGAR LOGGING BERJALAN)
    # Tambahkan ini sebelum loop
    with mlflow.start_run(run_name="Hyperparameter_Tuning"):  # ✅ run utama manual
        for n in n_estimators_range:
            for d in max_depth_range:
                # Nested run tidak dipakai (optional)
                model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)

                mlflow.log_metric(f"accuracy_n{n}_d{d}", acc)  # ✅ Hindari param overwrite
                print(f"Tuning - n_estimators={n}, max_depth={d}, accuracy={acc:.4f}")

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_params = {"n_estimators": n, "max_depth": d}
                    best_model = model

        mlflow.log_params(best_params)
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model_tuned",
            input_example=input_example
        )

    print(f"\nModel terbaik hasil tuning: {best_params} - Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
