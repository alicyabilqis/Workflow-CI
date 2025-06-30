import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
import os
import urllib.request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

def download_from_url(url: str, output_file: str = "dataset.csv") -> str:
    print(f"⬇️ Downloading dataset from {url} ...")
    urllib.request.urlretrieve(url, output_file)
    print(f"✅ Downloaded to: {output_file}")

    # Validasi: cek apakah file benar-benar CSV
    with open(output_file, "r", encoding="utf-8") as f:
        head = f.read(500)
        if "<html" in head.lower():
            raise ValueError("❌ File yang diunduh adalah halaman HTML, bukan file CSV yang valid.")

    return output_file

def main(data_path):
    # Unduh data langsung dari URL
    local_data_path = download_from_url(data_path)

    # Baca dan validasi CSV
    df = pd.read_csv(local_data_path)
    print("✅ Loaded dataset shape:", df.shape)
    print("✅ Columns:", df.columns.tolist())

    if "Cover_Type" not in df.columns:
        raise ValueError("❌ Kolom 'Cover_Type' tidak ditemukan dalam dataset.")

    # Pisahkan fitur dan target
    X = df.drop("Cover_Type", axis=1)
    y = df["Cover_Type"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest + Tuning
    est = RandomForestClassifier(random_state=42)
    params = {
        "n_estimators": [40, 50, 60],
        "max_depth": [12, 14, 16],
    }

    search = RandomizedSearchCV(
        estimator=est,
        param_distributions=params,
        n_iter=9,
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_

    # Evaluasi
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = (0, 0, 0, 0) if conf_matrix.shape != (2, 2) else conf_matrix.ravel()

    # Logging ke MLflow
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)

        if conf_matrix.shape == (2, 2):
            mlflow.log_metric("true_negative", tn)
            mlflow.log_metric("false_positive", fp)
            mlflow.log_metric("false_negative", fn)
            mlflow.log_metric("true_positive", tp)

        mlflow.sklearn.log_model(best_model, "model", input_example=X_test.iloc[:5])

    # Output hasil
    print("✅ Best Parameters:", best_params)
    print(f"✅ Accuracy: {acc}")
    print(f"✅ Recall: {recall}")
    print(f"✅ Precision: {precision}")
    print("✅ Confusion Matrix:\n", conf_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)

print("📍 Current MLflow tracking URI:", mlflow.get_tracking_uri())
