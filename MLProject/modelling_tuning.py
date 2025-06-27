import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
import os
import urllib.request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

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
    mlflow.set_tracking_uri("file:./mlruns")

    local_data_path = download_if_needed(data_path)
    df = pd.read_csv(local_data_path)

    X = df.drop("Cover_Type", axis=1)
    y = df["Cover_Type"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    est = RandomForestClassifier(random_state=42)
    params = {
        "n_estimators": [50, 100, 150],
        "max_depth": [10, 20, 30],
    }

    search = RandomizedSearchCV(
        estimator=est,
        param_distributions=params,
        n_iter=20,
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = (0, 0, 0, 0) if conf_matrix.shape != (2, 2) else conf_matrix.ravel()

    with mlflow.start_run(run_name="Hyperparameter_Tuning"):
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        if conf_matrix.shape == (2, 2):
            mlflow.log_metric("true_negative", tn)
            mlflow.log_metric("false_positive", fp)
            mlflow.log_metric("false_negative", fn)
            mlflow.log_metric("true_positive", tp)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_test.iloc[:5]
        )

    print("Best Parameters:", best_params)
    print(f"Accuracy: {acc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print("Confusion Matrix:\n", conf_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
