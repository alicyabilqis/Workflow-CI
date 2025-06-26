# modelling_tuning.py {FIXX}

import numpy as np

# Hyperparameter ranges
n_estimators_range = [50, 100, 150]
max_depth_range = [10, 20, 30]

best_accuracy = 0
best_params = {}
best_model = None

with mlflow.start_run(run_name="Hyperparameter Tuning"):
    for n in n_estimators_range:
        for d in max_depth_range:
            with mlflow.start_run(nested=True):
                model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)

                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", d)
                mlflow.log_metric("accuracy", acc)

                print(f"Tuning - n_estimators={n}, max_depth={d}, accuracy={acc:.4f}")

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_params = {"n_estimators": n, "max_depth": d}
                    best_model = model

    # Log model terbaik
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model_tuned",
        input_example=input_example
    )
    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_params(best_params)

    print(f"\nModel terbaik hasil tuning: {best_params} - Accuracy: {best_accuracy:.4f}")
