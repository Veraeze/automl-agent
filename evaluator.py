from sklearn.metrics import accuracy_score, r2_score
import os
from datetime import datetime

def evaluate_models(trained_models, dataset_name):
    print("\n Evaluating models...\n")

    results = []

    for name, model, X_test, y_test in trained_models:

        if y_test.dtype != "object":
            predictions = model.predict(X_test)
            score = r2_score(y_test, predictions)
            metric = "R² Score"
        else:
            predictions = model.predict(X_test)
            score = accuracy_score(y_test, predictions)
            metric = "Accuracy"

        print(f" {name} {metric}: {round(score, 4)}")
        results.append((name, model, score))
    
    # Save detailed report
    os.makedirs("reports", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = f"reports/report_{timestamp}.txt"

    with open(report_path, "w") as f:
        # Dataset info
        f.write(f"Dataset: {dataset_name}\n")  # Pass dataset_name from main.py
        f.write(f"Number of rows: {X_test.shape[0]}\n")
        f.write(f"Number of features: {X_test.shape[1]}\n")
        f.write(f"Target column: {y_test.name if hasattr(y_test, 'name') else 'target'}\n")
        f.write(f"Problem type: {'Regression' if y_test.dtype != 'object' else 'Classification'}\n\n")

        f.write("--- Models and Performance ---\n")
        for name, model, score in results:
            f.write(f"{name}\n")
            f.write(f"Score: {round(score,4)}\n")
            f.write(f"Parameters: {model.get_params()}\n\n")

        f.write(f"Report generated on: {timestamp}\n")

        print(f"\n Report saved to: {report_path}")

    return results

   