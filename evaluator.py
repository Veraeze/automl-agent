from sklearn.metrics import accuracy_score, r2_score
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_models(trained_models, dataset_name, dataset_goal):
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
        # Header Section
        f.write("========== Model Evaluation Report ==========\n")
        f.write(f"Report generated on: {timestamp}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Dataset Goal: {dataset_goal}\n")
        f.write("=============================================\n\n")

        # Dataset Information
        f.write("----- Dataset Information -----\n")
        f.write(f"Number of rows: {X_test.shape[0]}\n")
        f.write(f"Number of features: {X_test.shape[1]}\n")
        f.write(f"Target column: {y_test.name if hasattr(y_test, 'name') else 'target'}\n")
        f.write(f"Problem type: {'Regression' if y_test.dtype != 'object' else 'Classification'}\n\n")

        # Preprocessing Steps (Placeholder)
        f.write("----- Preprocessing Steps -----\n")
        f.write("Data was split into training and testing sets.\n")
        f.write("Features were used as-is without additional preprocessing.\n\n")

        # Model Performance
        f.write("----- Models and Performance -----\n")
        for name, model, score in results:
            f.write(f"Model: {name}\n")
            f.write(f"Performance Score: {round(score,4)}\n")
            f.write(f"Parameters:\n")
            params = model.get_params()
            for param_key, param_value in params.items():
                f.write(f"  - {param_key}: {param_value}\n")
            f.write("\n")

        f.write("=============================================\n")

        print(f"\n Report saved to: {report_path}")

    return results

def generate_post_training_eda(model, X_test, y_test, output_folder="eda_charts/post_training"):
    os.makedirs(output_folder, exist_ok=True)

    # Feature Importance Plot
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X_test.columns if hasattr(X_test, "columns") else [f"Feature {i}" for i in range(X_test.shape[1])]
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "feature_importances.png"))
        plt.close()

    # Classification: Confusion Matrix
    if y_test.dtype == "object":
        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
        plt.close()

    # Regression: Predicted vs Actual Scatter Plot
    else:
        predictions = model.predict(X_test)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "predicted_vs_actual.png"))
        plt.close()