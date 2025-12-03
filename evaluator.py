from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, ConfusionMatrixDisplay
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def evaluate_models(trained_models, dataset_name, dataset_goal):
    """
    Evaluates a list of trained models and generates reports and post-training EDA charts.
    
    trained_models: list of tuples (dataset_name, model_name, model, X_test, y_test)
    """
    print("\nEvaluating models...\n")
    results = []

    for ds_name, name, model, X_test, y_test in trained_models:
        if y_test.dtype != "object":
            predictions = model.predict(X_test)
            score = r2_score(y_test, predictions)
            metric = "R² Score"
        else:
            predictions = model.predict(X_test)
            score = accuracy_score(y_test, predictions)
            metric = "Accuracy"

        print(f"{name} {metric}: {round(score, 4)}")
        results.append((ds_name, name, model, score, X_test, y_test))

        # Generate post-training EDA charts for this model
        generate_post_training_eda(model, X_test, y_test, dataset_name=ds_name)

    # Save detailed report
    os.makedirs("reports", exist_ok=True)
    clean_name = dataset_name.replace(" ", "_").lower()
    report_path = f"reports/{clean_name}_report.txt"

    with open(report_path, "w") as f:
        f.write("========== Model Evaluation Report ==========\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Dataset Goal: {dataset_goal}\n")
        f.write("=============================================\n\n")

        f.write("----- Dataset Information -----\n")
        X_test_sample = trained_models[0][3]
        y_test_sample = trained_models[0][4]
        f.write(f"Number of rows: {X_test_sample.shape[0]}\n")
        f.write(f"Number of features: {X_test_sample.shape[1]}\n")
        f.write(f"Target column: {y_test_sample.name if hasattr(y_test_sample, 'name') else 'target'}\n")
        f.write(f"Problem type: {'Regression' if y_test_sample.dtype != 'object' else 'Classification'}\n\n")

        f.write("----- Models and Performance -----\n")
        for ds_name, name, model, score, _, _ in results:
            f.write(f"Model: {name}\n")
            f.write(f"Performance Score: {round(score, 4)}\n")
            f.write("Parameters:\n")
            params = model.get_params()
            for param_key, param_value in params.items():
                f.write(f"  - {param_key}: {param_value}\n")
            f.write("\n")

        f.write("=============================================\n")

    print(f"\nReport saved to: {report_path}")
    return results


def generate_post_training_eda(model, X_test, y_test, dataset_name, output_base="eda_charts"):
    """
    Generates post-training EDA charts for a specific dataset.
    Charts are saved in eda_charts/<dataset_name>/post_training/
    """
    # Clean dataset name for folder
    safe_name = dataset_name.replace(" ", "_").lower()
    output_folder = os.path.join(output_base, safe_name, "post_training")
    os.makedirs(output_folder, exist_ok=True)

    # Feature Importance Plot (for tree-based models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X_test.columns if hasattr(X_test, "columns") else [f"Feature {i}" for i in range(X_test.shape[1])]
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances - {dataset_name}")
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
        plt.title(f"Confusion Matrix - {dataset_name}")
        plt.tight_layout()
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
        plt.title(f"Predicted vs Actual - {dataset_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "predicted_vs_actual.png"))
        plt.close()