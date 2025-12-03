import joblib
import os

def select_and_save_best_model(results):
    """
    Selects the best model (highest score) and saves it.
    """
    # results = list of tuples: (dataset_name, model_name, model, score, X_test, y_test)
    best_model = sorted(results, key=lambda x: x[3], reverse=True)[0]  # sort by score

    dataset_name, model_name, model, score, _, _ = best_model

    # Save the model
    os.makedirs("models", exist_ok=True)
    safe_name = f"{dataset_name.replace(' ', '_').lower()}_{model_name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, f"models/{safe_name}")

    print(f"\nBest model: {model_name} (Score: {round(score, 4)}) for dataset '{dataset_name}'")
    print(f"Model saved to models/{safe_name}")