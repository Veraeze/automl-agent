from data_collector import download_dataset
from preprocessor import preprocess_data
from trainer import train_models
from evaluator import evaluate_models
from model_selector import select_and_save_best_model
from config import DATASET_SOURCES

def run_agent():
    print("\nStarting AutoML Agent...\n")

    for dataset_cfg in DATASET_SOURCES:
        file_path, dataset_name, dataset_goal = download_dataset(dataset_cfg["name"])
        if file_path:
            X, y = preprocess_data(file_path)
            trained_models = train_models(X, y, dataset_name)
            results = evaluate_models(trained_models, dataset_name, dataset_goal)
            select_and_save_best_model(results)
            print("\n-----------------------------\n")
        else:
            print(f"Dataset {dataset_name} not available")

if __name__ == "__main__":
    run_agent()