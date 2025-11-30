from data_collector import download_dataset
from preprocessor import preprocess_data
from trainer import train_models
from evaluator import evaluate_models
from model_selector import select_and_save_best_model

def run_agent():
    print("\n Starting AutoML Agent...\n")
    file_path, dataset_name = download_dataset()
    
    if file_path:
        X, y = preprocess_data(file_path)
        trained_models = train_models(X, y)
        results = evaluate_models(trained_models, dataset_name)
        select_and_save_best_model(results)

        print("\n AutoML process complete.\n")
    else:
        print(" Dataset not available")

if __name__ == "__main__":
    run_agent()
