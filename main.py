from data_collector import download_dataset

def run_agent():
    print("\n Starting AutoML Agent...\n")
    file_path, dataset_name = download_dataset()
    
    if file_path:
        print(f"\n Dataset '{dataset_name}' is ready for processing")
    else:
        print(" Dataset not available")

if __name__ == "__main__":
    run_agent()
