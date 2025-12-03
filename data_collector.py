import os
import requests
import random
from datetime import datetime
from config import DATASET_SOURCES


def download_dataset(dataset_name=None):
    """
    Downloads a dataset.
    If dataset_name is None, picks a random dataset from DATASET_SOURCES.
    Returns: file_path, dataset_name, goal
    """
    # Select dataset
    if dataset_name:
        dataset = next((d for d in DATASET_SOURCES if d["name"] == dataset_name), None)
        if not dataset:
            print(f"Dataset '{dataset_name}' not found in config.")
            return None, None, None
    else:
        dataset = random.choice(DATASET_SOURCES)

    name = dataset["name"]
    url = dataset["url"]
    goal = dataset.get("goal", "No goal specified")   

    print(f"\nSelected dataset: {name}")
    print(f"Downloading from: {url}")

    # Ensure data folder exists
    os.makedirs("data", exist_ok=True)

    # Create safe filename
    safe_name = name.lower().replace(" ", "_")
    file_path = f"data/{safe_name}.csv"

    # Request the file
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Dataset saved successfully: {file_path}")
        return file_path, name, goal
    except Exception as e:
        print(f"Dataset download failed: {e}")
        return None, None, None