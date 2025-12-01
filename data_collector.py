import os
import requests
import random
from datetime import datetime
from config import DATASET_SOURCES


def download_dataset():
    # Choose a dataset at random
    dataset = random.choice(DATASET_SOURCES)

    name = dataset["name"]
    url = dataset["url"]
    goal = dataset.get("goal", "No goal specified")   

    print(f"\n Selected dataset: {name}")
    print(f" Downloading from: {url}")

    # Make sure data folder exists
    os.makedirs("data", exist_ok=True)

    # Create safe filenmae from dataset name with timestamp
    safe_name = name.lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_path = f"data/{safe_name}_{timestamp}.csv"
    # Request the file from the internet
    response = requests.get(url)

    # If successful
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f" Dataset saved successfully: {file_path}")
        return file_path, name, goal

    else:
        print(" Dataset download failed")
        return None, None