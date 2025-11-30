import pandas as pd

def preprocess_data(file_path):
    print("\n Starting data preprocessing...\n")

    # Load dataset
    df = pd.read_csv(file_path)

    # If dataset has no headers, create some
    if df.columns.tolist()[0].startswith("Unnamed"):
        df.columns = [f"column_{i}" for i in range(len(df.columns))]

    print("Columns detected:", df.columns.tolist())

    # drop rows with missing values
    df = df.dropna()

    # assume last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # encode categorical data
    X = pd.get_dummies(X)

    print("\n Data preprocessing complete")
    return X, y