import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_pre_training_eda(df, dataset_name):
    # Clean dataset name for folder
    safe_name = dataset_name.replace(" ", "_").lower()
    save_dir = os.path.join("eda_charts", safe_name, "pre_training")
    os.makedirs(save_dir, exist_ok=True)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Histograms
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Histogram of {col}')
        plt.savefig(f"{save_dir}/histogram_{col}.png")
        plt.close()

    # Boxplots
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col].dropna())
        plt.title(f'Boxplot of {col}')
        plt.savefig(f"{save_dir}/boxplot_{col}.png")
        plt.close()

    # Correlation heatmap
    if numeric_cols:
        plt.figure(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Heatmap')
        plt.savefig(f"{save_dir}/correlation_heatmap.png")
        plt.close()


def preprocess_data(file_path):
    print("\nStarting data preprocessing...\n")

    # Try reading CSV with comma, fallback to semicolon
    try:
        df = pd.read_csv(file_path)
        if len(df.columns) == 1:
            df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

    # Handle missing headers
    if df.columns.tolist()[0].startswith("Unnamed"):
        df.columns = [f"column_{i}" for i in range(len(df.columns))]

    print("Columns detected:", df.columns.tolist())

    # Drop rows with missing values
    df = df.dropna()

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    perform_pre_training_eda(df, dataset_name)

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode categorical features
    X = pd.get_dummies(X)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nData preprocessing complete")
    return X_scaled, y