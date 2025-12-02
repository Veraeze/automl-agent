import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_pre_training_eda(df):
    # Create directory for saving plots
    save_dir = "eda_charts/pre_training"
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
    plt.figure(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.savefig(f"{save_dir}/correlation_heatmap.png")
    plt.close()

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

    perform_pre_training_eda(df)

    # assume last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # encode categorical data
    X = pd.get_dummies(X)

    # Scale the data (VERY important for ML)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n Data preprocessing complete")
    return X_scaled, y