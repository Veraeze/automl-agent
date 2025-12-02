import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="AutoML Dashboard", layout="wide")
st.title("AutoML Agent Dashboard")

# Sidebar: dataset uploader
uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"])

# Sidebar: select from existing datasets
existing_files = [f for f in os.listdir("data") if f.endswith(".csv")]
if existing_files:
    selected_existing = st.sidebar.selectbox("Select existing dataset", existing_files)
    if selected_existing:
        st.session_state['selected_dataset'] = os.path.join("data", selected_existing)
MAX_COLUMNS = 20
MAX_ROWS = 5000

if 'selected_dataset' not in st.session_state:
    st.session_state['selected_dataset'] = None

uploaded_df_valid = False

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        n_rows, n_cols = uploaded_df.shape
        if n_cols > MAX_COLUMNS or n_rows > MAX_ROWS:
            st.warning(f"Dataset too large: {n_cols} columns, {n_rows} rows. Max {MAX_COLUMNS} columns, {MAX_ROWS} rows.")
            st.info("Please upload a dataset within these limits to train the AutoML agent.")
            st.stop()
        else:
            uploaded_df_valid = True
            # Save uploaded dataset without timestamp for display
            clean_name = os.path.splitext(uploaded_file.name)[0]
            save_path = os.path.join("data", f"{clean_name}.csv")
            uploaded_df.to_csv(save_path, index=False)
            st.success(f"Dataset '{clean_name}' uploaded and saved successfully!")
            st.session_state['selected_dataset'] = save_path
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")

# Load dataset: use uploaded dataset if valid, else use preloaded selected dataset
@st.cache_data
def load_preloaded_datasets():
    files = [f for f in os.listdir("data") if f.endswith('.csv')]
    return files

preloaded_datasets = load_preloaded_datasets()
if st.session_state['selected_dataset'] is None and preloaded_datasets:
    st.session_state['selected_dataset'] = os.path.join("data", preloaded_datasets[0])

dataset_to_use = st.session_state['selected_dataset']
df = pd.read_csv(dataset_to_use)

# Display dataset stats
display_name = os.path.basename(dataset_to_use).rsplit('.',1)[0]
st.subheader(f"Dataset: {display_name}")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.dataframe(df.head(10))
st.write(df.describe())
st.write("Missing Values:")
st.write(df.isnull().sum())

# Pre-training EDA charts
st.subheader("Pre-training EDA Charts")
pre_training_path = "eda_charts/pre_training"
if os.path.exists(pre_training_path):
    pre_training_images = [f for f in os.listdir(pre_training_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    for img_file in pre_training_images:
        st.image(os.path.join(pre_training_path, img_file), use_container_width=True)

# Post-training EDA charts
st.subheader("Post-training EDA Charts")
post_training_path = "eda_charts/post_training"
if os.path.exists(post_training_path):
    post_training_images = [f for f in os.listdir(post_training_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    for img_file in post_training_images:
        st.image(os.path.join(post_training_path, img_file), use_container_width=True)

# Load trained models
st.subheader("Trained Models")
model_files = [f for f in os.listdir("models") if f.endswith(('.pkl', '.joblib'))]
if not model_files:
    st.write("No trained models found in the models/ directory.")
else:
    for model_file in model_files:
        st.markdown(f"### Model: {model_file}")
        model_path = os.path.join("models", model_file)
        try:
            model = joblib.load(model_path)
            st.write(model)
        except Exception as e:
            st.write(f"Could not load model {model_file}: {e}")
            continue

        # Download button for model
        with open(model_path, "rb") as file:
            st.download_button(
                label=f"Download Model {model_file}",
                data=file,
                file_name=model_file,
                mime="application/octet-stream"
            )