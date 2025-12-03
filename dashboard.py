import streamlit as st
import pandas as pd
import requests
import os
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from config import DATASET_SOURCES

if 'uploaded_active' not in st.session_state:
    st.session_state['uploaded_active'] = False

st.set_page_config(page_title="AutoML Dashboard", layout="wide")
st.title("AutoML Agent Dashboard")

DATASET_GOALS = {d["name"]: d.get("goal", "") for d in DATASET_SOURCES}

# Sample datasets (from config.py)
SAMPLE_DATASET_NAMES = [d["name"] for d in DATASET_SOURCES]
SAMPLE_DATASET_URLS = {d["name"]: d["url"] for d in DATASET_SOURCES}

st.sidebar.markdown("### Upload a CSV dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"], key="uploader")

MAX_COLUMNS = 20
MAX_ROWS = 5000

if 'selected_dataset' not in st.session_state:
    st.session_state['selected_dataset'] = None
if 'user_goal' not in st.session_state:
    st.session_state['user_goal'] = ""
if 'user_target_column' not in st.session_state:
    st.session_state['user_target_column'] = None

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.session_state['uploaded_columns'] = list(uploaded_df.columns)
        st.session_state['uploaded_active'] = True
        n_rows, n_cols = uploaded_df.shape
        if n_cols > MAX_COLUMNS or n_rows > MAX_ROWS:
            st.warning(f"Dataset too large: {n_cols} columns, {n_rows} rows. Max {MAX_COLUMNS} columns, {MAX_ROWS} rows.")
            st.info("Please upload a dataset within these limits to train the AutoML agent.")
            st.session_state['selected_dataset'] = None
            st.stop()
        else:
            # Save uploaded dataset without timestamp for display
            clean_name = os.path.splitext(uploaded_file.name)[0]
            save_path = os.path.join("data", f"{clean_name}.csv")
            uploaded_df.to_csv(save_path, index=False)
            st.success(f"Dataset '{clean_name}' uploaded and saved successfully!")
            st.session_state['selected_dataset'] = save_path
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")

# st.sidebar.markdown("### Training Details (Uploaded Dataset Only)")
# st.sidebar.subheader("Training Details (Required for Uploaded Dataset)")

if st.session_state.get("uploaded_active"):
    # Goal input (from user)
    user_goal = st.sidebar.text_area(
        "What is the goal of this dataset?",
        value=st.session_state.get("user_goal", ""),
        placeholder="e.g. Predict house prices using features..."
    )
    st.session_state['user_goal'] = user_goal

    # Target column selector (only show if file has been uploaded or selected)
    if st.session_state.get("selected_dataset") is not None:
        try:
            temp_df = pd.read_csv(st.session_state['selected_dataset'])
            target_column = st.sidebar.selectbox(
                "Select target column",
                options=temp_df.columns.tolist()
            )
            st.session_state['user_target_column'] = target_column
        except Exception:
            pass
    else:
        st.sidebar.info("Upload or select a dataset to choose target column.")
else:
    st.sidebar.info("Using predefined dataset goal and target settings.")

# Sidebar: select from user-uploaded datasets (only)
# Only show datasets that were uploaded by users (exclude sample datasets from config.py)
existing_files = [f for f in os.listdir("data") if f.endswith(".csv")]
user_uploaded_files = []

for f in existing_files:
    clean_name = os.path.splitext(f)[0]
    # if the name does NOT match any sample dataset name, treat it as user-uploaded
    if clean_name not in SAMPLE_DATASET_NAMES:
        user_uploaded_files.append(f)

st.sidebar.markdown("### User Uploaded Datasets")

if user_uploaded_files:
    selected_existing = st.sidebar.selectbox(
        "Select an uploaded dataset",
        user_uploaded_files,
        key="existing_selectbox"
    )
    if selected_existing:
        st.session_state['selected_dataset'] = os.path.join("data", selected_existing)
        st.session_state['uploaded_active'] = False
else:
    st.sidebar.info("No user-uploaded datasets found yet.")

st.sidebar.markdown("### Sample Datasets (From Config)")

sample_choice = st.sidebar.selectbox(
    "Select a sample dataset",
    SAMPLE_DATASET_NAMES,
    key="sample_selectbox"
)

if sample_choice:
    sample_url = SAMPLE_DATASET_URLS[sample_choice]
    safe_name = sample_choice.replace(" ", "_").lower()
    sample_path = os.path.join("data", f"{safe_name}.csv")

    if not os.path.exists(sample_path):
        try:
            response = requests.get(sample_url, verify=False, timeout=20)
            response.raise_for_status()
            with open(sample_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            st.sidebar.error(f"Could not load sample dataset: {e}")
            st.stop()

    st.session_state['selected_dataset'] = sample_path
    st.session_state['uploaded_active'] = False

# Custom styled button (green, more visible)
# st.sidebar.markdown("### Run AutoML Agent")
st.sidebar.markdown("""
    <style>
    div.stButton > button {
        background-color: #0B3D1F; /* dark green */
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: 600;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

train_clicked = st.sidebar.button("Run AutoML Agent")

if train_clicked:
    if st.session_state.get("uploaded_active"):
        if not st.session_state.get("user_goal"):
            st.sidebar.error("You must enter a goal before running the agent.")
            st.stop()

        if not st.session_state.get("user_target_column"):
            st.sidebar.error("You must select a target column before running the agent.")
            st.stop()

if train_clicked is False:
    st.info("Select a dataset from the left or upload one, then click **Run AutoML Agent**.")
    st.stop()

# Load dataset: use uploaded dataset if valid, else use preloaded selected dataset
@st.cache_data
def load_preloaded_datasets():
    files = [f for f in os.listdir("data") if f.endswith('.csv')]
    return files

preloaded_datasets = load_preloaded_datasets()
if train_clicked and st.session_state['selected_dataset'] is None and preloaded_datasets:
    st.session_state['selected_dataset'] = os.path.join("data", preloaded_datasets[0])

dataset_to_use = st.session_state['selected_dataset']

# Show training progress bar
st.subheader("AutoML Progress")
progress = st.progress(0)

import time
for i in range(100):
    time.sleep(0.01)
    progress.progress(i + 1)

st.success("AutoML training and analysis complete ")

df = pd.read_csv(dataset_to_use)

st.subheader("Selected Target Column")
st.write(f"Target column: **{st.session_state['user_target_column']}**")
st.session_state['target_column'] = st.session_state['user_target_column']

# Display dataset stats
display_name = os.path.basename(dataset_to_use).rsplit('.',1)[0]
st.subheader(f"Dataset: {display_name}")

dataset_goal = ""
for entry in DATASET_SOURCES:
    if display_name.lower() in entry["name"].lower():
        dataset_goal = entry.get("goal", "")
        break
if st.session_state.get("uploaded_active") and st.session_state.get("user_goal"):
    st.info(f"User-defined goal: {st.session_state['user_goal']}")
elif dataset_goal:
    st.info(f"Dataset goal: {dataset_goal}")

st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.dataframe(df.head(10))
st.write(df.describe())
st.write("Missing Values:")
st.write(df.isnull().sum())

report_path = os.path.join("reports", f"{display_name.replace(' ','_').lower()}_report.txt")
if os.path.exists(report_path):
    st.subheader("Latest Training Report")
    with open(report_path, "r") as r:
        st.text(r.read())

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