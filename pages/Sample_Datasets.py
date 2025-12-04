import streamlit as st
import pandas as pd
import os
import requests
import tempfile
import shutil
import joblib

from config import DATASET_SOURCES
from preprocessor import preprocess_data
from trainer import train_models
from evaluator import evaluate_models
from model_selector import select_and_save_best_model

st.set_page_config(page_title="Sample Datasets", layout="wide")
st.title("Sample Datasets")

# -------------------- Session state --------------------
if 'active_sample_dataset' not in st.session_state:
    st.session_state['active_sample_dataset'] = None

# Sample datasets
SAMPLE_DATASET_NAMES = [d["name"] for d in DATASET_SOURCES]
SAMPLE_DATASET_URLS = {d["name"]: d["url"] for d in DATASET_SOURCES}
DATASET_GOALS = {d["name"]: d.get("goal", "") for d in DATASET_SOURCES}

st.sidebar.markdown("### Sample Datasets")
sample_choice = st.sidebar.selectbox("Select a dataset", SAMPLE_DATASET_NAMES)

safe_name = sample_choice.replace(" ", "_").lower()
sample_path = os.path.join("data", f"{safe_name}.csv")

# Download dataset if not present
if not os.path.exists(sample_path):
    try:
        response = requests.get(SAMPLE_DATASET_URLS[sample_choice], verify=False, timeout=20)
        response.raise_for_status()
        with open(sample_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        st.error(f"Could not download dataset: {e}")
        st.stop()

st.session_state['active_sample_dataset'] = sample_path
dataset_goal = DATASET_GOALS[sample_choice]

# -------------------- AutoML Button --------------------
st.sidebar.markdown("""
    <style>
    div.stButton > button {
        background-color: #0B3D1F;
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

if not train_clicked:
    st.info("Select a sample dataset, then click **Run AutoML Agent**.")
    st.stop()

# -------------------- Run pipeline --------------------
dataset_to_use = st.session_state['active_sample_dataset']
display_name = os.path.basename(dataset_to_use).rsplit(".", 1)[0]

tmp_dir = tempfile.mkdtemp()
work_file = os.path.join(tmp_dir, f"{display_name}_tmp.csv")
shutil.copy(dataset_to_use, work_file)

st.subheader("AutoML Progress")
progress = st.progress(0)

with st.spinner("Running AutoML Agent..."):
    try:
        progress.progress(10)
        X, y = preprocess_data(work_file)
        progress.progress(40)
        trained_models = train_models(X, y, display_name)
        progress.progress(70)
        results = evaluate_models(trained_models, display_name, dataset_goal)
        progress.progress(90)
        select_and_save_best_model(results)
        progress.progress(100)
        st.success("AutoML Agent completed successfully ")
    finally:
        shutil.rmtree(tmp_dir)

# -------------------- Display results --------------------
df = pd.read_csv(dataset_to_use)
st.subheader("Dataset Preview")
st.write(f"**{display_name}**")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.dataframe(df.head(10))
st.write(df.describe())
st.write(df.isnull().sum())

st.subheader("Training Report")
report_path = os.path.join("reports", f"{display_name.replace(' ','_').lower()}_report.txt")
if os.path.exists(report_path):
    with open(report_path, "r") as f:
        st.text(f.read())

# -------------------- Pre-training EDA Charts --------------------
st.subheader("Pre-training EDA Charts")
pre_training_path = os.path.join("eda_charts", safe_name, "pre_training")
if os.path.exists(pre_training_path):
    pre_training_images = [f for f in os.listdir(pre_training_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    for img_file in pre_training_images:
        st.image(os.path.join(pre_training_path, img_file), use_container_width=True)

# -------------------- Post-training EDA Charts --------------------
st.subheader("Post-training EDA Charts")
post_training_path = os.path.join("eda_charts", safe_name, "post_training")
if os.path.exists(post_training_path):
    post_training_images = [f for f in os.listdir(post_training_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    for img_file in post_training_images:
        st.image(os.path.join(post_training_path, img_file), use_container_width=True)

# -------------------- Trained Models --------------------
st.subheader("Trained Models")
model_files = [f for f in os.listdir("models") if f.startswith(safe_name) and f.endswith(('.pkl', '.joblib'))]
if not model_files:
    st.write("No trained models found for this dataset in the models/ directory.")
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

        with open(model_path, "rb") as file:
            st.download_button(
                label=f"Download Model {model_file}",
                data=file,
                file_name=model_file,
                mime="application/octet-stream"
            )