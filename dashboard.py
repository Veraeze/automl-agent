import streamlit as st
import pandas as pd
import os
import tempfile
import shutil
import joblib

from preprocessor import preprocess_data
from trainer import train_models
from evaluator import evaluate_models
from model_selector import select_and_save_best_model

# -------------------- Page config --------------------
st.set_page_config(page_title="AutoML Dashboard - Upload & Train", layout="wide")
st.title("Upload & Train Dataset")

# -------------------- Session state --------------------
if 'uploaded_active' not in st.session_state:
    st.session_state['uploaded_active'] = False
if 'session_uploaded_files' not in st.session_state:
    st.session_state['session_uploaded_files'] = {}
if 'active_uploaded_dataset' not in st.session_state:
    st.session_state['active_uploaded_dataset'] = None
if 'user_goal' not in st.session_state:
    st.session_state['user_goal'] = ""
if 'user_target_column' not in st.session_state:
    st.session_state['user_target_column'] = None

MAX_COLUMNS = 20
MAX_ROWS = 5000
MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# -------------------- Upload CSV --------------------
st.sidebar.markdown(f"### Upload a CSV dataset (Max {MAX_FILE_SIZE_MB} MB)")
uploaded_file = st.sidebar.file_uploader("Drag and drop file here", type=["csv"])

if uploaded_file:
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(f"File too large. Max {MAX_FILE_SIZE_MB} MB.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
        n_rows, n_cols = df.shape

        if n_cols > MAX_COLUMNS or n_rows > MAX_ROWS:
            st.warning(f"Dataset too large: {n_cols} cols, {n_rows} rows. Max {MAX_COLUMNS} cols, {MAX_ROWS} rows.")
            st.stop()

        clean_name = os.path.splitext(uploaded_file.name)[0]
        save_path = os.path.join("data", f"{clean_name}.csv")
        df.to_csv(save_path, index=False)

        st.success(f"Dataset '{clean_name}' uploaded and saved.")

        st.session_state['session_uploaded_files'][clean_name] = df
        st.session_state['active_uploaded_dataset'] = save_path
        st.session_state['uploaded_active'] = True

    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")

# -------------------- Training details --------------------
if st.session_state.get("uploaded_active") and st.session_state.get("active_uploaded_dataset"):

    user_goal = st.sidebar.text_area(
        "What is the goal of this dataset?",
        value=st.session_state.get("user_goal", ""),
        placeholder="e.g. Predict house prices..."
    )
    st.session_state['user_goal'] = user_goal

    temp_df = pd.read_csv(st.session_state['active_uploaded_dataset'])
    target_column = st.sidebar.selectbox(
        "Select target column",
        options=temp_df.columns.tolist()
    )
    st.session_state['user_target_column'] = target_column

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
    st.info("Upload a CSV dataset, fill in goal and target, then click **Run AutoML Agent**.")
    st.stop()

# -------------------- Run pipeline --------------------
dataset_to_use = st.session_state['active_uploaded_dataset']
dataset_goal = st.session_state['user_goal']
display_name = os.path.basename(dataset_to_use).rsplit(".", 1)[0]

tmp_dir = tempfile.mkdtemp()
df_tmp = pd.read_csv(dataset_to_use)
cols = [c for c in df_tmp.columns if c != st.session_state['user_target_column']] + [st.session_state['user_target_column']]
df_tmp = df_tmp[cols]
work_file = os.path.join(tmp_dir, f"{display_name}_tmp.csv")
df_tmp.to_csv(work_file, index=False)

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
st.subheader("Dataset Preview")
df = pd.read_csv(dataset_to_use)
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

# -------------------- Display pre-training EDA --------------------
st.subheader("Pre-training EDA Charts")
pre_path = os.path.join("eda_charts", display_name.replace(" ", "_").lower(), "pre_training")
if os.path.exists(pre_path):
    for img in os.listdir(pre_path):
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            st.image(os.path.join(pre_path, img), use_container_width=True)
else:
    st.info("No pre-training EDA charts found.")

# -------------------- Display post-training EDA --------------------
st.subheader("Post-training EDA Charts")
post_path = os.path.join("eda_charts", display_name.replace(" ", "_").lower(), "post_training")
if os.path.exists(post_path):
    for img in os.listdir(post_path):
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            st.image(os.path.join(post_path, img), use_container_width=True)
else:
    st.info("No post-training EDA charts found.")

# -------------------- Display trained models --------------------
st.subheader("Trained Models")
model_files = [f for f in os.listdir("models") if f.startswith(display_name.replace(" ", "_").lower()) and f.endswith(('.pkl', '.joblib'))]
if not model_files:
    st.info("No trained models found for this dataset.")
else:
    for model_file in model_files:
        st.markdown(f"### Model: {model_file}")
        model_path = os.path.join("models", model_file)
        try:
            model = joblib.load(model_path)
            st.write(model)
        except Exception as e:
            st.warning(f"Could not load model {model_file}: {e}")
            continue

        with open(model_path, "rb") as file:
            st.download_button(
                label=f"Download Model {model_file}",
                data=file,
                file_name=model_file,
                mime="application/octet-stream"
            )
