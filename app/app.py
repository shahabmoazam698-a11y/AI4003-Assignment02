
from pathlib import Path
import sys

import joblib
import numpy as np
import streamlit as st

# Locate the main project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"

# Import the manually implemented logistic regression function
sys.path.insert(0, str(PROJECT_DIR / "src"))
from logistic import predict_proba

# Load saved preprocessing values
data = np.load(RESULTS_DIR / "prepared_data.npz")
mean_Q = data["mean_Q"]
std_Q = data["std_Q"]
mean_R = data["mean_R"]
std_R = data["std_R"]

# Load trained models
logistic_data = np.load(MODELS_DIR / "logistic_Q.npz")
theta_Q = logistic_data["theta"]
rbf_svm = joblib.load(MODELS_DIR / "svm_rbf.joblib")

# Page settings
st.set_page_config(
    page_title="Transmission Line Fault Detection",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Three-Phase Transmission Line Fault Detection")
st.write(
    "Enter the six electrical measurements below to predict "
    "whether the sample represents a normal condition or a fault."
)

st.info(
    "This is an academic demonstration using trained machine-learning "
    "models. It is not a certified electrical protection system."
)

# Input fields
st.subheader("Electrical Measurements")

col1, col2 = st.columns(2)

with col1:
    Ia = st.number_input("Ia — Current A", value=0.0, format="%.6f")
    Ib = st.number_input("Ib — Current B", value=0.0, format="%.6f")
    Ic = st.number_input("Ic — Current C", value=0.0, format="%.6f")

with col2:
    Va = st.number_input("Va — Voltage A", value=0.0, format="%.6f")
    Vb = st.number_input("Vb — Voltage B", value=0.0, format="%.6f")
    Vc = st.number_input("Vc — Voltage C", value=0.0, format="%.6f")

if st.button("Detect Fault", type="primary", use_container_width=True):
    # Original six features in the required order
    X_raw = np.array([[Ia, Ib, Ic, Va, Vb, Vc]], dtype=float)

    # Set Q: original features followed by their squares
    X_Q_raw = np.column_stack([X_raw, X_raw ** 2])

    # Use the training-set statistics saved during preparation
    X_Q = (X_Q_raw - mean_Q) / std_Q

    # Logistic Regression Set Q
    logistic_probability = float(predict_proba(X_Q, theta_Q)[0])
    logistic_prediction = int(logistic_probability >= 0.5)

    # RBF SVM uses standardized Set R features
    X_R = (X_raw - mean_R) / std_R
    svm_prediction = int(rbf_svm.predict(X_R)[0])

    st.subheader("Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.markdown("**Logistic Regression — Set Q**")
        if logistic_prediction == 1:
            st.error("Fault detected")
        else:
            st.success("Normal condition")
        st.write(f"Predicted fault probability: {logistic_probability:.2%}")

    with result_col2:
        st.markdown("**RBF SVM**")
        if svm_prediction == 1:
            st.error("Fault detected")
        else:
            st.success("Normal condition")

    st.caption(
        "Predictions depend on the input units and the distribution "
        "of the training dataset. Use realistic measurements."
    )
