import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 – Model Evaluation",
    layout="wide"
)

st.title("Machine Learning Assignment 2")
st.write("Model Evaluation and Comparison using Streamlit")

# --------------------------------------------------
# Load trained models and scaler
# --------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl"),
    }
    scaler = joblib.load("model/scaler.pkl")
    return models, scaler


models, scaler = load_models()

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Configuration")

model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

model = models[model_name]

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV file)",
    type=["csv"]
)

# --------------------------------------------------
# Main logic
# --------------------------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "target" not in data.columns:
        st.error("❌ Uploaded CSV must contain a 'target' column.")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        # Scale features
        X_scaled = scaler.transform(X)

        # Predictions
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
        st.subheader(f"Evaluation Metrics – {model_name}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.3f}")
        col2.metric("AUC", f"{roc_auc_score(y, y_prob):.3f}")
        col3.metric("Precision", f"{precision_score(y, y_pred):.3f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{recall_score(y, y_pred):.3f}")
        col5.metric("F1 Score", f"{f1_score(y, y_pred):.3f}")
        col6.metric("MCC", f"{matthews_corrcoef(y, y_pred):.3f}")

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
