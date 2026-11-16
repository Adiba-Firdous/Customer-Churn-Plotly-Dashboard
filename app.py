from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

st.set_page_config(page_title="Customer Churn Model Evaluation", page_icon="📊", layout="wide")
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "sample_predictions.csv"

st.title("📊 Customer Churn — Model Evaluation Dashboard")
st.caption("Interactive evaluation dashboard using Plotly. The bundled CSV contains synthetic demonstration predictions.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload your model predictions CSV", type=["csv"])
    st.markdown("---")
    st.markdown("**Required columns**")
    st.code("y_true, y_pred")
    st.markdown("**Optional probability column**")
    st.code("y_probability")
    threshold = st.slider("Classification threshold", 0.0, 1.0, 0.50, 0.01)
    st.caption("For probability-based predictions, changing the threshold recalculates y_pred.")

try:
    data = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(DEFAULT_DATA)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

required = {"y_true"}
if "y_probability" in data.columns:
    data["y_pred"] = (pd.to_numeric(data["y_probability"], errors="coerce") >= threshold).astype(int)
elif "y_pred" in data.columns:
    data["y_pred"] = pd.to_numeric(data["y_pred"], errors="coerce")
else:
    st.error("CSV must contain y_true and y_pred, or y_true and y_probability.")
    st.stop()

data["y_true"] = pd.to_numeric(data["y_true"], errors="coerce")
data = data.dropna(subset=["y_true", "y_pred"]).copy()
if data.empty:
    st.error("No valid rows found. Check that labels contain 0 and 1.")
    st.stop()

data["y_true"] = data["y_true"].astype(int)
data["y_pred"] = data["y_pred"].astype(int)
if not set(data["y_true"].unique()).issubset({0, 1}) or not set(data["y_pred"].unique()).issubset({0, 1}):
    st.error("This dashboard currently supports binary labels encoded as 0 and 1.")
    st.stop()

average = "binary"
accuracy = accuracy_score(data["y_true"], data["y_pred"])
precision = precision_score(data["y_true"], data["y_pred"], zero_division=0)
recall = recall_score(data["y_true"], data["y_pred"], zero_division=0)
f1 = f1_score(data["y_true"], data["y_pred"], zero_division=0)

st.info("Using bundled synthetic demo data." if uploaded is None else f"Using uploaded file: {uploaded.name}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy:.1%}")
c2.metric("Precision", f"{precision:.1%}")
c3.metric("Recall", f"{recall:.1%}")
c4.metric("F1-score", f"{f1:.1%}")

st.markdown("---")
left, right = st.columns(2)
with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(data["y_true"], data["y_pred"], labels=[0, 1])
    fig_cm = px.imshow(
        cm, text_auto=True, aspect="auto",
        x=["Predicted: No churn", "Predicted: Churn"],
        y=["Actual: No churn", "Actual: Churn"],
        color_continuous_scale="Blues", title="Actual vs predicted labels"
    )
    fig_cm.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig_cm, use_container_width=True)
with right:
    st.subheader("Evaluation Metrics")
    metric_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Score": [accuracy, precision, recall, f1]
    })
    fig_metrics = px.bar(metric_df, x="Metric", y="Score", text=metric_df["Score"].map(lambda x: f"{x:.1%}"),
                         range_y=[0, 1], color="Metric", title="Model performance")
    fig_metrics.update_traces(textposition="outside")
    st.plotly_chart(fig_metrics, use_container_width=True)

if "y_probability" in data.columns:
    st.subheader("ROC Curve")
    try:
        auc = roc_auc_score(data["y_true"], pd.to_numeric(data["y_probability"], errors="coerce"))
        fpr, tpr, _ = roc_curve(data["y_true"], pd.to_numeric(data["y_probability"], errors="coerce"))
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(dash="dash")))
        fig_roc.update_layout(title="Receiver Operating Characteristic", xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate", xaxis=dict(range=[0,1]),
                              yaxis=dict(range=[0,1]))
        st.plotly_chart(fig_roc, use_container_width=True)
    except Exception:
        st.warning("ROC curve needs both classes and valid probability values.")

st.subheader("Classification Report")
report = classification_report(data["y_true"], data["y_pred"], labels=[0,1],
                               target_names=["No churn (0)", "Churn (1)"], output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

st.subheader("Prediction Data")
st.dataframe(data, use_container_width=True, height=300)
st.download_button("⬇️ Download evaluated data", data=data.to_csv(index=False).encode("utf-8"),
                   file_name="evaluated_predictions.csv", mime="text/csv")

with st.expander("How to connect your previous trained model"):
    st.markdown("""
    1. Generate predictions on a held-out test set (data not used to train the model).
    2. Save a CSV with `y_true` and `y_pred`. You may also include `y_probability` (probability of class 1).
    3. Upload that CSV using the sidebar.
    4. Ensure `1 = churn` and `0 = no churn` for both actual and predicted labels.
    """)
