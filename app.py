# ============================================================
# app.py — FINAL Phase-5 Real-Time ECG Monitoring Dashboard
# ============================================================

import streamlit as st
import sys, os, pickle, types, time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Pandas backward compatibility shim (CRITICAL)
# ------------------------------------------------------------
numeric_module = types.ModuleType("pandas.core.indexes.numeric")
class Int64Index(pd.Index): pass
numeric_module.Int64Index = Int64Index
sys.modules["pandas.core.indexes.numeric"] = numeric_module

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------
sys.path.append(os.path.join(os.getcwd(), "src"))
from src.model import CNNMultiLabelECG
from src.preprocessing import ECGPreprocessor

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
LABELS = ["AF","LBBB","RBBB","PAC","PVC","STD","STE","Normal","Other"]

THRESHOLDS = np.array([0.45,0.40,0.40,0.35,0.35,0.50,0.50,0.60,0.30])

MODEL_PATH = "ecg_model_v1.pth"
SAMPLING_RATE = 250
WINDOW_SECONDS = 10
STEP_SECONDS = 5

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Load model (cached)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = CNNMultiLabelECG(num_classes=9)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()
preprocessor = ECGPreprocessor()

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def format_prob(p):
    return f"{min(float(p), 0.999):.3f}"

def clinical_explanation(label):
    rules = {
        "AF": "Irregular atrial rhythm detected. Consider anticoagulation assessment.",
        "LBBB": "Left bundle branch block pattern detected.",
        "RBBB": "Right bundle branch block pattern detected.",
        "PAC": "Premature atrial contractions detected.",
        "PVC": "Premature ventricular contractions detected.",
        "STD": "ST depression may suggest myocardial ischemia.",
        "STE": "ST elevation detected. Urgent cardiology review recommended.",
        "Normal": "Normal sinus rhythm detected.",
        "Other": "Unclassified arrhythmia pattern detected."
    }
    return rules[label]

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Real-Time ECG Arrhythmia Monitor", layout="wide")

st.title("🫀 Real-Time ECG Arrhythmia Monitoring Dashboard")
st.markdown(
    "Upload a **preprocessed ECG `.pk` file** to simulate real-time "
    "multi-label arrhythmia detection with clinical decision support."
)

uploaded_file = st.file_uploader("Upload preprocessed ECG (.pk)", type=["pk"])

# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
if uploaded_file:

    with st.spinner("Loading ECG file..."):
        signal_df, _ = pickle.load(uploaded_file)
        ecg = signal_df.values.T.astype(np.float32)
        ecg = preprocessor.preprocess(ecg)

    st.success("ECG loaded successfully.")

    total_samples = ecg.shape[1]
    window_size = WINDOW_SECONDS * SAMPLING_RATE
    step_size = STEP_SECONDS * SAMPLING_RATE

    st.subheader("📡 Simulated Real-Time Monitoring")

    plot_placeholder = st.empty()
    table_placeholder = st.empty()
    text_placeholder = st.empty()

    if st.button("▶ Start Monitoring"):

        for start in range(0, total_samples - window_size, step_size):
            end = start + window_size
            window = ecg[:, start:end]

            # ------------------ ECG Plot ------------------
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(window[1], color="cyan", linewidth=1)
            ax.set_title("Lead II ECG Signal (10-second window)")
            ax.set_xlabel("Time (samples @ 250Hz)")
            ax.set_ylabel("Amplitude (normalized)")
            ax.grid(alpha=0.3)
            plot_placeholder.pyplot(fig)
            plt.close(fig)

            # ------------------ Inference ------------------
            X = torch.tensor(window).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = model(X).cpu().numpy().squeeze()

            preds = (probs >= THRESHOLDS).astype(int)

            df = pd.DataFrame({
                "Arrhythmia": LABELS,
                "Probability": probs,
                "Threshold": THRESHOLDS,
                "Detected": preds
            }).sort_values("Probability", ascending=False)

            table_placeholder.dataframe(df, use_container_width=True)

            # ------------------ Clinical Support ------------------
            explanations = []
            for lbl, p, d in zip(LABELS, probs, preds):
                if d == 1:
                    explanations.append(
                        f"- **{lbl}** detected "
                        f"(confidence {format_prob(p)}): "
                        f"{clinical_explanation(lbl)}"
                    )

            if explanations:
                text_placeholder.markdown(
                    "### 🧠 Clinical Decision Support\n" + "\n".join(explanations)
                )
            else:
                text_placeholder.markdown(
                    "### 🧠 Clinical Decision Support\n"
                    "No pathological arrhythmias detected in this segment."
                )

            time.sleep(1)

        st.success("Real-time monitoring completed.")
