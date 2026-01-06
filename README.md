# 🫀 ECG Decision Support System (ECG-DSS)

### Multi-Label ECG Arrhythmia Detection & Real-Time Monitoring System

---

## Project Overview

ECG-DSS is an **end-to-end deep learning–based clinical decision support system** for **multi-label ECG arrhythmia detection**, developed during a **time-bound AI/ML hackathon**.

The project covers the complete lifecycle:

- Exploratory Data Analysis (EDA)
- ECG signal preprocessing and dataset engineering
- Deep learning model design and training
- Multi-label validation and threshold optimization
- Deployment as a **real-time ECG monitoring dashboard**

The system detects **multiple co-occurring cardiac arrhythmias** from **12-lead ECG signals**, with strong emphasis on **robust decision-making, interpretability, and clinical realism**.

---

## 🔗 Live Demo (Streamlit Deployment)

👉 **Deployed Application:** [**https://ecg-arrhythmia-multi-label-classification.streamlit.app/**](https://ecg-arrhythmia-multi-label-classification.streamlit.app/)

> The live app allows uploading **preprocessed `.pk` ECG files** and simulates real-time multi-label arrhythmia detection with clinical decision support explanations.

---

## Problem Statement

Traditional ECG classifiers assume **single-label outputs**, whereas real-world ECG recordings may exhibit **multiple simultaneous arrhythmias**.

This project addresses:

> **Multi-label cardiac arrhythmia classification from variable-length 12-lead ECG recordings**, explicitly accounting for **class imbalance**, **label co-occurrence**, and **temporal variability**.

---

## Supported Arrhythmia Classes (9)

- **AF** — Atrial Fibrillation
- **LBBB** — Left Bundle Branch Block
- **RBBB** — Right Bundle Branch Block
- **PAC** — Premature Atrial Contractions
- **PVC** — Premature Ventricular Contractions
- **STD** — ST Depression
- **STE** — ST Elevation
- **Normal**
- **Other**

Each ECG sample may contain **multiple active labels simultaneously**.

---

## Dataset Information

### Original Dataset (CPSC 2018)

- **Name:** China Physiological Signal Challenge (CPSC 2018)
- **Source:** PhysioNet
- **Link:** [PhysioNet Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/)
- **Sampling Rate:** 500 Hz
- **Duration:** Variable (6–60 seconds)
- **Leads:** 12-lead ECG
- **Labels:** Multi-label (9 classes)
- **Size:** 6,877 clinical ECG recordings

**Reference Paper:** [Frontiers in Physiology Article](https://www.frontiersin.org/articles/10.3389/fphys.2021.678597/full)

### Preprocessed Dataset

- **Source:** Figshare (community-preprocessed)
- **Link:** [Figshare ECG Data](https://figshare.com/articles/dataset/ECG_data/)
- **Sampling Rate:** 250 Hz
- **Duration:** Normalized to 60 seconds
- **Format:** Ready-to-use signals for deep learning

### Included Sample Files

Due to dataset licensing and size constraints, **only a limited set of preprocessed `.pk` samples** is included under the `samples/` directory **for demo and testing**.

Users must download the **full dataset** from the links above to reproduce training results.

---

## PHASE 1 – Exploratory Data Analysis (EDA)

### Key Observations

- High temporal variability and lead-specific ECG morphology
- Severe class imbalance (Normal and AF dominate)
- ~7% of ECGs contain **multiple simultaneous arrhythmias**
- Frequent label co-occurrence justifies multi-label modeling
- Lead correlations show shared cardiac activity with unique patterns
- Padding and temporal modeling are mandatory

> **Note:** Baseline correction is intentionally disabled to preserve clinically relevant ST-segment morphology.

---

## PHASE 2 – Preprocessing & Dataset Engineering

- Resampling to 250 Hz
- Duration normalization to 60 seconds
- Lead-wise normalization
- Multi-hot label encoding
- Memory-efficient `.pk` storage
- **Lazy loading** to prevent RAM exhaustion (Windows-safe)

---

## PHASE 3 – Model Architecture & Training

### Model Architecture

- 1D CNN with **ResNet-style residual blocks**
- Temporal feature extraction across 12 leads
- **Attention pooling** for temporal importance weighting
- Sigmoid activation for independent label probabilities

### Training Strategy

- Multi-label learning with **Focal Loss**
- GPU acceleration (CUDA / MPS / CPU fallback)
- Lazy dataset loading for scalability
- Real-time training progress with `tqdm`

### Evaluation Metrics

- Hamming Loss
- F1 Score (Macro & Micro)
- Per-class AUC-ROC
- Subset Accuracy (reported cautiously)

---

## PHASE 4 – Multi-Label Validation & Decision Optimization

Phase 4 focuses on **decision quality**, not representation learning.

- Binary Cross-Entropy vs Focal Loss (conceptual comparison)
- Explicit class imbalance handling
- Multi-label stratified data splitting
- Cross-validation strategy explanation
- **Per-class threshold optimization**
- Significant post-training metric improvement

> Subset accuracy remains inherently low due to its strict definition in multi-label tasks.

---

## PHASE 5 – Real-Time ECG Monitoring Dashboard

### Features

- Streamlit-based real-time ECG monitoring
- Upload preprocessed `.pk` ECG files
- Sliding-window simulation of live ECG streams
- CUDA-accelerated inference
- Optimized per-class thresholds
- Clinical decision-support explanations

> Explanations are **assistive**, not autonomous diagnoses.

---

## Hackathon Context

- Strict time constraints
- Solo participation
- Limited hardware reliability
- Emphasis on **end-to-end system completeness**

---

## Technology Stack

```
# -------------------------------
# Core Scientific Stack
# -------------------------------
numpy>=1.23
pandas>=1.5
scipy>=1.9
matplotlib>=3.6
seaborn>=0.12

# -------------------------------
# Machine Learning & Metrics
# -------------------------------
scikit-learn>=1.2
imbalanced-learn>=0.10

# -------------------------------
# Deep Learning (CUDA-enabled)
# -------------------------------
torch>=2.0
torchvision>=0.15
torchaudio>=2.0

# NOTE:
# CUDA support is provided by the installed PyTorch build.
# Install CUDA-specific wheels via:
# https://pytorch.org/get-started/locally/

# -------------------------------
# ECG / Biomedical Signal Processing
# -------------------------------
wfdb>=4.1

# -------------------------------
# Progress Bars / UX
# -------------------------------
tqdm>=4.65

# -------------------------------
# Jupyter Environment (Optional but Used)
# -------------------------------
jupyter>=1.0
ipykernel>=6.20

# -------------------------------
# Deployment
# -------------------------------
streamlit>=1.52
```

---

## Project Structure (Folder-Level)

```plaintext
ECG-Decision-Support-System/
├── app.py
├── run_phase3.py
├── ecg_model_v1.pth
├── samples/
├── data/
├── notebooks/
├── src/
├── experiments/
├── requirements.txt
├── split_train_into_train_val.py
└── README.md

```

---

## 🚀 Setup & Usage Guide

### 1️⃣ Clone the Repository

```bash
git clone [https://github.com/Surya-Teja-Kancharla/ECG-Decision-Support-System.git](https://github.com/Surya-Teja-Kancharla/ECG-Decision-Support-System.git)
cd ECG-DSS

```

### 2️⃣ Create & Activate Virtual Environment

**Windows**

```bash
python -m venv ecg_env
ecg_env\Scripts\activate

```

**Linux / macOS**

```bash
python3 -m venv ecg_env
source ecg_env/bin/activate

```

### 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

> CUDA support is automatically enabled if a CUDA-compatible PyTorch build is installed.

### 4️⃣ Train the Model (Optional)

```bash
python run_phase3.py

```

> Requires full preprocessed dataset (not included in repo).

### 5️⃣ Run the Streamlit App Locally

```bash
streamlit run app.py

```

> Upload a `.pk` ECG file from the `samples/` directory to test.

---

## Final Remarks

ECG-DSS demonstrates:

* Correct multi-label ECG modeling
* Robust handling of severe class imbalance
* Clinically motivated validation strategies
* A complete pipeline from EDA → Modeling → Validation → Deployment

