# 🫀 Multi-Label ECG Arrhythmia Detection & Real-Time Monitoring

## Project Overview

This project implements an **end-to-end multi-label ECG arrhythmia detection system**, designed and built during a time-bound AI/ML hackathon.  
It spans the full lifecycle from **EDA and data preprocessing** to **deep learning model training**, **multi-label validation**, **threshold optimization**, and a **real-time clinical monitoring dashboard**.

The system supports **simultaneous detection of multiple arrhythmias** from 12-lead ECG signals and emphasizes **decision quality, robustness, and clinical interpretability**.

---

## Supported Arrhythmia Classes (9)

- AF (Atrial Fibrillation)
- LBBB (Left Bundle Branch Block)
- RBBB (Right Bundle Branch Block)
- PAC (Premature Atrial Contractions)
- PVC (Premature Ventricular Contractions)
- STD (ST Depression)
- STE (ST Elevation)
- Normal
- Other

---

## Data Summary

- **Sampling rate:** 250 Hz
- **Signal length:** Fixed 60-second segments
- **Format:** Preprocessed `.pk` files (Pandas objects)
- **Leads:** Standard 12-lead ECG
- **Labels:** Multi-hot encoded (multi-label)

---

## PHASE 1 – Exploratory Data Analysis (EDA)

### Key EDA Observations

- ECG signals exhibit high temporal variability and lead-specific morphology
- Class distribution is severely imbalanced, with Normal and AF dominating
- ~7% of samples contain multiple simultaneous arrhythmias
- Certain arrhythmias frequently co-occur, justifying multi-label modeling
- Lead correlations indicate shared cardiac activity while preserving unique patterns
- Padding and temporal modeling are mandatory for downstream learning

### Phase 1 Summary

- Verified signal integrity and duration normalization
- Visualized 12-lead ECG morphology
- Identified severe class imbalance
- Quantified multi-label co-occurrence patterns
- Established need for weighted multi-label learning

> **Note:**  
> Baseline correction is intentionally disabled by default to avoid altering clinically relevant ST-segment morphology.

---

## PHASE 2 – Preprocessing & Dataset Engineering

- Signal normalization and resampling to 250 Hz
- Duration standardization (60 seconds)
- Label encoding into multi-hot vectors
- Memory-efficient `.pk` storage
- Lazy loading strategy to prevent RAM exhaustion on Windows

---

## PHASE 3 – Model Architecture & Training

### Model Design

- 1D CNN with Residual Blocks (ResNet-style)
- Temporal feature extraction across all 12 leads
- Attention pooling to support variable-length modeling
- Sigmoid activation for independent label probabilities

### Training Strategy

- Multi-label classification using **Focal Loss**
- GPU-accelerated training (CUDA / MPS / CPU fallback)
- Lazy data loading to scale to thousands of ECG samples
- Real-time progress tracking with `tqdm`

### Evaluation Metrics

- Hamming Loss
- F1-Score (Macro & Micro)
- Per-class AUC-ROC
- Subset Accuracy (reported with caution)

---

## Why Stratified Splitting Matters (Multi-Label ECG)

Random splits distort rare arrhythmia prevalence and break label co-occurrence structure.

Stratification preserves:

- Per-class prevalence
- Co-occurrence structure
- Clinical realism

This project uses **label-aware splitting prior to Phase 3 training**.

---

## Cross-Validation Strategy (Multi-Label Aware)

- Full K-fold cross-validation is computationally expensive for ECG
- We rely on:
  - Dedicated validation split
  - Ensemble averaging
  - Threshold calibration

This achieves variance reduction without violating label dependencies.

---

## PHASE 4 – Multi-Label Validation & Decision Optimization

Phase 4 focuses on **decision quality**, not representation learning.

### Phase 4 Summary

- ✓ Loss functions compared conceptually (Binary Cross-Entropy vs Focal Loss)
- ✓ Class imbalance quantified and addressed
- ✓ Multi-label stratification justified
- ✓ Cross-validation strategy explained
- ✓ Per-class threshold optimization implemented
- ✓ Significant post-training metric improvement achieved

### Notes on Subset Accuracy

Subset accuracy is intentionally low in multi-label ECG tasks due to its strict definition.  
After per-class threshold optimization, a measurable improvement is observed, indicating better joint label consistency without retraining the model.

---

## PHASE 5 – Real-Time ECG Monitoring Dashboard

### Features

- Streamlit-based real-time ECG monitoring dashboard
- Upload preprocessed `.pk` ECG files
- Sliding-window simulation of live ECG streams
- CUDA-accelerated inference
- Multi-label probability table with thresholds
- Clinical decision support explanations per detected arrhythmia

### Clinical Decision Support

Detected arrhythmias are accompanied by rule-based clinical interpretations, clearly labeled as **decision support**, not autonomous diagnosis.

---

## Technology Stack

- Python
- PyTorch (CUDA / MPS support)
- NumPy, Pandas, SciPy
- scikit-learn
- tqdm
- wfdb
- Streamlit
- Matplotlib / Seaborn

---

## Project Structure

Ignite Hack/
│
├── app.py # Phase 5 real-time dashboard
├── run_phase3.py # Training entry point
├── src/
│ ├── model.py # 1D CNN + ResNet + Attention
│ ├── train.py # Training loop
│ ├── evaluate.py # Multi-label metrics
│ ├── ensemble.py # Ensemble inference
│ ├── preprocessing.py
│ ├── augmentation.py
│
├── data/
│ └── processed/
│
├── notebooks/
│ └── 01_eda.ipynb
│
└── README.md

---

## Final Remarks

This project demonstrates:

- Correct handling of multi-label ECG classification
- Robust modeling under severe class imbalance
- Clinically motivated validation strategies
- A complete path from EDA → Model → Validation → Deployment

The system is suitable for **hackathon demos, academic evaluation, and prototype clinical decision-support pipelines**.
