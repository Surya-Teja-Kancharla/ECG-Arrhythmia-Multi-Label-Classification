# ============================================================
# run_phase3.py (Memory Optimized)
# ============================================================
# Purpose:
#   Phase-3 Training Entry Point with MEMORY OPTIMIZATION
#
# Key Improvements vs Previous Version:
#   - Uses lazy loading instead of loading entire dataset into RAM
#   - Prevents Windows OOM / RAM exhaustion
#   - Scales safely to 5k–7k ECG samples
#
# Responsibilities:
#   - Load TRAIN and CROSS_VALID datasets lazily
#   - Apply ECG preprocessing and augmentation
#   - Train multi-label ECG classifier
#   - Save trained model
#   - Verify ensemble logic
#
# Notes:
#   - cross_valid is treated as validation set
#   - test set is NOT touched
#   - CUDA / MPS / CPU automatically selected
# ============================================================


# ------------------------------------------------------------
# 0. Pandas backward-compatibility shim (CRITICAL)
# ------------------------------------------------------------
# Required to unpickle legacy Pandas objects
# MUST be executed before importing pickle
import sys
import types
import pandas as pd

numeric_module = types.ModuleType("pandas.core.indexes.numeric")

class Int64Index(pd.Index):
    pass

numeric_module.Int64Index = Int64Index
sys.modules["pandas.core.indexes.numeric"] = numeric_module
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Standard & Torch Imports
# ------------------------------------------------------------
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# ------------------------------------------------------------


# ------------------------------------------------------------
# 2. Ensure `src` is importable
# ------------------------------------------------------------
sys.path.append(os.path.join(os.getcwd(), "src"))
# ------------------------------------------------------------


# ------------------------------------------------------------
# 3. Project Imports
# ------------------------------------------------------------
from src.model import CNNMultiLabelECG, FocalLoss
from src.preprocessing import ECGPreprocessor
from src.augmentation import ECGAugmentor
from src.train import train_model
from src.ensemble import ensemble_predict
# ------------------------------------------------------------


# ------------------------------------------------------------
# 4. Robust Device Detection
# ------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[INFO] Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[INFO] Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("[WARNING] No GPU detected. Using CPU.")
# ------------------------------------------------------------


# ------------------------------------------------------------
# 5. Dataset Directories
# ------------------------------------------------------------
TRAIN_DIR = "data/processed/train_split/250Hz/60s"
VAL_DIR   = "data/processed/val_split/250Hz/60s"
# ------------------------------------------------------------


# ------------------------------------------------------------
# 6. Lazy Dataset (CORE MEMORY FIX)
# ------------------------------------------------------------
class LazyECGDataset(Dataset):
    """
    Lazily loads ECG samples from disk.

    Benefits:
    ---------
    - Only one ECG sample is loaded into RAM at a time
    - Prevents memory overflow on Windows
    - Scales to large ECG datasets safely

    Each item:
    ----------
    Returns:
        signal : Tensor (12, T)
        label  : Tensor (9,)
    """

    def __init__(self, folder, augment=False):
        self.folder = folder
        self.augment = augment
        self.files = [f for f in os.listdir(folder) if f.endswith(".pk")]
        self.preprocessor = ECGPreprocessor()
        self.augmentor = ECGAugmentor()

        print(f"[INFO] Initialized LazyDataset for {folder} with {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    # Inside run_phase3.py

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder, self.files[idx])

        with open(file_path, "rb") as f:
            signal_df, label_dict = pickle.load(f)

        # 1. Load and Cast to Float32 immediately
        sig = signal_df.values.T.astype(np.float32)

        # 2. Preprocess
        sig = self.preprocessor.preprocess(sig)

        # 3. Augment (only if training)
        if self.augment:
            sig = self.augmentor.augment(sig)

        # 4. Load Labels
        lbl = label_dict["classes_one_hot"].values.astype(np.float32)

        # 5. CRITICAL FIX: Enforce torch.float32
        # .copy() prevents "negative stride" errors common with signal processing
        return torch.tensor(sig.copy(), dtype=torch.float32), torch.tensor(lbl.copy(), dtype=torch.float32)
# ------------------------------------------------------------


# ------------------------------------------------------------
# 7. Initialize Datasets
# ------------------------------------------------------------
train_data = LazyECGDataset(TRAIN_DIR, augment=True)
val_data   = LazyECGDataset(VAL_DIR, augment=False)
# ------------------------------------------------------------


# ------------------------------------------------------------
# 8. DataLoaders
# ------------------------------------------------------------
# num_workers=0 is REQUIRED on Windows for pickle safety
train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=16,
    shuffle=False,
    num_workers=0
)
# ------------------------------------------------------------


# ------------------------------------------------------------
# 9. Model & Loss Setup
# ------------------------------------------------------------
model = CNNMultiLabelECG(num_classes=9)
criterion = FocalLoss(gamma=2)
# ------------------------------------------------------------


# ------------------------------------------------------------
# 10. Training
# ------------------------------------------------------------
print("[INFO] Starting Training...")

trained_model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    epochs=20,
    lr=1e-3
)
# ------------------------------------------------------------


# ------------------------------------------------------------
# 11. Save Trained Model
# ------------------------------------------------------------
MODEL_PATH = "ecg_model_v1.pth"
torch.save(trained_model.state_dict(), MODEL_PATH)
print(f"[INFO] Model saved: {MODEL_PATH}")
# ------------------------------------------------------------


# ------------------------------------------------------------
# 12. Ensemble Verification
# ------------------------------------------------------------
print("\n[INFO] Validating Ensemble Logic...")

models = [trained_model, trained_model]

X_sample, _ = next(iter(val_loader))
X_sample = X_sample.to(device)

ensemble_preds = ensemble_predict(models, X_sample)
print(f"Ensemble Prediction Shape: {ensemble_preds.shape}")
# ------------------------------------------------------------
