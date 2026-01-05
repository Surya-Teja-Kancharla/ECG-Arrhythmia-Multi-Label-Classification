# ------------------------------------------------------------
# 1. Pandas backward-compatibility shim (MUST BE FIRST)
# ------------------------------------------------------------
import sys
import types
import pandas as pd

numeric_module = types.ModuleType("pandas.core.indexes.numeric")

class Int64Index(pd.Index):
    pass

numeric_module.Int64Index = Int64Index
sys.modules["pandas.core.indexes.numeric"] = numeric_module
# ------------------------------------------------------------

import os
import shutil
import pickle
import numpy as np

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ============================================================
# CONFIG
# ============================================================
SOURCE_DIR = "data/processed/train/preprocessed/250Hz/60s"

TRAIN_OUT = "data/processed/train_split/250Hz/60s"
VAL_OUT   = "data/processed/val_split/250Hz/60s"

VAL_RATIO = 0.2
RANDOM_STATE = 42

# ============================================================
# PREP OUTPUT DIRECTORIES
# ============================================================
os.makedirs(TRAIN_OUT, exist_ok=True)
os.makedirs(VAL_OUT, exist_ok=True)

# ============================================================
# LOAD FILE LIST & LABELS (NO SIGNAL LOADING)
# ============================================================
files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith(".pk")])

labels = []

print("[INFO] Reading labels for stratified split...")

for f in files:
    with open(os.path.join(SOURCE_DIR, f), "rb") as file:
        _, label_dict = pickle.load(file)
        labels.append(label_dict["classes_one_hot"].values)

labels = np.array(labels)  # shape: (N, 9)

print(f"[INFO] Total samples: {len(files)}")
print(f"[INFO] Total labels per class: {labels.sum(axis=0)}")

# ============================================================
# MULTI-LABEL STRATIFIED SPLIT
# ============================================================
splitter = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=VAL_RATIO,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

train_files = [files[i] for i in train_idx]
val_files   = [files[i] for i in val_idx]

print(f"[INFO] Train split size: {len(train_files)}")
print(f"[INFO] Val split size  : {len(val_files)}")

# ============================================================
# COPY FILES (NON-DESTRUCTIVE)
# ============================================================
print("[INFO] Copying train files...")
for f in train_files:
    shutil.copy2(
        os.path.join(SOURCE_DIR, f),
        os.path.join(TRAIN_OUT, f)
    )

print("[INFO] Copying validation files...")
for f in val_files:
    shutil.copy2(
        os.path.join(SOURCE_DIR, f),
        os.path.join(VAL_OUT, f)
    )

print("[INFO] Stratified train/val split completed successfully.")
