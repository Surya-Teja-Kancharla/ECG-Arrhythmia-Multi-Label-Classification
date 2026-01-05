# src/evaluate.py

import numpy as np
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    f1_score,
    roc_auc_score
)

def evaluate_multilabel(y_true, y_prob):
    """
    Evaluates with ADAPTIVE thresholding.
    """
    best_threshold = 0.5
    best_f1 = 0.0
    
    # FIX: Scan lower thresholds (0.01 to 0.50)
    # This catches predictions when model confidence is low (early training)
    thresholds = np.concatenate([
        np.arange(0.01, 0.1, 0.01),  # Fine scan 0.01 - 0.09
        np.arange(0.1, 0.55, 0.05)   # Standard scan 0.1 - 0.5
    ])
    
    for thresh in thresholds:
        y_pred_temp = (y_prob >= thresh).astype(int)
        f1_temp = f1_score(y_true, y_pred_temp, average="macro")
        
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = thresh

    # Final Predictions
    y_pred = (y_prob >= best_threshold).astype(int)

    metrics = {
        "Best Threshold": best_threshold,
        "Hamming Loss": hamming_loss(y_true, y_pred),
        "Subset Accuracy": accuracy_score(y_true, y_pred),
        "F1 Macro": f1_score(y_true, y_pred, average="macro"),
        "F1 Micro": f1_score(y_true, y_pred, average="micro"),
    }

    auc_scores = {}
    for i in range(y_true.shape[1]):
        try:
            if len(np.unique(y_true[:, i])) > 1:
                auc_scores[f"Class_{i}_AUC"] = roc_auc_score(y_true[:, i], y_prob[:, i])
            else:
                auc_scores[f"Class_{i}_AUC"] = 0.5
        except ValueError:
            auc_scores[f"Class_{i}_AUC"] = 0.0

    return metrics, auc_scores