# src/ensemble.py

import torch


def ensemble_predict(models, X):
    """
    models: list of trained torch models
    X: torch.Tensor (B, 12, T)
    """
    probs = []

    for model in models:
        model.eval()
        with torch.no_grad():
            probs.append(model(X))

    return torch.mean(torch.stack(probs), dim=0)
