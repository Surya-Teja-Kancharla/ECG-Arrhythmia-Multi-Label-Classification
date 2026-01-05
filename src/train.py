# src/train.py
import torch
import numpy as np
from tqdm import tqdm
from src.evaluate import evaluate_multilabel

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    epochs=20,
    lr=1e-3
):
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"[INFO] Training on {device}")

    # 2. Setup
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # scheduler to reduce LR if loss stagnates (helps convergence)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 3. Training Loop
    for epoch in range(epochs):
        # -------- TRAIN --------
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for X, y in pbar:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # -------- VALIDATE --------
        model.eval()
        y_true, y_prob = [], []

        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                X = X.to(device)
                preds = model(X).cpu().numpy()
                y_prob.append(preds)
                y_true.append(y.numpy())

        y_true = np.vstack(y_true)
        y_prob = np.vstack(y_prob)

        # Debug: Print stats to see if model is dead
        print(f"\n[DEBUG] Avg Pred Prob: {y_prob.mean():.4f} | Max Pred Prob: {y_prob.max():.4f}")

        # Calculate Metrics
        metrics, aucs = evaluate_multilabel(y_true, y_prob)

        # Step Scheduler
        scheduler.step(total_loss)

        # Print Summary
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"  Best Threshold: {metrics['Best Threshold']:.2f}")
        print(f"  Hamming Loss: {metrics['Hamming Loss']:.4f}")
        print(f"  F1 Macro: {metrics['F1 Macro']:.4f}")
        print(f"  F1 Micro: {metrics['F1 Micro']:.4f}")

    return model