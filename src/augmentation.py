# src/augmentation.py

import numpy as np
import random

class ECGAugmentor:
    """
    ECG augmentation for robustness.
    Applied ONLY during training.
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        lead_dropout_prob: float = 0.3,
        max_dropped_leads: int = 2,
        time_mask_prob: float = 0.3,
        max_mask_fraction: float = 0.1
    ):
        self.noise_std = noise_std
        self.lead_dropout_prob = lead_dropout_prob
        self.max_dropped_leads = max_dropped_leads
        self.time_mask_prob = time_mask_prob
        self.max_mask_fraction = max_mask_fraction
        
        # Initialize Random Generator for direct float32 generation
        self.rng = np.random.default_rng()

    # --------------------------------------------------
    # Gaussian Noise (Memory Optimized)
    # --------------------------------------------------
    def add_gaussian_noise(self, ecg: np.ndarray) -> np.ndarray:
        # Generate float32 noise DIRECTLY (no intermediate float64 array)
        # standard_normal generates mean=0, std=1. We scale it by noise_std.
        noise = self.rng.standard_normal(ecg.shape, dtype=np.float32) * self.noise_std
        
        # In-place addition to save memory
        ecg += noise 
        return ecg

    # --------------------------------------------------
    # Lead Dropout
    # --------------------------------------------------
    def lead_dropout(self, ecg: np.ndarray) -> np.ndarray:
        if random.random() > self.lead_dropout_prob:
            return ecg

        # Operate on a copy to avoid corrupting original data
        ecg_aug = ecg.copy()
        num_leads = ecg.shape[0]
        k = random.randint(1, self.max_dropped_leads)

        drop_leads = random.sample(range(num_leads), k)
        ecg_aug[drop_leads, :] = 0.0

        return ecg_aug

    # --------------------------------------------------
    # Time Masking
    # --------------------------------------------------
    def time_mask(self, ecg: np.ndarray) -> np.ndarray:
        if random.random() > self.time_mask_prob:
            return ecg

        ecg_aug = ecg.copy()
        T = ecg.shape[1]

        mask_len = int(T * random.uniform(0.02, self.max_mask_fraction))
        start = random.randint(0, T - mask_len)

        ecg_aug[:, start:start + mask_len] = 0.0
        return ecg_aug

    # --------------------------------------------------
    # Full augmentation pipeline
    # --------------------------------------------------
    def augment(self, ecg: np.ndarray) -> np.ndarray:
        # Ensure input is float32 copy before modifying
        ecg_out = ecg.astype(np.float32, copy=True)

        ecg_out = self.add_gaussian_noise(ecg_out)
        ecg_out = self.lead_dropout(ecg_out)
        ecg_out = self.time_mask(ecg_out)

        return ecg_out