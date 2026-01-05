# src/preprocessing.py

import numpy as np
from scipy.signal import butter, filtfilt


class ECGPreprocessor:
    """
    Model-aware ECG preprocessing.
    This does NOT alter dataset-level preprocessing
    (sampling rate, padding, truncation).
    """

    def __init__(
        self,
        sampling_rate: int = 250,
        apply_baseline_correction: bool = False,
        baseline_cutoff: float = 0.5
    ):
        self.fs = sampling_rate
        self.apply_baseline = apply_baseline_correction
        self.baseline_cutoff = baseline_cutoff

    # --------------------------------------------------
    # Lead-wise Z-score normalization (REQUIRED)
    # --------------------------------------------------
    @staticmethod
    def zscore_per_lead(ecg: np.ndarray) -> np.ndarray:
        """
        ecg shape: (12, T)
        """
        mean = ecg.mean(axis=1, keepdims=True)
        std = ecg.std(axis=1, keepdims=True) + 1e-8
        return (ecg - mean) / std

    # --------------------------------------------------
    # Optional baseline drift correction (LIGHT)
    # --------------------------------------------------
    def baseline_correction(self, ecg: np.ndarray) -> np.ndarray:
        """
        High-pass Butterworth filter.
        VERY mild – disabled by default.
        """
        nyq = 0.5 * self.fs
        cutoff = self.baseline_cutoff / nyq
        b, a = butter(N=2, Wn=cutoff, btype="high")
        return filtfilt(b, a, ecg, axis=1)

    # --------------------------------------------------
    # Main preprocessing pipeline
    # --------------------------------------------------
    def preprocess(self, ecg: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline.
        """
        assert ecg.shape[0] == 12, "Expected 12-lead ECG"

        ecg_out = ecg.copy()

        if self.apply_baseline:
            ecg_out = self.baseline_correction(ecg_out)

        ecg_out = self.zscore_per_lead(ecg_out)

        return ecg_out
