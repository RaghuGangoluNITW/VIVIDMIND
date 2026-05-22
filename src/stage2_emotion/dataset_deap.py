"""
DEAP Dataset Loader  (Stage 2 — emotion encoder pre-training)

Dataset: Koelstra et al. 2012. DEAP: A Database for Emotion Analysis using
         Physiological Signals. IEEE T-AFFCOMP 3(1):18-31.

Structure of each s01.mat – s32.mat:
    data   : (40, 40, 8064)   — 40 trials × 40 channels × 8064 samples
    labels : (40,  4)         — valence / arousal / dominance / liking  (1–9)

Channels 0–31  : 32 EEG
Channels 32–39 : peripheral (not used here)
Sampling rate  : 128 Hz (already pre-processed by DEAP authors)
Baseline       : first 3 s (384 samples) per trial — removed

Binary label: valence ≥ 5 → high valence (1), else low valence (0)
             (configurable via DEAP_LABEL_COL and DEAP_THRESHOLD)
"""

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

from src.config import (
    DEAP_DIR,
    N_EEG_CHANNELS_DEAP,
    SFREQ_DEAP,
    EPOCH_SEC,
    EPOCH_OVERLAP,
    AMP_THRESHOLD,
    DEAP_LABEL_COL,
    DEAP_THRESHOLD,
    DEAP_N_SUBJECTS,
)
from src.utils.eeg_utils import preprocess_recording


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_subject(subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one DEAP subject.

    Returns
    -------
    epochs : (n_epochs, 32, epoch_samples)
    labels : (n_epochs,)  int  {0, 1}
    """
    fname = os.path.join(DEAP_DIR, f"s{subject_id:02d}.mat")
    mat   = sio.loadmat(fname)

    raw_data   = mat["data"]    # (40, 40, 8064)
    raw_labels = mat["labels"]  # (40, 4)

    all_epochs, all_labels = [], []

    for trial_idx in range(raw_data.shape[0]):
        # Select 32 EEG channels, drop 3 s baseline (384 samples @ 128 Hz)
        eeg = raw_data[trial_idx, :N_EEG_CHANNELS_DEAP, 384:]   # (32, 7680)

        # Preprocess: filter → epoch → reject artefacts → z-score
        epochs, _ = preprocess_recording(
            eeg,
            sfreq       = SFREQ_DEAP,
            epoch_sec   = EPOCH_SEC,
            overlap     = EPOCH_OVERLAP,
            amp_thresh  = AMP_THRESHOLD,
        )

        if len(epochs) == 0:
            continue

        # Binarise label
        raw_score = raw_labels[trial_idx, DEAP_LABEL_COL]
        label     = int(raw_score >= DEAP_THRESHOLD)

        all_epochs.append(epochs)
        all_labels.extend([label] * len(epochs))

    if len(all_epochs) == 0:
        return np.empty((0, N_EEG_CHANNELS_DEAP, 0)), np.empty(0, dtype=int)

    return np.concatenate(all_epochs, axis=0), np.array(all_labels, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class DEAPDataset(Dataset):
    """
    PyTorch Dataset for DEAP.

    Parameters
    ----------
    subject_ids : list of subject indices 1..32 to include
    preloaded   : if True, all data is kept in RAM; otherwise loaded per call
                  (always True for this dataset — files are small)
    """

    def __init__(
        self,
        subject_ids: Optional[List[int]] = None,
    ) -> None:
        if subject_ids is None:
            subject_ids = list(range(1, DEAP_N_SUBJECTS + 1))

        epochs_list, labels_list, subj_list = [], [], []
        for sid in subject_ids:
            ep, lb = _load_subject(sid)
            if len(ep) == 0:
                continue
            epochs_list.append(ep)
            labels_list.append(lb)
            subj_list.extend([sid] * len(ep))

        self.epochs  = np.concatenate(epochs_list, axis=0).astype(np.float32)
        self.labels  = np.concatenate(labels_list, axis=0).astype(np.int64)
        self.subjects = np.array(subj_list, dtype=np.int32)

    # ── Dataset API ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.epochs[idx])   # (32, T)
        y = torch.tensor(self.labels[idx])
        return x, y

    # ── Helpers for LOSO-CV ──────────────────────────────────────────────────

    def get_subject_split(
        self, test_subject: int
    ) -> Tuple["DEAPDataset", "DEAPDataset"]:
        """
        Return (train_dataset, test_dataset) for Leave-One-Subject-Out CV.
        Works on the *already loaded* arrays — no disk re-read.
        """
        test_mask  = self.subjects == test_subject
        train_mask = ~test_mask

        train_ds = _SubsetDataset(
            self.epochs[train_mask], self.labels[train_mask]
        )
        test_ds  = _SubsetDataset(
            self.epochs[test_mask],  self.labels[test_mask]
        )
        return train_ds, test_ds


class _SubsetDataset(Dataset):
    """Lightweight wrapper for numpy arrays — avoids re-loading DEAP."""

    def __init__(self, epochs: np.ndarray, labels: np.ndarray) -> None:
        self.epochs = epochs.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.epochs[idx]),
            torch.tensor(self.labels[idx]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check (run as script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading DEAP — all subjects …")
    ds = DEAPDataset()
    print(f"Total epochs : {len(ds)}")
    x, y = ds[0]
    print(f"Epoch shape  : {x.shape}")
    print(f"Label        : {y.item()}")
    pos = (ds.labels == 1).sum()
    neg = (ds.labels == 0).sum()
    print(f"Class balance: pos={pos}  neg={neg}  ratio={pos/max(neg,1):.2f}")
