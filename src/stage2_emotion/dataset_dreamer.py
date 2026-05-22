"""
DREAMER Dataset Loader  (Stage 2 — emotion encoder pre-training)

Dataset: Katsigiannis & Ramzan 2018. DREAMER: A Database for Emotion
         Recognition Through EEG and ECG Signals from Wireless Low-cost
         Off-the-Shelf Devices. IEEE J-BHI 22(1):98-107.

Structure of DREAMER.mat:
    DREAMER            : struct
      .Data            : (1, 23)  cell — one entry per subject
        {s}.EEG        : struct
          .stimuli     : (1, 18)  cell — EEG during stimulus (14 ch × T)
          .baseline    : (1, 18)  cell — baseline before stimulus (14 ch × T)
        {s}.ScoreValence   : (18, 1)   — integer 1–5
        {s}.ScoreArousal   : (18, 1)   — integer 1–5
        {s}.ScoreDominance : (18, 1)   — integer 1–5

Channels : 14 Emotiv EPOC locations
Fs       : 128 Hz
"""

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

from src.config import (
    DREAMER_MAT,
    SFREQ_DREAMER,
    EPOCH_SEC,
    EPOCH_OVERLAP,
    AMP_THRESHOLD,
    DREAMER_LABEL_COL,
    DREAMER_THRESHOLD,
    DREAMER_N_SUBJECTS,
    N_EEG_CHANNELS_DREAMER,
)
from src.utils.eeg_utils import preprocess_recording


# ─────────────────────────────────────────────────────────────────────────────
# Raw extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_dreamer(mat_path: str):
    """
    Parse the nested DREAMER.mat struct into Python lists.

    Returns
    -------
    subjects_data : list of dict, each has
        'eeg_list'  : list of 18 np arrays shaped (14, T_i)
        'scores'    : (18, 3) array [valence, arousal, dominance]
    """
    raw       = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    dreamer   = raw["DREAMER"]
    data_cell = dreamer.Data         # object array of length 23

    subjects_data = []
    for subj_obj in data_cell:
        eeg_obj   = subj_obj.EEG
        stimuli   = eeg_obj.stimuli  # object array of 18 EEG matrices (14 × T)

        eeg_list = []
        for stim in stimuli:
            arr = np.array(stim, dtype=np.float64)   # (T, 14) or (14, T)
            if arr.ndim == 2 and arr.shape[0] != N_EEG_CHANNELS_DREAMER:
                arr = arr.T                           # ensure (14, T)
            eeg_list.append(arr)

        valence   = np.array(subj_obj.ScoreValence,   dtype=np.float32).flatten()
        arousal   = np.array(subj_obj.ScoreArousal,   dtype=np.float32).flatten()
        dominance = np.array(subj_obj.ScoreDominance, dtype=np.float32).flatten()
        scores    = np.stack([valence, arousal, dominance], axis=1)  # (18, 3)

        subjects_data.append({"eeg_list": eeg_list, "scores": scores})

    return subjects_data


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class DREAMERDataset(Dataset):
    """
    PyTorch Dataset for DREAMER.

    DREAMER_LABEL_COL : 0=valence, 1=arousal, 2=dominance
    DREAMER_THRESHOLD : binarises the chosen score (default 3 → ≥3 = high)
    """

    def __init__(
        self,
        subject_ids: Optional[List[int]] = None,
    ) -> None:
        """
        subject_ids : 0-indexed list (0..22). None → all 23 subjects.
        """
        if subject_ids is None:
            subject_ids = list(range(DREAMER_N_SUBJECTS))

        subjects_data = _extract_dreamer(DREAMER_MAT)

        epochs_list, labels_list, subj_list = [], [], []

        for sid in subject_ids:
            sdata = subjects_data[sid]
            for trial_idx, eeg in enumerate(sdata["eeg_list"]):
                # eeg: (14, T)
                epochs, _ = preprocess_recording(
                    eeg,
                    sfreq      = SFREQ_DREAMER,
                    epoch_sec  = EPOCH_SEC,
                    overlap    = EPOCH_OVERLAP,
                    amp_thresh = AMP_THRESHOLD,
                )
                if len(epochs) == 0:
                    continue

                raw_score = sdata["scores"][trial_idx, DREAMER_LABEL_COL]
                label     = int(raw_score >= DREAMER_THRESHOLD)

                epochs_list.append(epochs)
                labels_list.extend([label] * len(epochs))
                subj_list.extend([sid] * len(epochs))

        if len(epochs_list) == 0:
            self.epochs   = np.empty((0, N_EEG_CHANNELS_DREAMER, 0), dtype=np.float32)
            self.labels   = np.empty(0, dtype=np.int64)
            self.subjects = np.empty(0, dtype=np.int32)
            return

        self.epochs   = np.concatenate(epochs_list, axis=0).astype(np.float32)
        self.labels   = np.array(labels_list, dtype=np.int64)
        self.subjects = np.array(subj_list,   dtype=np.int32)

    # ── Dataset API ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.epochs[idx])   # (14, T)
        y = torch.tensor(self.labels[idx])
        return x, y

    # ── Helpers for LOSO-CV ──────────────────────────────────────────────────

    def get_subject_split(self, test_subject: int) -> Tuple[Dataset, Dataset]:
        test_mask  = self.subjects == test_subject
        train_mask = ~test_mask
        train_ds = _SubsetDataset(self.epochs[train_mask], self.labels[train_mask])
        test_ds  = _SubsetDataset(self.epochs[test_mask],  self.labels[test_mask])
        return train_ds, test_ds


class _SubsetDataset(Dataset):
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
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading DREAMER …")
    ds = DREAMERDataset()
    print(f"Total epochs : {len(ds)}")
    if len(ds) > 0:
        x, y = ds[0]
        print(f"Epoch shape  : {x.shape}")
        print(f"Label        : {y.item()}")
        pos = (ds.labels == 1).sum()
        neg = (ds.labels == 0).sum()
        print(f"Class balance: pos={pos}  neg={neg}  ratio={pos/max(neg,1):.2f}")
