"""
Stage 3 — Sleep-EDF Data Loader (Chennu 2014 / DOC Proxy)

Sleep-EDF maps onto the DOC clinical hierarchy:

    Sleep stage   →  DOC analogue   →  Class index
    ───────────────────────────────────────────────
    Wake (W)      →  Healthy Control (HC)   →  2
    REM / N1      →  MCS (Minimally Conscious) → 1
    N2 / N3 (SWS) →  UWS/VS (unresponsive)     → 0

Rationale:
  • Global cortical synchrony in N2/N3 resembles the disconnected EEG patterns
    seen in UWS/VS patients (Massimini et al. 2005; Casali et al. 2013).
  • REM/N1 show reactive, intermediate connectivity analogous to MCS.
  • Wake EEG matches healthy-control high-complexity patterns.
  This mapping enables a full Coma→Conscious hierarchy before clinical
  DOC data (Chennu 2014) is received.

Dataset:
  PhysioNet Sleep-EDF Expanded (EDF-X)
  https://physionet.org/content/sleep-edfx/1.0.0/
  ► 197 subjects, PSG recordings with EEG Fpz-Cz and Pz-Oz channels + hypnogram

Download command (run once from project root):
  wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/ -P data/sleep_edf/

Expected directory layout:
  data/sleep_edf/
    SC4001E0-PSG.edf    ... (PSG recordings — EEG + EOG + EMG)
    SC4001EC-Hypnogram.edf  ... (annotation files)
    SC4002E0-PSG.edf
    ...

Usage:
    from src.stage3_doc.dataset_sleepedf import SleepEDFDataset, get_sleepedf_loader
    ds = SleepEDFDataset()
    loader = get_sleepedf_loader(batch_size=32)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_ROOT,
    SFREQ_CHENNU,        # 250 Hz — also the Sleep-EDF native rate
    BANDPASS_LOW,
    BANDPASS_HIGH,
    NOTCH_FREQ,
    EPOCH_SEC,
    EPOCH_OVERLAP,
    AMP_THRESHOLD,
    STAGE3_BATCH_SIZE,
    RANDOM_SEED,
    NUM_DOC_CLASSES,
)
from src.utils.eeg_utils import (
    bandpass_filter,
    notch_filter,
    epoch_signal,
    reject_artifacts,
    zscore_epochs,
)

log = logging.getLogger(__name__)

SLEEP_EDF_DIR = DATA_ROOT / "sleep_edf"

# ─────────────────────────────────────────────────────────────────────────────
# Stage-to-DOC label mapping
# ─────────────────────────────────────────────────────────────────────────────

# Sleep-EDF hypnogram annotations use these string labels
_SLEEP_STAGE_TO_DOC: Dict[str, int] = {
    # Wake →  HC  (class 2)
    "Sleep stage W":  2,
    "W":              2,
    # REM  →  MCS (class 1)
    "Sleep stage R":  1,
    "R":              1,
    # N1   →  MCS (class 1, lightest NREM)
    "Sleep stage 1":  1,
    "1":              1,
    # N2   →  UWS (class 0)
    "Sleep stage 2":  0,
    "2":              0,
    # N3/SWS → UWS (class 0)
    "Sleep stage 3":  0,
    "3":              0,
    "Sleep stage 4":  0,   # older Rechtschaffen & Kales scoring
    "4":              0,
    # Movement artefacts — skip
    "Sleep stage ?":  -1,
    "?":              -1,
}

# EEG channels to use (Sleep-EDF has Fpz-Cz and Pz-Oz)
_EEG_CHANNEL_NAMES = ["EEG Fpz-Cz", "EEG Pz-Oz"]
N_SLEEPEDF_CHANNELS = len(_EEG_CHANNEL_NAMES)   # 2

# Native Sleep-EDF sampling rate
_SLEEP_EDF_SFREQ = 100.0   # Hz — EEG channels in Sleep-EDF are 100 Hz


# ─────────────────────────────────────────────────────────────────────────────
# File discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_psg_hypnogram_pairs(
    root: Path,
) -> List[Tuple[Path, Path]]:
    """
    Return list of (psg_path, hypnogram_path) tuples.

    Sleep-EDF naming:
        SC4001E0-PSG.edf  ↔  SC4001EC-Hypnogram.edf
        SC4011E0-PSG.edf  ↔  SC4011EC-Hypnogram.edf
    """
    psg_files  = sorted(root.rglob("*-PSG.edf"))
    pairs: List[Tuple[Path, Path]] = []

    for psg in psg_files:
        # Derive hypnogram filename: replace last part after subject code
        stem = psg.stem            # e.g. 'SC4001E0-PSG'
        prefix = stem[:6]          # e.g. 'SC4001'
        hyp_candidates = list(root.rglob(f"{prefix}*Hypnogram.edf"))
        if not hyp_candidates:
            log.debug(f"  No hypnogram for {psg.name}, skipping")
            continue
        pairs.append((psg, hyp_candidates[0]))

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Per-recording loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_one_recording(
    psg_path:  Path,
    hyp_path:  Path,
    epoch_sec: float = EPOCH_SEC,   # 4 s default (30 s in clinical PSG)
    amp_thresh: float = AMP_THRESHOLD,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load a single PSG + hypnogram pair.

    Returns
    -------
    (epochs, labels) where:
        epochs : (n_clean_epochs, N_SLEEPEDF_CHANNELS, epoch_samples)
        labels : (n_clean_epochs,)  int  DOC class index  {0,1,2}
    Returns None on failure.
    """
    try:
        import mne
        mne.set_log_level("ERROR")
    except ImportError:
        raise ImportError("pip install mne")

    # ── Load PSG ──────────────────────────────────────────────────────────────
    try:
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    except Exception as exc:
        log.debug(f"  PSG read error {psg_path.name}: {exc}")
        return None

    # Pick EEG channels; gracefully handle name variants
    available_ch = {c.upper(): c for c in raw.ch_names}
    eeg_picks = []
    for want in _EEG_CHANNEL_NAMES:
        if want.upper() in available_ch:
            eeg_picks.append(available_ch[want.upper()])
    if not eeg_picks:
        # Try picking any EEG-typed channels as fallback
        eeg_picks = [c for c in raw.ch_names if "EEG" in c.upper()][:2]
    if not eeg_picks:
        log.debug(f"  No EEG channels in {psg_path.name}")
        return None

    raw.pick_channels(eeg_picks)
    sfreq    = raw.info["sfreq"]   # typically 100 Hz
    data_uv  = raw.get_data() * 1e6   # V → µV  (n_ch, n_samples)

    if data_uv.shape[0] < 1:
        return None

    # If only 1 channel found, duplicate it to match N_SLEEPEDF_CHANNELS
    if data_uv.shape[0] == 1:
        data_uv = np.repeat(data_uv, N_SLEEPEDF_CHANNELS, axis=0)

    # Clip to exactly N_SLEEPEDF_CHANNELS
    data_uv = data_uv[:N_SLEEPEDF_CHANNELS]

    # ── Load hypnogram ────────────────────────────────────────────────────────
    try:
        annot = mne.read_annotations(str(hyp_path))
    except Exception as exc:
        log.debug(f"  Hypnogram read error {hyp_path.name}: {exc}")
        return None

    # Build sample-level label array
    n_samples       = data_uv.shape[-1]
    sample_labels   = np.full(n_samples, fill_value=-1, dtype=np.int8)
    for onset, duration, desc in zip(annot.onset, annot.duration, annot.description):
        doc_label = _SLEEP_STAGE_TO_DOC.get(desc, -1)
        if doc_label == -1:
            continue
        onset_samp = int(onset * sfreq)
        end_samp   = min(int((onset + duration) * sfreq), n_samples)
        sample_labels[onset_samp:end_samp] = doc_label

    # ── Filter ────────────────────────────────────────────────────────────────
    try:
        filtered = bandpass_filter(data_uv, sfreq,
                                    low=BANDPASS_LOW, high=min(BANDPASS_HIGH, sfreq / 2 - 1))
        # Sleep-EDF 100 Hz → notch at 50 Hz would alias; skip if sfreq < 2*notch
        if sfreq > 2 * NOTCH_FREQ:
            filtered = notch_filter(filtered, sfreq, freq=NOTCH_FREQ)
    except Exception as exc:
        log.debug(f"  Filter error {psg_path.name}: {exc}")
        return None

    # ── Epoch ────────────────────────────────────────────────────────────────
    epoch_len   = int(epoch_sec * sfreq)
    step        = int(epoch_len * (1.0 - EPOCH_OVERLAP))

    epoch_list:  List[np.ndarray] = []
    label_list:  List[int]        = []

    for start in range(0, n_samples - epoch_len + 1, step):
        end   = start + epoch_len
        epoch = filtered[:, start:end]

        # Majority-vote label for this window
        window_labels = sample_labels[start:end]
        valid = window_labels[window_labels >= 0]
        if len(valid) < epoch_len * 0.8:   # need ≥80% labelled samples
            continue
        counts     = np.bincount(valid.astype(np.int32), minlength=NUM_DOC_CLASSES)
        doc_label  = int(counts.argmax())
        epoch_list.append(epoch)
        label_list.append(doc_label)

    if not epoch_list:
        return None

    epochs_arr = np.stack(epoch_list, axis=0).astype(np.float32)  # (N, C, T)
    labels_arr = np.array(label_list, dtype=np.int64)

    # ── Artifact rejection ───────────────────────────────────────────────────
    clean, mask = reject_artifacts(epochs_arr, amp_thresh=amp_thresh)
    labels_arr  = labels_arr[mask]
    if clean.shape[0] == 0:
        return None

    # ── Normalise ────────────────────────────────────────────────────────────
    clean = zscore_epochs(clean)

    return clean, labels_arr


# ─────────────────────────────────────────────────────────────────────────────
# Build corpus arrays
# ─────────────────────────────────────────────────────────────────────────────

def build_sleepedf_epochs(
    root:            Path  = SLEEP_EDF_DIR,
    max_subjects:    int   = 197,
    epoch_sec:       float = EPOCH_SEC,
    amp_thresh:      float = AMP_THRESHOLD,
    verbose:         bool  = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all Sleep-EDF recordings and return epoch arrays.

    Returns
    -------
    epochs      : (N, N_SLEEPEDF_CHANNELS, T)
    labels      : (N,)   DOC class  {0=UWS, 1=MCS, 2=HC}
    subject_ids : (N,)   int subject index (for LOSO-CV)
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Sleep-EDF directory not found: {root}\n"
            f"Download with:\n"
            f"  wget -r -N -c -np "
            f"https://physionet.org/files/sleep-edfx/1.0.0/ -P {root}"
        )

    pairs = _find_psg_hypnogram_pairs(root)
    if not pairs:
        raise FileNotFoundError(
            f"No PSG/Hypnogram pairs found in {root}.  "
            f"Ensure files match pattern *-PSG.edf / *Hypnogram.edf"
        )

    pairs = pairs[:max_subjects]
    log.info(f"Sleep-EDF: {len(pairs)} recordings found, loading up to {max_subjects}")

    all_epochs:   List[np.ndarray] = []
    all_labels:   List[np.ndarray] = []
    all_subj_ids: List[np.ndarray] = []
    loaded = 0

    for subj_idx, (psg, hyp) in enumerate(pairs):
        result = _load_one_recording(psg, hyp, epoch_sec=epoch_sec,
                                     amp_thresh=amp_thresh)
        if result is None:
            continue
        epochs, labels = result
        all_epochs.append(epochs)
        all_labels.append(labels)
        all_subj_ids.append(np.full(len(labels), subj_idx, dtype=np.int64))
        loaded += 1

        if verbose and loaded % 20 == 0:
            n_so_far = sum(e.shape[0] for e in all_epochs)
            log.info(f"  Loaded {loaded} recordings ({n_so_far:,} epochs so far)")

    if not all_epochs:
        raise RuntimeError(
            "No usable Sleep-EDF recordings loaded.  "
            "Check file paths and MNE installation."
        )

    epochs_arr  = np.concatenate(all_epochs,   axis=0).astype(np.float32)
    labels_arr  = np.concatenate(all_labels,   axis=0).astype(np.int64)
    subj_arr    = np.concatenate(all_subj_ids, axis=0).astype(np.int64)

    # Log class distribution
    for cls, name in [(0, "UWS/N2-N3"), (1, "MCS/REM-N1"), (2, "HC/Wake")]:
        n = (labels_arr == cls).sum()
        log.info(f"  Class {cls} ({name}): {n:,} epochs  ({100*n/len(labels_arr):.1f}%)")

    return epochs_arr, labels_arr, subj_arr


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class _SubjectSplitDataset(Dataset):
    """Internal helper."""
    def __init__(self, epochs: np.ndarray, labels: np.ndarray) -> None:
        self.epochs = torch.from_numpy(epochs)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.epochs[idx], self.labels[idx]


class SleepEDFDataset(Dataset):
    """
    Sleep-EDF dataset with subject-aware LOSO-CV splits.

    Parameters
    ----------
    root         : path to Sleep-EDF data directory
    max_subjects : cap on number of subjects to load
    epoch_sec    : epoch length in seconds

    Attributes
    ----------
    epochs      : (N, C, T) float32 tensor
    labels      : (N,) int64 tensor  — DOC class {0,1,2}
    subject_ids : (N,) int64 tensor  — subject indices for LOSO
    n_channels  : int  — always N_SLEEPEDF_CHANNELS (2)
    """

    def __init__(
        self,
        root:         Path  = SLEEP_EDF_DIR,
        max_subjects: int   = 197,
        epoch_sec:    float = EPOCH_SEC,
    ) -> None:
        epochs, labels, subj_ids = build_sleepedf_epochs(
            root=root,
            max_subjects=max_subjects,
            epoch_sec=epoch_sec,
        )
        self.epochs      = torch.from_numpy(epochs)
        self.labels      = torch.from_numpy(labels)
        self.subject_ids = torch.from_numpy(subj_ids)
        self.n_channels  = N_SLEEPEDF_CHANNELS

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.epochs[idx], self.labels[idx]

    def get_subject_split(
        self,
        test_subject: int,
    ) -> Tuple[Dataset, Dataset]:
        """
        Return (train_dataset, test_dataset) for LOSO cross-validation.

        test_subject : subject index to hold out
        """
        test_mask  = (self.subject_ids == test_subject)
        train_mask = ~test_mask

        train_ds = _SubjectSplitDataset(
            self.epochs[train_mask].numpy(),
            self.labels[train_mask].numpy(),
        )
        test_ds = _SubjectSplitDataset(
            self.epochs[test_mask].numpy(),
            self.labels[test_mask].numpy(),
        )
        return train_ds, test_ds

    @property
    def n_subjects(self) -> int:
        return int(self.subject_ids.max().item()) + 1

    def class_counts(self) -> Dict[int, int]:
        return {
            cls: int((self.labels == cls).sum().item())
            for cls in range(NUM_DOC_CLASSES)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def get_sleepedf_loader(
    root:         Path = SLEEP_EDF_DIR,
    max_subjects: int  = 197,
    batch_size:   int  = STAGE3_BATCH_SIZE,
    num_workers:  int  = 0,
) -> Tuple[DataLoader, int]:
    """
    Build the full Sleep-EDF DataLoader (no subject split).
    Returns (loader, n_channels).
    """
    ds     = SleepEDFDataset(root=root, max_subjects=max_subjects)
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
        generator   = torch.Generator().manual_seed(RANDOM_SEED),
    )
    log.info(
        f"Sleep-EDF DataLoader: {len(ds):,} epochs, "
        f"{ds.n_subjects} subjects, {ds.n_channels} channels"
    )
    return loader, ds.n_channels


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

    log.info("Sleep-EDF smoke test — loading 3 subjects …")
    try:
        ds = SleepEDFDataset(max_subjects=3)
        log.info(f"  Total epochs  : {len(ds)}")
        log.info(f"  Channels      : {ds.n_channels}")
        log.info(f"  Class counts  : {ds.class_counts()}")

        x, y = ds[0]
        log.info(f"  Epoch shape   : {x.shape}")
        log.info(f"  Label         : {y.item()} (0=UWS, 1=MCS, 2=HC)")
        log.info("  Smoke test PASSED ✓")
    except FileNotFoundError as exc:
        log.error(str(exc))
