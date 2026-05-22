"""
Stage 1 — TUH EEG Data Loader

Loads the Temple University Hospital (TUH) EEG corpus for self-supervised
FractalSSL pre-training.

Dataset structure (v2.0.1):
    data/tuh_eeg/v2.0.1/edf/
        {batch}/{patient}/{session}/{montage}/{recording}.edf

Montage types encountered:
    01_tcp_ar  — TCP AR (linked average reference)
    02_tcp_le  — TCP LE (linked ears)

Channel strategy:
    We extract the 19 canonical 10-20 electrodes present in both montages.
    Channels are matched by a flexible name lookup that strips montage suffixes
    (e.g. "EEG FP1-REF" → "FP1", "EEG FP1-LE" → "FP1").
    Recordings that yield fewer than MIN_CHANNELS clean channels are skipped.

Output:
    TUHDataset  — torch Dataset yielding (epoch_tensor, 0) tuples
                  epoch_tensor shape: (N_TUH_CHANNELS, epoch_samples)
    get_tuh_loader() — convenience wrapper returning a DataLoader ready for
                       FractalSSL.pretrain()

Usage:
    from src.stage1_pretrain.dataset_tuh import get_tuh_loader
    loader = get_tuh_loader(max_recordings=500)
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TUH_DIR,
    SFREQ_TUH,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    NOTCH_FREQ,
    EPOCH_SEC,
    EPOCH_OVERLAP,
    AMP_THRESHOLD,
    STAGE2_BATCH_SIZE,
    RANDOM_SEED,
)
from src.utils.eeg_utils import (
    bandpass_filter,
    notch_filter,
    epoch_signal,
    reject_artifacts,
    zscore_epochs,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Channel map — 19 canonical 10-20 electrodes
# ─────────────────────────────────────────────────────────────────────────────

# Short names (order preserved → consistent channel index across recordings)
CANONICAL_CHANNELS: List[str] = [
    "FP1", "FP2",
    "F3",  "F4",
    "C3",  "C4",
    "P3",  "P4",
    "O1",  "O2",
    "F7",  "F8",
    "T3",  "T4",   # same as T7/T8 in TUH
    "T5",  "T6",   # same as P7/P8 in TUH
    "FZ",  "CZ",  "PZ",
]

# TUH uses various suffixes; map alternative names → canonical
_ALIAS: dict = {
    "T7": "T3", "T8": "T4",
    "P7": "T5", "P8": "T6",
    "FPZ": "FP2",          # rare; just collapse to nearest
}

N_TUH_CHANNELS = len(CANONICAL_CHANNELS)   # 19
MIN_CHANNELS   = 16   # skip recording if fewer channels found after matching


def _strip_to_canonical(raw_name: str) -> Optional[str]:
    """
    'EEG FP1-REF'  →  'FP1'
    'EEG T7-LE'    →  'T3'   (via alias)
    'ECG EKG-REF'  →  None   (not in 10-20)
    """
    # Remove 'EEG ' prefix and reference suffix (e.g. '-REF', '-LE', '-REF2')
    name = raw_name.upper().strip()
    if name.startswith("EEG "):
        name = name[4:]
    if "-" in name:
        name = name.split("-")[0]
    name = name.strip()
    name = _ALIAS.get(name, name)
    return name if name in CANONICAL_CHANNELS else None


# ─────────────────────────────────────────────────────────────────────────────
# Single-file loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_edf(
    edf_path: Path,
    target_sfreq: float = SFREQ_TUH,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Load one .edf file with MNE, resample to target_sfreq, and return
    (data_array, channel_list) where data_array has shape (n_ch, n_samples).

    Returns None on any error (corrupt file, missing library, etc.).
    """
    try:
        import mne
        mne.set_log_level("ERROR")   # suppress verbose MNE output
    except ImportError:
        raise ImportError(
            "MNE-Python is required for TUH loading.  "
            "Install with:  pip install mne"
        )

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as exc:
        log.debug(f"Skipping {edf_path.name}: {exc}")
        return None

    # Crop to at most 20 minutes to avoid OOM in FFT resampler for long recordings
    MAX_DURATION_SEC = 1200.0
    if raw.times[-1] > MAX_DURATION_SEC:
        raw.crop(tmax=MAX_DURATION_SEC)

    # Resample if necessary
    current_sfreq = raw.info["sfreq"]
    if abs(current_sfreq - target_sfreq) > 1.0:
        raw.resample(target_sfreq, verbose=False)

    data      = raw.get_data()   # (n_raw_ch, n_samples)  in Volts
    ch_names  = [c.upper() for c in raw.ch_names]
    data_uv   = data * 1e6       # convert V → µV

    return data_uv, ch_names


def _extract_canonical(
    data_uv: np.ndarray,
    ch_names: List[str],
) -> Optional[np.ndarray]:
    """
    Map raw channels onto CANONICAL_CHANNELS ordering.
    Returns (N_TUH_CHANNELS, n_samples) or None if too few channels.
    """
    name_to_idx = {}
    for i, raw_name in enumerate(ch_names):
        canon = _strip_to_canonical(raw_name)
        if canon is not None and canon not in name_to_idx:
            name_to_idx[canon] = i

    found = [c for c in CANONICAL_CHANNELS if c in name_to_idx]
    if len(found) < MIN_CHANNELS:
        return None

    # Build output array; missing channels filled with zeros
    n_samples = data_uv.shape[-1]
    out = np.zeros((N_TUH_CHANNELS, n_samples), dtype=np.float32)
    for slot_idx, canon in enumerate(CANONICAL_CHANNELS):
        if canon in name_to_idx:
            out[slot_idx] = data_uv[name_to_idx[canon]]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Corpus-level scan
# ─────────────────────────────────────────────────────────────────────────────

def _find_edf_files(
    tuh_root: Path,
    max_files: int = 10_000,
) -> List[Path]:
    """
    Recursively walk tuh_root and return up to max_files .edf paths.
    Prefers AR montage (01_tcp_ar) for consistency when both are present.
    """
    edfs = sorted(tuh_root.rglob("*.edf"))
    if not edfs:
        raise FileNotFoundError(
            f"No .edf files found under {tuh_root}.  "
            f"Check TUH_DIR in src/config.py."
        )
    # Prefer AR montage (01_tcp_ar) first for consistency
    ar = [p for p in edfs if "01_tcp_ar" in str(p)]
    le = [p for p in edfs if "02_tcp_le" in str(p)]
    other = [p for p in edfs if p not in ar and p not in le]
    ordered = ar + le + other
    return ordered[:max_files]


# ─────────────────────────────────────────────────────────────────────────────
# Build epoch array from corpus
# ─────────────────────────────────────────────────────────────────────────────

def build_tuh_epochs(
    tuh_root:        Path  = TUH_DIR,
    max_recordings:  int   = 500,
    max_epochs:      int   = 50_000,
    target_sfreq:    float = SFREQ_TUH,
    epoch_sec:       float = EPOCH_SEC,
    overlap:         float = EPOCH_OVERLAP,
    amp_thresh:      float = AMP_THRESHOLD,
    bandpass_low:    float = BANDPASS_LOW,
    bandpass_high:   float = BANDPASS_HIGH,
    notch_freq:      float = NOTCH_FREQ,
    verbose:         bool  = True,
) -> np.ndarray:
    """
    Scan the TUH corpus and return a single numpy array of all clean epochs.

    Returns
    -------
    epochs : np.ndarray, shape (N_epochs, N_TUH_CHANNELS, epoch_samples)
    """
    # Resolve versioned subdirectory (e.g. v2.0.1/edf/)
    edf_root = tuh_root
    for candidate in ["v2.0.1/edf", "v2.0.0/edf", "edf", "."]:
        candidate_path = tuh_root / candidate
        if candidate_path.exists():
            edf_root = candidate_path
            break

    edf_files = _find_edf_files(edf_root, max_files=max_recordings * 5)
    log.info(f"Found {len(edf_files)} .edf files; targeting {max_recordings} recordings")

    all_epochs: List[np.ndarray] = []
    loaded, skipped = 0, 0

    for edf_path in edf_files:
        if loaded >= max_recordings:
            break

        result = _load_edf(edf_path, target_sfreq=target_sfreq)
        if result is None:
            skipped += 1
            continue

        data_uv, ch_names = result
        canonical = _extract_canonical(data_uv, ch_names)
        if canonical is None:
            skipped += 1
            continue

        # Filtering
        try:
            filtered = bandpass_filter(canonical, target_sfreq,
                                        low=bandpass_low, high=bandpass_high)
            filtered = notch_filter(filtered, target_sfreq, freq=notch_freq)
        except Exception as exc:
            log.debug(f"Filter error {edf_path.name}: {exc}")
            skipped += 1
            continue

        # Epoching
        epochs = epoch_signal(filtered, target_sfreq,
                               epoch_sec=epoch_sec, overlap=overlap)
        if epochs.shape[0] == 0:
            skipped += 1
            continue

        # Artifact rejection
        clean, mask = reject_artifacts(epochs, amp_thresh=amp_thresh)
        if clean.shape[0] == 0:
            skipped += 1
            continue

        # Z-score per epoch, cast to float32 immediately to halve RAM
        clean = zscore_epochs(clean).astype(np.float32)

        all_epochs.append(clean)
        loaded += 1

        n_so_far = sum(e.shape[0] for e in all_epochs)
        if verbose and loaded % 50 == 0:
            log.info(f"  Loaded {loaded}/{max_recordings} recordings "
                     f"({n_so_far:,} epochs so far, {skipped} skipped)")

        if n_so_far >= max_epochs:
            log.info(f"  Reached max_epochs={max_epochs:,} cap — stopping early at {loaded} recordings")
            break

    if not all_epochs:
        raise RuntimeError(
            f"No usable recordings loaded from {edf_root}.  "
            f"Check channel availability and file integrity."
        )

    all_epochs_arr = np.concatenate(all_epochs, axis=0)  # already float32
    log.info(
        f"TUH corpus: {loaded} recordings, {len(all_epochs_arr):,} epochs, "
        f"shape={all_epochs_arr.shape}, skipped={skipped}"
    )
    return all_epochs_arr


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TUHDataset(Dataset):
    """
    Unlabelled TUH EEG dataset for self-supervised pre-training.

    Each item is (epoch_tensor, 0) where epoch_tensor has shape
    (N_TUH_CHANNELS, epoch_samples).  The '0' dummy label is required by
    the `pretrain()` function which expects (x, _) tuples.

    Parameters
    ----------
    epochs : np.ndarray, shape (N, C, T)
        Pre-built epoch array from build_tuh_epochs().
    """

    def __init__(self, epochs: np.ndarray) -> None:
        self.epochs = torch.from_numpy(epochs)   # (N, C, T)

    def __len__(self) -> int:
        return len(self.epochs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.epochs[idx], 0


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def get_tuh_loader(
    tuh_root:       Path = TUH_DIR,
    max_recordings: int  = 500,
    max_epochs:     int  = 50_000,
    batch_size:     int  = STAGE2_BATCH_SIZE,
    num_workers:    int  = 0,
    **epoch_kwargs,
) -> Tuple[DataLoader, int]:
    """
    Build and return a DataLoader for FractalSSL pre-training, plus
    the number of EEG channels (n_channels) needed to instantiate the model.

    Returns
    -------
    loader     : DataLoader  — yields (batch_tensor, dummy_label) each step
    n_channels : int         — always N_TUH_CHANNELS (19)

    Example
    -------
    loader, n_channels = get_tuh_loader(max_recordings=500)
    model = FractalSSL(n_channels=n_channels)
    pretrain(n_channels, loader, device='cuda')
    """
    epochs = build_tuh_epochs(
        tuh_root=tuh_root,
        max_recordings=max_recordings,
        max_epochs=max_epochs,
        **epoch_kwargs,
    )
    ds     = TUHDataset(epochs)
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,    # ensures consistent batch sizes for NT-Xent
        generator   = torch.Generator().manual_seed(RANDOM_SEED),
    )
    log.info(f"TUH DataLoader: {len(ds):,} epochs, {len(loader)} batches/epoch")
    return loader, N_TUH_CHANNELS


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

    log.info("TUH dataset smoke test — loading up to 10 recordings …")
    try:
        loader, n_ch = get_tuh_loader(max_recordings=10, batch_size=8)
        batch, labels = next(iter(loader))
        log.info(f"  Batch shape : {batch.shape}   (B, C={n_ch}, T)")
        log.info(f"  dtype       : {batch.dtype}")
        log.info(f"  value range : [{batch.min():.2f}, {batch.max():.2f}]")
        log.info("  Smoke test PASSED ✓")
    except FileNotFoundError as e:
        log.error(f"TUH data not found: {e}")
        log.error("Set TUH_DIR in src/config.py and ensure EDF files are present.")
