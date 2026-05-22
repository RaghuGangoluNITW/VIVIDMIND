"""
Stage 3 — I-CARE EEG Data Loader

International Cardiac Arrest REsearch (I-CARE) consortium dataset.
PhysioNet 2023 Challenge: "Predicting Neurological Recovery from Coma After Cardiac Arrest"
https://physionet.org/content/i-care/2.1/

CPC → DOC class mapping
───────────────────────
CPC 1  (good recovery, independent)       →  HC   (class 2)
CPC 2  (moderate disability, independent) →  HC   (class 2)
CPC 3  (severe disability)                →  MCS  (class 1)
CPC 4  (unresponsive wakefulness / UWS)   →  UWS  (class 0)
CPC 5  (dead)                             →  skip (-1)

Lorentzian geometry rationale:
  CPC 4 ⊂ CPC 3 ⊂ CPC 1-2 is a strict neurological recovery hierarchy —
  identical in topology to the UWS ⊂ MCS ⊂ Conscious DOC hierarchy.
  H^n naturally embeds this tree structure.

PDI-CCS story:
  Comatose patients labelled CPC 4 whose encoder predictions DISAGREE (high PDI)
  may harbour residual recovery potential. Flagging them prevents premature
  withdrawal of life-support — the "self-fulfilling prophecy" problem.

Data layout (produced by icare_eeg_download.py):
  ~/icare/training/
    {patient_id}/
      {patient_id}.txt                    — clinical metadata (CPC, age, etc.)
      {patient_id}_{seg}_{hour}_EEG.mat   — WFDB MAT4 signal file
      {patient_id}_{seg}_{hour}_EEG.hea   — WFDB header

EEG:
  19 channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8,
               T3, T4, T5, T6, Fz, Cz, Pz
  Native sampling rate: 500 Hz (resampled to SFREQ_ICARE = 256 Hz)
  Duration per file: ~52 min

Download command (run from WSL):
  python3 icare_eeg_download.py --cpc 1 2 3 4 5 --max-patients 50 --max-hours 2

Usage:
  from src.stage3_doc.dataset_icare import ICareDataset, get_icare_loader
  ds = ICareDataset()
  train_ds, test_ds = ds.get_subject_split(test_subject=42)
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
    ICARE_DIR,
    SFREQ_ICARE,
    BANDPASS_LOW,
    BANDPASS_HIGH,
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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Standard 10-20 channel order (same as TUH CANONICAL_CHANNELS)
ICARE_CHANNELS: List[str] = [
    "Fp1", "Fp2",
    "F3",  "F4",
    "C3",  "C4",
    "P3",  "P4",
    "O1",  "O2",
    "F7",  "F8",
    "T3",  "T4",
    "T5",  "T6",
    "Fz",  "Cz",  "Pz",
]
N_ICARE_CHANNELS = len(ICARE_CHANNELS)   # 19

ICARE_NATIVE_SFREQ = 500.0   # Hz — confirmed from .hea headers
ICARE_NOTCH_FREQ   = 50.0    # Hz — confirmed from #Utility frequency in .hea

# CPC → DOC class label
CPC_TO_DOC: Dict[int, int] = {
    1: 2,   # good recovery  →  HC
    2: 2,   # moderate, independent  →  HC
    3: 1,   # severe disability  →  MCS
    4: 0,   # UWS/vegetative  →  UWS
    5: -1,  # dead  →  skip
}


# ─────────────────────────────────────────────────────────────────────────────
# Metadata parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_patient_txt(patient_dir: Path, patient_id: str) -> Optional[Dict[str, str]]:
    """
    Parse {patient_id}.txt for CPC score and other clinical metadata.
    Returns dict or None if the file is missing.
    """
    txt_path = patient_dir / f"{patient_id}.txt"
    if not txt_path.exists():
        return None
    result = {}
    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip()
    return result


def _get_cpc(patient_dir: Path, patient_id: str) -> int:
    """Return integer CPC (1-5) or -1 if unknown."""
    info = _parse_patient_txt(patient_dir, patient_id)
    if info is None:
        return -1
    cpc_raw = info.get("CPC", "").strip()
    try:
        return int(float(cpc_raw))
    except (ValueError, TypeError):
        return -1


# ─────────────────────────────────────────────────────────────────────────────
# WFDB file reading
# ─────────────────────────────────────────────────────────────────────────────

def _read_wfdb_record(record_path: Path) -> Optional[Tuple[np.ndarray, float, List[str]]]:
    """
    Read a WFDB record (pair of .hea + .mat files) and return
    (signal_uv, sfreq, channel_names).

    signal_uv : (n_channels, n_samples) in µV
    sfreq     : sampling frequency in Hz
    ch_names  : list of channel name strings

    Tries wfdb library first; falls back to manual .hea + scipy.io.loadmat.
    """
    record_stem = str(record_path)   # path without extension

    # ── Method 1: wfdb library ────────────────────────────────────────────────
    try:
        import wfdb
        rec = wfdb.rdrecord(record_stem)
        # rec.p_signal: (n_samples, n_channels) in physical units (already µV for EEG)
        data = rec.p_signal.T.astype(np.float32)   # → (n_channels, n_samples)
        sfreq = float(rec.fs)
        ch_names = [c.strip() for c in rec.sig_name]
        return data, sfreq, ch_names
    except ImportError:
        pass   # fall through to manual method
    except Exception as exc:
        log.debug(f"wfdb failed for {record_path.name}: {exc}")
        # fall through

    # ── Method 2: manual .hea + scipy.io.loadmat ─────────────────────────────
    hea_path = Path(record_stem + ".hea")
    mat_path = Path(record_stem + ".mat")

    if not hea_path.exists() or not mat_path.exists():
        return None

    try:
        import scipy.io
    except ImportError:
        raise ImportError("pip install scipy   (or pip install wfdb)")

    # Parse header
    hea_lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_line = hea_lines[0].split()
    n_channels = int(header_line[1])
    sfreq      = float(header_line[2])
    n_samples  = int(header_line[3])

    ch_names: List[str] = []
    gains:    List[float] = []
    baselines: List[float] = []
    for line in hea_lines[1:n_channels + 1]:
        parts = line.split()
        if not parts:
            continue
        ch_names.append(parts[-1].strip())
        # Format: "16+24 gain(baseline)/unit 16 0 ..."
        # Extract gain and baseline from the format field
        try:
            fmt_field = parts[1]   # e.g. "17.980/nu" or "17.98(23877)/nu"
            gain_str  = fmt_field.split("/")[0]
            if "(" in gain_str:
                gain_val  = float(gain_str.split("(")[0])
                base_val  = float(gain_str.split("(")[1].rstrip(")"))
            else:
                gain_val  = float(gain_str)
                base_val  = 0.0
        except (IndexError, ValueError):
            gain_val = 1.0
            base_val = 0.0
        gains.append(gain_val)
        baselines.append(base_val)

    # Load raw integer samples from MAT4 file
    try:
        mat_data = scipy.io.loadmat(str(mat_path))
    except Exception as exc:
        log.debug(f"scipy.io.loadmat failed for {mat_path.name}: {exc}")
        return None

    # WFDB MAT4 stores all channels in a single matrix variable (first array-type key)
    raw_key = None
    for k, v in mat_data.items():
        if not k.startswith("_") and isinstance(v, np.ndarray) and v.ndim == 2:
            raw_key = k
            break
    if raw_key is None:
        return None

    raw = mat_data[raw_key].astype(np.float32)   # (n_channels, n_samples)
    if raw.shape[0] != n_channels:
        if raw.shape[1] == n_channels:
            raw = raw.T
        else:
            log.debug(f"Channel count mismatch in {mat_path.name}")
            return None

    # Convert to physical units (µV)
    gains_arr     = np.array(gains,     dtype=np.float32).reshape(-1, 1)
    baselines_arr = np.array(baselines, dtype=np.float32).reshape(-1, 1)
    data = (raw - baselines_arr) / gains_arr   # µV

    return data, sfreq, ch_names


def _align_channels(
    data: np.ndarray,
    ch_names: List[str],
) -> Optional[np.ndarray]:
    """
    Reorder channels to match ICARE_CHANNELS ordering.
    Channels present in data but not in ICARE_CHANNELS are dropped.
    Missing ICARE_CHANNELS are filled with zeros.
    Returns (N_ICARE_CHANNELS, n_samples) or None if fewer than 16 channels found.
    """
    name_map = {c.upper(): i for i, c in enumerate(ch_names)}
    found = sum(1 for c in ICARE_CHANNELS if c.upper() in name_map)
    if found < 16:
        return None

    n_samples = data.shape[-1]
    out = np.zeros((N_ICARE_CHANNELS, n_samples), dtype=np.float32)
    for slot, canon in enumerate(ICARE_CHANNELS):
        if canon.upper() in name_map:
            out[slot] = data[name_map[canon.upper()]]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Corpus scan and preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _find_eeg_files(patient_dir: Path) -> List[Path]:
    """Return sorted list of *_EEG.mat files in a patient directory."""
    mats = sorted(patient_dir.glob("*_EEG.mat"))
    return mats


def _preprocess_recording(
    mat_path:       Path,
    target_sfreq:   float = SFREQ_ICARE,
    epoch_sec:      float = EPOCH_SEC,
    overlap:        float = EPOCH_OVERLAP,
    amp_thresh:     float = AMP_THRESHOLD,
) -> Optional[np.ndarray]:
    """
    Load, resample, filter, epoch, reject artefacts, and z-score one EEG file.

    Returns (n_epochs, N_ICARE_CHANNELS, epoch_samples) or None.
    """
    record_stem = str(mat_path).replace(".mat", "")
    result = _read_wfdb_record(Path(record_stem))
    if result is None:
        return None

    data, sfreq, ch_names = result
    aligned = _align_channels(data, ch_names)
    if aligned is None:
        return None

    # Resample if needed
    if abs(sfreq - target_sfreq) > 1.0:
        try:
            from scipy.signal import resample_poly
            ratio       = target_sfreq / sfreq
            up          = int(target_sfreq)
            down        = int(sfreq)
            from math import gcd
            g           = gcd(up, down)
            aligned     = resample_poly(aligned, up // g, down // g, axis=-1)
            aligned     = aligned.astype(np.float32)
            sfreq       = target_sfreq
        except Exception as exc:
            log.debug(f"Resample failed {mat_path.name}: {exc}")
            return None

    # Filter: bandpass + notch (50 Hz confirmed in .hea)
    try:
        filtered = bandpass_filter(aligned, sfreq,
                                    low=BANDPASS_LOW, high=min(BANDPASS_HIGH, sfreq / 2 - 1))
        filtered = notch_filter(filtered, sfreq, freq=ICARE_NOTCH_FREQ)
    except Exception as exc:
        log.debug(f"Filter error {mat_path.name}: {exc}")
        return None

    # Epoch
    epochs = epoch_signal(filtered, sfreq, epoch_sec=epoch_sec, overlap=overlap)
    if epochs.shape[0] == 0:
        return None

    # Artifact rejection
    clean, _mask = reject_artifacts(epochs, amp_thresh=amp_thresh)
    if clean.shape[0] == 0:
        return None

    return zscore_epochs(clean).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Build corpus arrays
# ─────────────────────────────────────────────────────────────────────────────

def build_icare_epochs(
    icare_root:   Path  = ICARE_DIR,
    target_sfreq: float = SFREQ_ICARE,
    epoch_sec:    float = EPOCH_SEC,
    amp_thresh:   float = AMP_THRESHOLD,
    verbose:      bool  = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Walk the I-CARE training directory and build epoch arrays.

    Returns
    -------
    epochs      : (N, N_ICARE_CHANNELS, epoch_samples)  float32
    labels      : (N,)   DOC class  {0=UWS, 1=MCS, 2=HC}
    subject_ids : (N,)   int  patient index for LOSO-CV
    """
    if not icare_root.exists():
        raise FileNotFoundError(
            f"I-CARE directory not found: {icare_root}\n"
            f"Run from WSL:\n"
            f"  python3 icare_eeg_download.py --cpc 1 2 3 4 5 --max-patients 50 --max-hours 2\n"
            f"Then check ICARE_DIR in src/config.py points to ~/icare/training"
        )

    patient_dirs = sorted(p for p in icare_root.iterdir() if p.is_dir())
    if not patient_dirs:
        raise FileNotFoundError(f"No patient directories found in {icare_root}")

    log.info(f"I-CARE: scanning {len(patient_dirs)} patient directories …")

    all_epochs:   List[np.ndarray] = []
    all_labels:   List[np.ndarray] = []
    all_subj_ids: List[np.ndarray] = []
    skipped_cpc5 = 0
    skipped_no_eeg = 0
    loaded = 0
    subj_idx = 0

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name

        # Get CPC label
        cpc = _get_cpc(patient_dir, patient_id)
        if cpc == -1:
            log.debug(f"  {patient_id}: no CPC metadata, skipping")
            continue
        doc_label = CPC_TO_DOC.get(cpc, -1)
        if doc_label == -1:
            skipped_cpc5 += 1
            continue   # CPC 5 = dead, excluded

        # Collect epochs from all EEG files for this patient
        mat_files = _find_eeg_files(patient_dir)
        if not mat_files:
            skipped_no_eeg += 1
            continue

        patient_epochs: List[np.ndarray] = []
        for mat_path in mat_files:
            epochs = _preprocess_recording(
                mat_path,
                target_sfreq=target_sfreq,
                epoch_sec=epoch_sec,
                amp_thresh=amp_thresh,
            )
            if epochs is not None and epochs.shape[0] > 0:
                patient_epochs.append(epochs)

        if not patient_epochs:
            skipped_no_eeg += 1
            continue

        patient_arr = np.concatenate(patient_epochs, axis=0)   # (N_p, C, T)
        n_p         = patient_arr.shape[0]

        all_epochs.append(patient_arr)
        all_labels.append(np.full(n_p, doc_label, dtype=np.int64))
        all_subj_ids.append(np.full(n_p, subj_idx, dtype=np.int64))
        subj_idx += 1
        loaded   += 1

        if verbose and loaded % 10 == 0:
            n_so_far = sum(e.shape[0] for e in all_epochs)
            log.info(
                f"  Loaded {loaded} patients ({n_so_far:,} epochs)  "
                f"skipped: CPC5={skipped_cpc5}, no_eeg={skipped_no_eeg}"
            )

    if not all_epochs:
        raise RuntimeError(
            f"No usable I-CARE patients loaded from {icare_root}.  "
            f"Check that metadata .txt files and EEG .mat files are present."
        )

    epochs_arr  = np.concatenate(all_epochs,   axis=0).astype(np.float32)
    labels_arr  = np.concatenate(all_labels,   axis=0).astype(np.int64)
    subj_arr    = np.concatenate(all_subj_ids, axis=0).astype(np.int64)

    log.info(f"I-CARE corpus: {loaded} patients, {len(epochs_arr):,} epochs total")
    log.info(f"  Skipped CPC5 (dead): {skipped_cpc5} | no usable EEG: {skipped_no_eeg}")
    for cls, name in [(0, "UWS / CPC4"), (1, "MCS / CPC3"), (2, "HC / CPC1-2")]:
        n = int((labels_arr == cls).sum())
        log.info(f"  Class {cls} ({name}): {n:,} epochs  ({100*n/len(labels_arr):.1f}%)")

    return epochs_arr, labels_arr, subj_arr


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class _SubjectSplitDataset(Dataset):
    def __init__(self, epochs: np.ndarray, labels: np.ndarray) -> None:
        self.epochs = torch.from_numpy(epochs)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.epochs[idx], self.labels[idx]


class ICareDataset(Dataset):
    """
    I-CARE EEG dataset with LOSO-CV subject splits.

    Attributes
    ----------
    epochs      : (N, N_ICARE_CHANNELS, T)  float32 tensor
    labels      : (N,) int64  — DOC class {0=UWS, 1=MCS, 2=HC}
    subject_ids : (N,) int64  — patient index (for LOSO)
    n_channels  : int  — always N_ICARE_CHANNELS (19)
    """

    def __init__(
        self,
        root:         Path  = ICARE_DIR,
        target_sfreq: float = SFREQ_ICARE,
        epoch_sec:    float = EPOCH_SEC,
    ) -> None:
        epochs, labels, subj_ids = build_icare_epochs(
            icare_root   = root,
            target_sfreq = target_sfreq,
            epoch_sec    = epoch_sec,
        )
        self.epochs      = torch.from_numpy(epochs)
        self.labels      = torch.from_numpy(labels)
        self.subject_ids = torch.from_numpy(subj_ids)
        self.n_channels  = N_ICARE_CHANNELS

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.epochs[idx], self.labels[idx]

    @property
    def n_subjects(self) -> int:
        return int(self.subject_ids.max().item()) + 1

    def class_counts(self) -> Dict[int, int]:
        return {cls: int((self.labels == cls).sum().item())
                for cls in range(NUM_DOC_CLASSES)}

    def get_subject_split(
        self,
        test_subject: int,
    ) -> Tuple[Dataset, Dataset]:
        """LOSO split — hold out one patient, train on rest."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def get_icare_loader(
    root:        Path = ICARE_DIR,
    batch_size:  int  = STAGE3_BATCH_SIZE,
    num_workers: int  = 0,
) -> Tuple[DataLoader, int]:
    """
    Build a DataLoader for the full I-CARE dataset (no subject split).
    Returns (loader, n_channels).
    """
    ds     = ICareDataset(root=root)
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
        f"I-CARE DataLoader: {len(ds):,} epochs, "
        f"{ds.n_subjects} patients, {ds.n_channels} channels  "
        f"class dist: {ds.class_counts()}"
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

    log.info("I-CARE smoke test …")
    try:
        ds = ICareDataset()
        log.info(f"  Total epochs  : {len(ds):,}")
        log.info(f"  Channels      : {ds.n_channels}")
        log.info(f"  Subjects      : {ds.n_subjects}")
        log.info(f"  Class counts  : {ds.class_counts()}")

        x, y = ds[0]
        log.info(f"  Epoch shape   : {x.shape}  (C={ds.n_channels}, T)")
        log.info(f"  Label         : {y.item()}  (0=UWS, 1=MCS, 2=HC)")
        log.info(f"  Value range   : [{x.min():.2f}, {x.max():.2f}]")
        log.info("  Smoke test PASSED ✓")
    except FileNotFoundError as exc:
        log.error(str(exc))
