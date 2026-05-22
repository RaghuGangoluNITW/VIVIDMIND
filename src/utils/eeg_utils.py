"""
EEG preprocessing utilities shared across all stages.

Handles: bandpass filtering, epoching, artifact rejection,
         normalisation, and common electrode operations.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    low: float = 1.0,
    high: float = 45.0,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.

    data  : (..., n_samples)  — last axis is time
    returns same shape
    """
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch_filter(
    data: np.ndarray,
    sfreq: float,
    freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """
    IIR notch filter to remove power-line noise.

    data  : (..., n_samples)
    """
    b, a = iirnotch(freq, quality, fs=sfreq)
    from scipy.signal import lfilter
    return lfilter(b, a, data, axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Epoching
# ─────────────────────────────────────────────────────────────────────────────

def epoch_signal(
    data: np.ndarray,
    sfreq: float,
    epoch_sec: float = 4.0,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Slice a continuous EEG signal into overlapping fixed-length epochs.

    data      : (n_channels, n_samples)
    sfreq     : sampling frequency in Hz
    epoch_sec : length of each epoch in seconds
    overlap   : fraction of overlap between consecutive epochs (0 to 1)

    returns   : (n_epochs, n_channels, epoch_samples)
    """
    n_samples   = data.shape[-1]
    epoch_len   = int(epoch_sec * sfreq)
    step        = int(epoch_len * (1.0 - overlap))

    starts = list(range(0, n_samples - epoch_len + 1, step))
    if not starts:
        return np.empty((0, data.shape[0], epoch_len), dtype=data.dtype)
    epochs = np.stack([data[..., s: s + epoch_len] for s in starts], axis=0)
    return epochs


# ─────────────────────────────────────────────────────────────────────────────
# Artifact rejection
# ─────────────────────────────────────────────────────────────────────────────

def reject_artifacts(
    epochs: np.ndarray,
    amp_thresh: float = 150.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove epochs where any channel exceeds the amplitude threshold (peak-to-peak).

    epochs     : (n_epochs, n_channels, n_samples)
    amp_thresh : µV threshold

    returns    : (clean_epochs, kept_mask)
    """
    ptp = epochs.max(axis=-1) - epochs.min(axis=-1)   # (n_epochs, n_channels)
    mask = (ptp < amp_thresh).all(axis=-1)             # (n_epochs,)
    return epochs[mask], mask


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def zscore_epochs(epochs: np.ndarray) -> np.ndarray:
    """
    Z-score normalise each channel across time within each epoch.

    epochs : (n_epochs, n_channels, n_samples)
    """
    mu    = epochs.mean(axis=-1, keepdims=True)
    sigma = epochs.std(axis=-1, keepdims=True) + 1e-8
    return (epochs - mu) / sigma


def global_channel_normalize(
    epochs: np.ndarray,
) -> np.ndarray:
    """
    Normalise across all epochs per channel (global statistics).
    Useful when training across subjects.

    epochs : (n_epochs, n_channels, n_samples)
    """
    mu    = epochs.mean(axis=(0, 2), keepdims=True)
    sigma = epochs.std(axis=(0, 2), keepdims=True) + 1e-8
    return (epochs - mu) / sigma


# ─────────────────────────────────────────────────────────────────────────────
# Band power features (for GNN edge construction)
# ─────────────────────────────────────────────────────────────────────────────

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}


def compute_band_power(
    epoch: np.ndarray,
    sfreq: float,
    band: str = "alpha",
) -> np.ndarray:
    """
    Compute average power in a frequency band per channel via FFT.

    epoch  : (n_channels, n_samples)
    returns: (n_channels,)
    """
    low, high = BANDS[band]
    n         = epoch.shape[-1]
    freqs     = np.fft.rfftfreq(n, d=1.0 / sfreq)
    fft_vals  = np.abs(np.fft.rfft(epoch, axis=-1)) ** 2
    idx       = (freqs >= low) & (freqs <= high)
    return fft_vals[:, idx].mean(axis=-1)


def compute_dwpli_matrix(
    epoch: np.ndarray,
    sfreq: float,
    band: str = "alpha",
) -> np.ndarray:
    """
    Compute debiased Weighted Phase Lag Index (dwPLI) connectivity matrix.

    epoch  : (n_channels, n_samples)
    returns: (n_channels, n_channels)  — symmetric dwPLI matrix

    Reference: Vinck et al., NeuroImage 2011.
    """
    low, high = BANDS[band]
    n_ch, n_t = epoch.shape
    n_seg     = 16
    seg_len   = n_t // n_seg

    cross_spec_imag_sum  = np.zeros((n_ch, n_ch))
    cross_spec_imag_abs  = np.zeros((n_ch, n_ch))
    cross_spec_imag_sq   = np.zeros((n_ch, n_ch))

    freqs = np.fft.rfftfreq(seg_len, d=1.0 / sfreq)
    idx   = (freqs >= low) & (freqs <= high)

    for s in range(n_seg):
        seg  = epoch[:, s * seg_len: (s + 1) * seg_len]
        fft  = np.fft.rfft(seg, axis=-1)[:, idx]          # (n_ch, n_freq)
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                cs    = fft[i] * np.conj(fft[j])
                imag  = np.imag(cs)
                cross_spec_imag_sum[i, j]  += imag.sum()
                cross_spec_imag_abs[i, j]  += np.abs(imag).sum()
                cross_spec_imag_sq[i, j]   += (imag ** 2).sum()

    # dwPLI formula
    dwpli = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            num  = cross_spec_imag_sum[i, j] ** 2 - cross_spec_imag_sq[i, j]
            den  = cross_spec_imag_abs[i, j] ** 2 - cross_spec_imag_sq[i, j]
            val  = num / (den + 1e-10)
            dwpli[i, j] = val
            dwpli[j, i] = val

    return np.clip(dwpli, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Full preprocessing pipeline for a single recording
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_recording(
    raw: np.ndarray,
    sfreq: float,
    bandpass: Tuple[float, float] = (1.0, 45.0),
    notch: Optional[float] = 50.0,
    epoch_sec: float = 4.0,
    overlap: float = 0.5,
    amp_thresh: float = 150.0,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing for a single (n_channels, n_samples) EEG recording.

    Returns: (clean_epochs, kept_mask)
        clean_epochs : (n_epochs, n_channels, epoch_samples)
        kept_mask    : (n_raw_epochs,) bool
    """
    # 1. Filter
    data = bandpass_filter(raw, sfreq, low=bandpass[0], high=bandpass[1])
    if notch is not None:
        data = notch_filter(data, sfreq, freq=notch)

    # 2. Epoch
    epochs = epoch_signal(data, sfreq, epoch_sec=epoch_sec, overlap=overlap)

    # 3. Artifact rejection
    epochs, mask = reject_artifacts(epochs, amp_thresh=amp_thresh)

    # 4. Normalise
    if normalize and len(epochs) > 0:
        epochs = zscore_epochs(epochs)

    return epochs, mask
