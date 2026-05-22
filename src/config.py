"""
Central configuration for the DOC O4 Journal project.

All paths, hyperparameters, and experimental settings live here.
Modify ONLY this file to adapt the project to a new machine or dataset location.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. ROOT PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
CKPT_ROOT    = RESULTS_ROOT / "checkpoints"
PLOT_ROOT    = RESULTS_ROOT / "plots"
TABLE_ROOT   = RESULTS_ROOT / "tables"

# Create output dirs on import
for _d in [CKPT_ROOT, PLOT_ROOT, TABLE_ROOT]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET PATHS
# ─────────────────────────────────────────────────────────────────────────────
DEAP_DIR     = DATA_ROOT / "DEAP" / "DEAP" / "data_preprocessed_matlab"
DREAMER_MAT  = DATA_ROOT / "DREAMER" / "DREAMER" / "DREAMER.mat"
TUH_DIR      = DATA_ROOT / "tuh_eeg"         # v2.0.1/edf/ — 2905 recordings

# I-CARE (International Cardiac Arrest REsearch) — PhysioNet 2023
# 607 patients, 344 EEG files — fully downloaded to data/icare/training/
ICARE_DIR     = DATA_ROOT / "icare" / "training"

# ─────────────────────────────────────────────────────────────────────────────
# 3. EEG PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
SFREQ_DEAP    = 128    # Hz  — DEAP is already downsampled
SFREQ_DREAMER = 128    # Hz  — Emotiv EPOC native
SFREQ_ICARE   = 256    # Hz  — resample target for I-CARE (native 500 Hz → 256)
SFREQ_TUH     = 256    # Hz  — resample target for TUH (variable in raw)

BANDPASS_LOW  = 1.0    # Hz
BANDPASS_HIGH = 45.0   # Hz
NOTCH_FREQ    = 50.0   # Hz  (60.0 for US datasets)

EPOCH_SEC      = 4.0    # seconds per epoch
OVERLAP        = 0.5    # 50% overlap between epochs
EPOCH_OVERLAP  = OVERLAP  # alias used by data loaders
AMP_THRESH     = 150.0  # µV — amplitude-based artifact rejection
AMP_THRESHOLD  = AMP_THRESH  # alias used by data loaders

# DEAP specific: 32 EEG channels (first 32 of 40)
DEAP_EEG_CH         = 32
N_EEG_CHANNELS_DEAP = DEAP_EEG_CH   # alias
DEAP_TRIALS         = 40
DEAP_SRATE          = 128
DEAP_PRESEC         = 3      # pre-trial baseline seconds
DEAP_N_SUBJECTS     = 32
# Label column: 0=valence, 1=arousal, 2=dominance, 3=liking
DEAP_LABEL_COL = 0
DEAP_THRESHOLD = 5.0   # ≥5 → high valence (binary)

# DREAMER specific: 14 EEG channels (Emotiv EPOC)
DREAMER_EEG_CH          = 14
N_EEG_CHANNELS_DREAMER  = DREAMER_EEG_CH   # alias
DREAMER_TRIALS          = 18
DREAMER_N_SUBJECTS      = 23
# Label column: 0=valence, 1=arousal, 2=dominance
DREAMER_LABEL_COL  = 0
DREAMER_THRESHOLD  = 3.0   # ≥3 → high valence (binary, scale 1-5)

# I-CARE specific
ICARE_EEG_CH   = 19    # standard 10-20 montage (Fp1...Pz)
ICARE_TRIALS   = 40    # typical clean epochs per 2-hour download

# Chennu (legacy — kept for backward compat with graph_encoder.py smoke test)
CHENNU_EEG_CH  = ICARE_EEG_CH  # remapped to I-CARE equivalent
CHENNU_TRIALS  = ICARE_TRIALS

# ─────────────────────────────────────────────────────────────────────────────
# 4. LABEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
# DEAP/DREAMER: binarise valence and arousal at midpoint of scale
DEAP_VALENCE_THRESHOLD  = 5.0   # out of 9
DEAP_AROUSAL_THRESHOLD  = 5.0
DREAMER_VA_THRESHOLD    = 3.0   # out of 5

# DOC classes — I-CARE CPC mapping (CPC 1→HC, 2→HC, 3→MCS, 4→UWS, 5→excluded)
DOC_CLASSES    = {0: "UWS", 1: "MCS", 2: "HC"}
NUM_DOC_CLASSES = 3

# Emotion classes (binary per dimension)
EMOTION_CLASSES    = {0: "low", 1: "high"}
NUM_EMOTION_CLASSES = 2

# ─────────────────────────────────────────────────────────────────────────────
# 5. LORENTZIAN ENCODER (E2) HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
LORENTZ_DIM       = 64    # dimension of Lorentz manifold (n in H^n)
LORENTZ_CURV      = 1.0   # curvature c (fixed, not learned)
TEMPORAL_CHANNELS     = [32, 64, 128]   # CNN channel progression
TEMPORAL_CNN_CHANNELS = TEMPORAL_CHANNELS  # alias
TEMPORAL_KERNELS      = [25, 15, 7]     # kernel sizes (in samples)
TEMPORAL_CNN_KERNELS  = TEMPORAL_KERNELS   # alias
DROPOUT_RATE      = 0.3
ENCODER_HIDDEN    = 256   # FC hidden size before manifold projection

# ─────────────────────────────────────────────────────────────────────────────
# 6. FRACTALSSL ENCODER (E1) HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
FRACTAL_DIM_RANGE = (1.0, 2.0)   # Hurst exponent range for augmentation
FRACTAL_PROJ_DIM  = 128           # projection head output dim
SSL_TEMPERATURE   = 0.07
SSL_NEG_SAMPLES   = 256

# ─────────────────────────────────────────────────────────────────────────────
# 7. GRAPH ENCODER (E3) HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
GNN_HIDDEN_DIM  = 64
GNN_LAYERS      = 3
GNN_HEADS       = 4     # GAT attention heads
GNN_DROPOUT     = 0.2
FREQ_BANDS      = ["delta", "theta", "alpha", "beta"]  # dwPLI bands

# ─────────────────────────────────────────────────────────────────────────────
# 8. PDI-CCS FUSION HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
PDI_ALPHA = 0.6   # weight of mean prediction vs disagreement in CCS
PDI_BETA  = 0.4   # weight of (1 - PDI) in CCS
CCS_COVERT_THRESHOLD = 0.65  # CCS above this in VS → flag covert awareness

# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
STAGE2_LR          = 1e-4
STAGE2_EPOCHS      = 100
STAGE2_BATCH_SIZE  = 64
STAGE2_WEIGHT_DECAY = 1e-4
STAGE2_PATIENCE    = 15      # early stopping patience

STAGE3_LR          = 5e-5
STAGE3_EPOCHS      = 80
STAGE3_BATCH_SIZE  = 32
STAGE3_PATIENCE    = 20

# Label noise robustness — Generalised Cross-Entropy loss
GCE_Q = 0.7   # q in (0,1]: closer to 0 = more noise robust, 1 = standard CE

# ─────────────────────────────────────────────────────────────────────────────
# 10. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
CV_FOLDS    = 5
LOSO        = True    # Leave-One-Subject-Out for DOC evaluation
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# 11. DEVICE
# ─────────────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# 12. SCIENTIFIC REPORTS — BINARY NEUROLOGICAL OUTCOME TASK
# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY task: good (CPC1-2) vs poor (CPC3-4) outcome prediction
# Maps to 3-class DOC label space: HC (class 2) = good; UWS+MCS (classes 0,1) = poor
BINARY_GOOD_DOC_CLASS   = 2        # HC → CPC 1-2 (good neurological outcome)
BINARY_POOR_DOC_CLASSES = [0, 1]   # UWS, MCS → CPC 3-4 (poor outcome)
PATIENT_LEVEL_EVAL      = True     # ALWAYS aggregate epochs → patient before reporting metrics
BOOTSTRAP_N_RESAMPLES   = 1000     # Bootstrap resamples for 95% CI on AUC

# Usable I-CARE patients (actual downloaded + processed cohort)
# 55 patients have complete EEG recordings with CPC outcome labels
# Class breakdown: HC (CPC1-2) = 39 | poor (CPC3-4) = 16
ICARE_USABLE_PATIENTS   = 55
ICARE_HC_PATIENTS       = 39   # good outcome (CPC1-2)
ICARE_POOR_PATIENTS     = 16   # poor outcome (CPC3-4)

# 2025-2026 SOTA baselines (patient-level binary AUC for fair comparison)
# Raveendran 2025: window-level evaluation — NOT directly comparable
SOTA_LIU2025_BINARY_AUC       = 0.78   # Liu et al. 2025 (IEEE CCC, n=28)
SOTA_DELLAB2025_BINARY_AUC    = 0.81   # Della Bella et al. 2025 (Commun. Biol., n=237)
SOTA_TARGET_BINARY_AUC        = 0.85   # minimum target to exceed SOTA on primary task
