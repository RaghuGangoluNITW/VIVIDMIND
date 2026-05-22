"""
Stage 2 — Emotion Encoder Training

Trains the LorentzianEncoder (E2) on DEAP + DREAMER using:
  • Generalised Cross-Entropy (GCE) loss  — robust to DEAP/CRS-R label noise
  • AdamW optimiser with cosine LR schedule
  • Leave-One-Subject-Out cross-validation (LOSO)
  • Best checkpoint saved to results/checkpoints/e2_lorentzian_best.pt

Usage:
    python -m src.stage2_emotion.train_emotion_encoder \
        --dataset deap          # deap | dreamer | both
        --label   valence       # valence | arousal | dominance
        --epochs  100
        --device  cuda
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Add project root to path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEVICE,
    STAGE2_LR,
    STAGE2_EPOCHS,
    STAGE2_BATCH_SIZE,
    STAGE2_WEIGHT_DECAY,
    STAGE2_PATIENCE,
    CKPT_ROOT,
    GCE_Q,
    RANDOM_SEED,
    DEAP_N_SUBJECTS,
    DREAMER_N_SUBJECTS,
    N_EEG_CHANNELS_DEAP,
    N_EEG_CHANNELS_DREAMER,
    LORENTZ_DIM,
    TEMPORAL_CNN_CHANNELS,
    TEMPORAL_CNN_KERNELS,
    DEAP_LABEL_COL,
    DREAMER_LABEL_COL,
)
from src.models.lorentzian_encoder import LorentzianEncoder, EuclideanEncoder
from src.stage2_emotion.dataset_deap import DEAPDataset, _SubsetDataset
from src.stage2_emotion.dataset_dreamer import DREAMERDataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Generalised Cross-Entropy Loss
# ─────────────────────────────────────────────────────────────────────────────

class GeneralisedCrossEntropyLoss(nn.Module):
    """
    GCE loss (Zhang & Sabuncu, NeurIPS 2018).

    L_GCE(p_y, y) = (1 - p_y^q) / q

    q in (0, 1]:
        q → 0   : Mean Absolute Error (maximally noise-robust)
        q = 1   : Standard Cross-Entropy
        q = 0.7 : good empirical trade-off for EEG label noise

    This is critical for DOC: CRS-R has 20–40% misclassification rate.
    """

    def __init__(self, q: float = GCE_Q, num_classes: int = 2) -> None:
        super().__init__()
        assert 0 < q <= 1, "q must be in (0, 1]"
        self.q           = q
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, num_classes)
        targets : (B,)  int
        """
        probs = torch.softmax(logits, dim=1)                            # (B, C)
        p_y   = probs[torch.arange(len(targets)), targets]             # (B,)
        loss  = (1.0 - p_y ** self.q) / self.q
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float, float]:
    """Returns (loss, accuracy, f1, auc)."""
    model.eval()
    all_logits, all_labels = [], []
    running_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            running_loss += criterion(logits, y).item()
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0).numpy()
    probs      = torch.softmax(logits_cat, dim=1)[:, 1].numpy()
    preds      = logits_cat.argmax(dim=1).numpy()

    loss = running_loss / len(loader)
    acc  = accuracy_score(labels_cat, preds)
    f1   = f1_score(labels_cat, preds, zero_division=0)

    unique_cls = np.unique(labels_cat)
    auc  = roc_auc_score(labels_cat, probs) if len(unique_cls) > 1 else 0.5

    return loss, acc, f1, auc


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (one fold)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(
    train_ds:   Dataset,
    test_ds:    Dataset,
    n_channels: int,
    fold_id:    int,
    device:     str,
    epochs:     int   = STAGE2_EPOCHS,
    batch_size: int   = STAGE2_BATCH_SIZE,
    model_cls         = None,
) -> dict:
    """
    Train on train_ds, evaluate on test_ds.
    Returns dict with best metrics for this fold.

    model_cls : LorentzianEncoder (default) or EuclideanEncoder.
    Both share the same TemporalCNN + SpatialAttention backbone and the
    same FC projection; they differ only in the manifold layer:
      - LorentzianEncoder  : projects onto H^n (Lorentz hyperboloid)
      - EuclideanEncoder   : L2-normalises onto the unit hypersphere
    This parametric design ensures the training loop is identical for both
    and the only variable is the geometry — making the ablation fair.
    """
    if model_cls is None:
        model_cls = LorentzianEncoder

    torch.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = device == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = 0,
    )

    model = model_cls(
        n_eeg_ch     = n_channels,
        cnn_channels  = TEMPORAL_CNN_CHANNELS,
        kernels       = TEMPORAL_CNN_KERNELS,
        lorentz_dim   = LORENTZ_DIM,
        num_classes   = 2,
    ).to(device)

    criterion = GeneralisedCrossEntropyLoss(q=GCE_Q)
    optimiser = AdamW(
        model.parameters(),
        lr           = STAGE2_LR,
        weight_decay = STAGE2_WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)

    best_auc   = -1.0   # allows AUC=0.0 to be saved on first eval
    patience   = 0
    # Initialise with the untrained model so best_state is never None
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for ep in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item()
        scheduler.step()

        # ── eval ──
        val_loss, val_acc, val_f1, val_auc = evaluate(
            model, test_loader, criterion, device
        )

        if ep % 10 == 0 or ep == 1:
            log.info(
                f"Fold {fold_id:02d}  Ep {ep:03d}/{epochs}  "
                f"train_loss={train_loss/len(train_loader):.4f}  "
                f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  val_auc={val_auc:.3f}"
            )

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1

        if patience >= STAGE2_PATIENCE:
            log.info(f"Fold {fold_id:02d}  Early stop at epoch {ep}")
            break

    return {
        "fold":     fold_id,
        "best_auc": best_auc,
        "state":    best_state,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOSO-CV for DEAP
# ─────────────────────────────────────────────────────────────────────────────

def run_loso_deap(args, model_cls=None, ckpt_suffix: str = "lorentzian") -> None:
    log.info(f"=== Stage 2 LOSO-CV on DEAP  [{ckpt_suffix} geometry] ===")
    log.info("Loading all DEAP subjects …")
    full_ds = DEAPDataset()
    log.info(f"  Total epochs: {len(full_ds)}")

    aucs: List[float] = []
    f1s:  List[float] = []
    accs: List[float] = []
    fold_metrics: List[Dict] = []
    best_overall_auc = -1.0          # allow any AUC (even 0.5) to be saved
    best_model_state = None

    for sid in range(1, DEAP_N_SUBJECTS + 1):
        train_ds, test_ds = full_ds.get_subject_split(sid)
        if len(test_ds) == 0:
            log.warning(f"  Subject {sid:02d}: no test data, skipping")
            continue

        result = train_one_fold(
            train_ds,
            test_ds,
            n_channels = N_EEG_CHANNELS_DEAP,
            fold_id    = sid,
            device     = args.device,
            epochs     = args.epochs,
            model_cls  = model_cls,
        )

        # Final eval with best weights
        _mcls = model_cls if model_cls is not None else LorentzianEncoder
        model = _mcls(
            n_eeg_ch=N_EEG_CHANNELS_DEAP, num_classes=2
        ).to(args.device)
        model.load_state_dict(result["state"])
        loader = DataLoader(test_ds, batch_size=STAGE2_BATCH_SIZE * 2)
        crit   = GeneralisedCrossEntropyLoss()
        _, acc, f1, auc = evaluate(model, loader, crit, args.device)

        aucs.append(auc)
        f1s.append(f1)
        accs.append(acc)
        fold_metrics.append({"subject_id": sid, "auc": auc, "f1": f1, "acc": acc})
        log.info(f"  ▸ Subject {sid:02d}  AUC={auc:.3f}  F1={f1:.3f}  ACC={acc:.3f}")

        if auc > best_overall_auc:
            best_overall_auc = auc
            best_model_state = result["state"]

    log.info("─── DEAP LOSO-CV Results ───")
    if not aucs:
        log.warning("  No valid folds — all subjects skipped.")
        return
    log.info(f"  AUC  : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    log.info(f"  F1   : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    log.info(f"  ACC  : {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    ckpt_path = CKPT_ROOT / f"e2_{ckpt_suffix}_deap_best.pt"
    if best_model_state is not None:
        # ── .pt checkpoint ────────────────────────────────────────────────────
        ckpt_payload = {
            "model_state_dict": best_model_state,
            "config": {
                "n_eeg_ch":    N_EEG_CHANNELS_DEAP,
                "lorentz_dim": LORENTZ_DIM,
                "num_classes": 2,
                "dataset":     "DEAP",
                "stage":       f"stage2_{ckpt_suffix}_emotion",
                "geometry":    ckpt_suffix,
            },
            "loso_metrics": {
                "fold_details":  fold_metrics,
                "mean_auc":      float(np.mean(aucs)),
                "std_auc":       float(np.std(aucs)),
                "mean_f1":       float(np.mean(f1s)),
                "std_f1":        float(np.std(f1s)),
                "mean_acc":      float(np.mean(accs)),
                "std_acc":       float(np.std(accs)),
                "best_fold_auc": float(best_overall_auc),
            },
        }
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt_payload, ckpt_path)
        log.info(f"  Best model saved → {ckpt_path}")
        # ── .pkl for Code Ocean (same payload, pickle-serialised) ──────────────
        pkl_path = ckpt_path.with_suffix(".pkl")
        with open(pkl_path, "wb") as _f:
            pickle.dump(ckpt_payload, _f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"  PKL copy saved  → {pkl_path}")
    else:
        log.warning("  No best model state to save (all folds may have been single-class).")


# ─────────────────────────────────────────────────────────────────────────────
# LOSO-CV for DREAMER
# ─────────────────────────────────────────────────────────────────────────────

def run_loso_dreamer(args, model_cls=None, ckpt_suffix: str = "lorentzian") -> None:
    log.info(f"=== Stage 2 LOSO-CV on DREAMER  [{ckpt_suffix} geometry] ===")
    log.info("Loading all DREAMER subjects …")
    full_ds = DREAMERDataset()
    log.info(f"  Total epochs: {len(full_ds)}")

    aucs: List[float] = []
    f1s:  List[float] = []
    accs: List[float] = []
    fold_metrics: List[Dict] = []
    best_overall_auc = -1.0          # allow any AUC (even 0.5) to be saved
    best_model_state = None

    for sid in range(DREAMER_N_SUBJECTS):
        train_ds, test_ds = full_ds.get_subject_split(sid)
        if len(test_ds) == 0:
            continue

        result = train_one_fold(
            train_ds,
            test_ds,
            n_channels = N_EEG_CHANNELS_DREAMER,
            fold_id    = sid,
            device     = args.device,
            epochs     = args.epochs,
            model_cls  = model_cls,
        )

        _mcls = model_cls if model_cls is not None else LorentzianEncoder
        model = _mcls(
            n_eeg_ch=N_EEG_CHANNELS_DREAMER, num_classes=2
        ).to(args.device)
        model.load_state_dict(result["state"])
        loader = DataLoader(test_ds, batch_size=STAGE2_BATCH_SIZE * 2)
        crit   = GeneralisedCrossEntropyLoss()
        _, acc, f1, auc = evaluate(model, loader, crit, args.device)

        aucs.append(auc)
        f1s.append(f1)
        accs.append(acc)
        fold_metrics.append({"subject_id": sid, "auc": auc, "f1": f1, "acc": acc})
        log.info(f"  ▸ Subject {sid:02d}  AUC={auc:.3f}  F1={f1:.3f}  ACC={acc:.3f}")

        if auc > best_overall_auc:
            best_overall_auc = auc
            best_model_state = result["state"]

    log.info("─── DREAMER LOSO-CV Results ───")
    if not aucs:
        log.warning("  No valid folds — all subjects skipped.")
        return
    log.info(f"  AUC  : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    log.info(f"  F1   : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    log.info(f"  ACC  : {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    ckpt_path = CKPT_ROOT / f"e2_{ckpt_suffix}_dreamer_best.pt"
    if best_model_state is not None:
        # ── .pt checkpoint ────────────────────────────────────────────────────
        ckpt_payload = {
            "model_state_dict": best_model_state,
            "config": {
                "n_eeg_ch":    N_EEG_CHANNELS_DREAMER,
                "lorentz_dim": LORENTZ_DIM,
                "num_classes": 2,
                "dataset":     "DREAMER",
                "stage":       f"stage2_{ckpt_suffix}_emotion",
                "geometry":    ckpt_suffix,
            },
            "loso_metrics": {
                "fold_details":  fold_metrics,
                "mean_auc":      float(np.mean(aucs)),
                "std_auc":       float(np.std(aucs)),
                "mean_f1":       float(np.mean(f1s)),
                "std_f1":        float(np.std(f1s)),
                "mean_acc":      float(np.mean(accs)),
                "std_acc":       float(np.std(accs)),
                "best_fold_auc": float(best_overall_auc),
            },
        }
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt_payload, ckpt_path)
        log.info(f"  Best model saved → {ckpt_path}")
        # ── .pkl for Code Ocean ────────────────────────────────────────────────
        pkl_path = ckpt_path.with_suffix(".pkl")
        with open(pkl_path, "wb") as _f:
            pickle.dump(ckpt_payload, _f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"  PKL copy saved  → {pkl_path}")
    else:
        log.warning("  No best model state to save (all folds may have been single-class).")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train E2 Encoder on DEAP / DREAMER  (Lorentzian or Euclidean geometry)"
    )
    p.add_argument(
        "--dataset",
        choices=["deap", "dreamer", "both"],
        default="both",
        help="Which dataset to use for LOSO-CV",
    )
    p.add_argument(
        "--geometry",
        choices=["lorentzian", "euclidean"],
        default="lorentzian",
        help=(
            "Manifold geometry for E2. "
            "'lorentzian': embeds on the Lorentz hyperboloid H^n (proposed). "
            "'euclidean': L2-normalises onto the unit hypersphere (ablation baseline). "
            "Saves to e2_<geometry>_<dataset>_best.pt so both can coexist."
        ),
    )
    p.add_argument("--epochs", type=int, default=STAGE2_EPOCHS)
    p.add_argument("--device", default=DEVICE)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log.info(f"Device: {args.device}")
    log.info(f"Geometry: {args.geometry}")

    model_cls   = EuclideanEncoder if args.geometry == "euclidean" else LorentzianEncoder
    ckpt_suffix = args.geometry    # 'lorentzian' or 'euclidean'

    if args.dataset in ("deap", "both"):
        run_loso_deap(args, model_cls=model_cls, ckpt_suffix=ckpt_suffix)

    if args.dataset in ("dreamer", "both"):
        run_loso_dreamer(args, model_cls=model_cls, ckpt_suffix=ckpt_suffix)
