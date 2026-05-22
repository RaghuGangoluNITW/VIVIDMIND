"""
Stage 2b — Train LorentzianEncoder (E2) directly on I-CARE DOC data.

Trains a LorentzianEncoder(n_eeg_ch=19, num_classes=3) on I-CARE using:
  • Hyperbolic geometry (Lorentz model H^64) — natural DOC hierarchy
  • WeightedRandomSampler  — fixes 10/14/76% class imbalance
  • FocalLoss (gamma=2)    — focuses on hard minority-class examples
  • Patient-stratified 5-fold CV

Note: This trains from scratch (not fine-tuning DEAP weights — channel
mismatch makes transfer impractical). The novel contribution is applying
Lorentzian geometry to I-CARE DOC classification directly.

Output:
  results/checkpoints/e2_doc_icare.pt
  results/checkpoints/e2_doc_icare.pkl

CLI:
  python -m src.stage2_emotion.finetune_e2_doc [options]

Options:
  --epochs  INT    Training epochs per fold (default 80)
  --lr      FLOAT  Learning rate (default 5e-4)
  --batch   INT    Batch size (default 64)
  --folds   INT    Patient CV folds (default 5)
  --device  STR    "cuda" or "cpu"
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ICARE_DIR,
    CKPT_ROOT,
    NUM_DOC_CLASSES,
    DOC_CLASSES,
    LORENTZ_DIM,
    RANDOM_SEED,
)
from src.models.lorentzian_encoder import LorentzianEncoder

log = logging.getLogger(__name__)

CKPT_PATH   = CKPT_ROOT / "e2_doc_icare.pt"
N_EEG_ICARE = 19   # I-CARE canonical channels


# ─────────────────────────────────────────────────────────────────────────────
# FocalLoss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p  = F.log_softmax(logits, dim=-1)
        p      = log_p.exp()
        p_true = p.gather(1, targets.view(-1, 1)).squeeze(1).clamp(min=1e-7)
        lp_t   = log_p.gather(1, targets.view(-1, 1)).squeeze(1)
        fw     = (1.0 - p_true) ** self.gamma
        if self.alpha is not None:
            fw = self.alpha[targets] * fw
        return (-fw * lp_t).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_loader(
    dataset: Subset,
    labels:  np.ndarray,
    batch_size: int,
) -> DataLoader:
    counts  = np.bincount(labels, minlength=NUM_DOC_CLASSES).astype(float)
    inv_f   = 1.0 / np.maximum(counts, 1.0)
    sw      = torch.tensor(inv_f[labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


@torch.no_grad()
def _eval(model, loader, device) -> Tuple[float, float]:
    model.eval()
    probs_l, labels_l = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits, _ = model(xb)
        probs_l.append(F.softmax(logits, dim=-1).cpu().numpy())
        labels_l.append(yb.numpy())
    probs  = np.concatenate(probs_l)
    labels = np.concatenate(labels_l)
    acc    = float((probs.argmax(1) == labels).mean())
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return acc, auc


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_e2_doc(args) -> None:
    # ── Load I-CARE ─────────────────────────────────────────────────────────
    from src.stage3_doc.dataset_icare import ICareDataset
    log.info("Loading I-CARE dataset …")
    icare_ds  = ICareDataset()
    epochs_t  = icare_ds.epochs        # (N, 19, T)
    labels_t  = icare_ds.labels        # (N,)
    subj_t    = icare_ds.subject_ids   # (N,)

    labels_np = labels_t.numpy()
    subj_np   = subj_t.numpy()
    n_subjects = int(subj_np.max()) + 1
    log.info(f"  {len(epochs_t):,} epochs, {n_subjects} patients")
    log.info(f"  Class counts: {np.bincount(labels_np).tolist()}")

    n_eeg_ch = epochs_t.shape[1]  # should be 19
    log.info(f"  EEG channels: {n_eeg_ch}")

    # ── Patient-level CV ─────────────────────────────────────────────────────
    patient_labels = np.array([
        int(np.bincount(labels_np[subj_np == s]).argmax())
        for s in range(n_subjects)
    ])
    subjects = np.arange(n_subjects)
    skf      = StratifiedKFold(n_splits=args.folds, shuffle=True,
                               random_state=RANDOM_SEED)
    full_ds  = TensorDataset(epochs_t, labels_t)

    fold_aucs:   List[float] = []
    best_val_auc = -1.0
    best_state:  Optional[dict] = None

    for fold_idx, (train_subs, val_subs) in enumerate(
        skf.split(subjects, patient_labels), start=1
    ):
        log.info(f"\n{'='*60}")
        log.info(f"Fold {fold_idx}/{args.folds}")

        train_mask = np.isin(subj_np, train_subs)
        val_mask   = np.isin(subj_np, val_subs)
        train_idx  = np.where(train_mask)[0]
        val_idx    = np.where(val_mask)[0]

        tr_labels = labels_np[train_mask]
        counts    = np.bincount(tr_labels, minlength=NUM_DOC_CLASSES).astype(float)
        inv_f     = 1.0 / np.maximum(counts, 1.0)
        alpha_w   = torch.tensor(inv_f / inv_f.sum(), dtype=torch.float32).to(args.device)

        train_loader = _weighted_loader(Subset(full_ds, train_idx), tr_labels, args.batch)
        val_loader   = DataLoader(Subset(full_ds, val_idx),
                                  batch_size=args.batch * 2, shuffle=False, num_workers=0)

        model = LorentzianEncoder(
            n_eeg_ch    = n_eeg_ch,
            lorentz_dim = LORENTZ_DIM,
            num_classes = NUM_DOC_CLASSES,
        ).to(args.device)

        criterion = FocalLoss(gamma=2.0, alpha=alpha_w)
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)

        best_fold_auc   = -1.0
        best_fold_state: Optional[dict] = None
        patience_cnt    = 0
        PATIENCE        = 15

        for ep in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(args.device), yb.to(args.device)
                optimiser.zero_grad()
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                total_loss += loss.item()
            scheduler.step()

            acc, auc = _eval(model, val_loader, args.device)
            auc_str  = f"{auc:.4f}" if not np.isnan(auc) else "—"
            if ep % 10 == 0 or ep == 1:
                log.info(f"  ep {ep:3d}/{args.epochs}  "
                         f"loss={total_loss/max(len(train_loader),1):.4f}  "
                         f"acc={acc:.4f}  auc={auc_str}")

            if not np.isnan(auc) and auc > best_fold_auc:
                best_fold_auc   = auc
                best_fold_state = {k: v.cpu().clone()
                                   for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    log.info(f"  Early stop at epoch {ep}")
                    break

        fold_aucs.append(best_fold_auc)
        log.info(f"  Fold {fold_idx} best AUC = {best_fold_auc:.4f}")

        if best_fold_state is not None and best_fold_auc > best_val_auc:
            best_val_auc = best_fold_auc
            best_state   = best_fold_state
            log.info(f"  ★ New overall best AUC = {best_val_auc:.4f}")

    log.info(f"\nCV  mean AUC = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    log.info(f"    best AUC = {best_val_auc:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    if best_state is not None:
        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": best_state,
            "config": {
                "n_eeg_ch":    n_eeg_ch,
                "lorentz_dim": LORENTZ_DIM,
                "num_classes": NUM_DOC_CLASSES,
                "stage":       "e2_doc_icare",
                "dataset":     "I-CARE",
            },
            "cv_aucs":  fold_aucs,
            "best_auc": best_val_auc,
        }
        torch.save(payload, CKPT_PATH)
        pkl_path = CKPT_PATH.with_suffix(".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"\nSaved → {CKPT_PATH}")
        log.info(f"PKL    → {pkl_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train E2 LorentzianEncoder on I-CARE DOC")
    p.add_argument("--epochs", type=int,   default=80)
    p.add_argument("--lr",     type=float, default=5e-4)
    p.add_argument("--batch",  type=int,   default=64)
    p.add_argument("--folds",  type=int,   default=5)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        level=logging.INFO)
    args = _parse_args()
    log.info(f"E2 DOC training (LorentzianEncoder on I-CARE) — device: {args.device}")
    train_e2_doc(args)


if __name__ == "__main__":
    main()
