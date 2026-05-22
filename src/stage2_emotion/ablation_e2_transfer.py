"""
Ablation: DEAP pre-training benefit for E2 Lorentzian encoder.

Compares TWO fine-tuning strategies on I-CARE (5-fold stratified CV):
  A) Random initialisation   — already stored in e2_doc_icare.pt
  B) DEAP pre-training       — transfer TemporalCNN + SpatialAttention +
                               proj + lorentz_lin from e2_lorentzian_deap_best.pt;
                               replace the 2-class LorentzMLR head with a
                               fresh 3-class head before fine-tuning.

This answers the reviewer question:
  "Does cross-domain DEAP pre-training actually help over training from scratch?"

The TemporalCNN processes each EEG channel independently as a 1-D signal,
so its weights are fully channel-count-agnostic and transfer directly from
32-channel DEAP to 19-channel I-CARE.  Only the LorentzMLR head is replaced
(2-class → 3-class).

Outputs
-------
  results/checkpoints/e2_doc_icare_from_deap.pt  — best checkpoint
  results/tables/ablation_e2_transfer.csv         — fold-level AUC comparison
  Logs the final comparison to stdout.

CLI
---
  python -m src.stage2_emotion.ablation_e2_transfer [options]

Options
-------
  --epochs  INT    Epochs per fold  (default 80)
  --lr      FLOAT  Learning rate    (default 5e-4)
  --batch   INT    Batch size       (default 64)
  --folds   INT    CV folds         (default 5)
  --device  STR    "cuda" or "cpu"
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CKPT_ROOT,
    TABLE_ROOT,
    NUM_DOC_CLASSES,
    LORENTZ_DIM,
    RANDOM_SEED,
)
from src.models.lorentzian_encoder import LorentzianEncoder

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

DEAP_CKPT  = CKPT_ROOT / "e2_lorentzian_deap_best.pt"
ICARE_CKPT = CKPT_ROOT / "e2_doc_icare.pt"
OUT_CKPT   = CKPT_ROOT / "e2_doc_icare_from_deap.pt"
OUT_CSV    = TABLE_ROOT / "ablation_e2_transfer.csv"

N_EEG_ICARE = 19


# ─────────────────────────────────────────────────────────────────────────────
# Weight transfer
# ─────────────────────────────────────────────────────────────────────────────

def _load_pretrained_backbone(deap_ckpt: Path, device: str) -> dict:
    """
    Load DEAP checkpoint and extract all transferable keys.

    Transferable layers (channel-count-independent because TemporalCNN
    processes each channel as a separate 1-D signal):
        temporal_cnn.*   — per-channel 1-D CNN (fully transferable)
        spatial_attn.*   — attention MLP over channel features (transferable)
        proj.*           — FC + LayerNorm projection to lorentz_dim
        lorentz_lin.*    — LorentzLinear manifold layer

    Non-transferable:
        classifier.*     — 2-class LorentzMLR head → replaced by a fresh 3-class head
    """
    ckpt  = torch.load(deap_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    backbone_state = {
        k: v for k, v in state.items()
        if not k.startswith("classifier.")
    }
    log.info(
        f"  Extracted {len(backbone_state)}/{len(state)} keys from DEAP checkpoint "
        f"(dropped classifier.*)"
    )
    return backbone_state


def build_icare_model_from_deap(deap_ckpt: Path, device: str) -> LorentzianEncoder:
    """
    Construct a 19-channel, 3-class LorentzianEncoder pre-loaded with
    the DEAP TemporalCNN/SpatialAttention/proj/lorentz_lin weights.
    The LorentzMLR head is freshly initialised.
    """
    model = LorentzianEncoder(
        n_eeg_ch    = N_EEG_ICARE,
        lorentz_dim = LORENTZ_DIM,
        num_classes = NUM_DOC_CLASSES,
    )
    backbone_state = _load_pretrained_backbone(deap_ckpt, device)
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    log.info(f"  Missing keys (expected — new classifier head): {missing}")
    if unexpected:
        log.warning(f"  Unexpected keys: {unexpected}")
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers  (copied from finetune_e2_doc.py for self-containment)
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


def _weighted_loader(dataset: Subset, labels: np.ndarray, batch_size: int) -> DataLoader:
    counts  = np.bincount(labels, minlength=NUM_DOC_CLASSES).astype(float)
    inv_f   = 1.0 / np.maximum(counts, 1.0)
    sw      = torch.tensor(inv_f[labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


@torch.no_grad()
def _eval(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    probs_l, labels_l = [], []
    for xb, yb in loader:
        logits, _ = model(xb.to(device))
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


def _run_cv(
    epochs_t: torch.Tensor,
    labels_t: torch.Tensor,
    subj_np:  np.ndarray,
    args,
    init_mode: str,           # "random" | "deap"
) -> Tuple[List[float], float, Optional[dict]]:
    """5-fold patient-stratified CV. Returns (fold_aucs, best_auc, best_state)."""
    labels_np  = labels_t.numpy()
    n_subjects = int(subj_np.max()) + 1
    patient_labels = np.array([
        int(np.bincount(labels_np[subj_np == s]).argmax())
        for s in range(n_subjects)
    ])
    subjects = np.arange(n_subjects)
    skf      = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_SEED)
    full_ds  = TensorDataset(epochs_t, labels_t)

    fold_aucs:   List[float] = []
    best_auc     = -1.0
    best_state:  Optional[dict] = None
    PATIENCE = 15

    for fold_idx, (train_subs, val_subs) in enumerate(
        skf.split(subjects, patient_labels), start=1
    ):
        log.info(f"\n[{init_mode}] Fold {fold_idx}/{args.folds}")

        train_mask = np.isin(subj_np, train_subs)
        val_mask   = np.isin(subj_np, val_subs)
        tr_labels  = labels_np[train_mask]
        counts     = np.bincount(tr_labels, minlength=NUM_DOC_CLASSES).astype(float)
        inv_f      = 1.0 / np.maximum(counts, 1.0)
        alpha_w    = torch.tensor(inv_f / inv_f.sum(), dtype=torch.float32).to(args.device)

        train_loader = _weighted_loader(
            Subset(full_ds, np.where(train_mask)[0]), tr_labels, args.batch)
        val_loader   = DataLoader(
            Subset(full_ds, np.where(val_mask)[0]),
            batch_size=args.batch * 2, shuffle=False, num_workers=0)

        # Initialise model
        if init_mode == "deap":
            model = build_icare_model_from_deap(DEAP_CKPT, args.device)
        else:
            model = LorentzianEncoder(
                n_eeg_ch=N_EEG_ICARE, lorentz_dim=LORENTZ_DIM,
                num_classes=NUM_DOC_CLASSES,
            ).to(args.device)

        criterion = FocalLoss(gamma=2.0, alpha=alpha_w)
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)

        best_fold_auc   = -1.0
        best_fold_state: Optional[dict] = None
        patience_cnt    = 0

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
            if ep % 10 == 0 or ep == 1:
                log.info(
                    f"  ep {ep:3d}/{args.epochs}  "
                    f"loss={total_loss/max(len(train_loader),1):.4f}  "
                    f"acc={acc:.4f}  auc={auc:.4f}"
                )

            if not np.isnan(auc) and auc > best_fold_auc:
                best_fold_auc   = auc
                best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    log.info(f"  Early stop at epoch {ep}")
                    break

        fold_aucs.append(best_fold_auc)
        log.info(f"  [{init_mode}] Fold {fold_idx} best AUC = {best_fold_auc:.4f}")

        if best_fold_state is not None and best_fold_auc > best_auc:
            best_auc   = best_fold_auc
            best_state = best_fold_state

    return fold_aucs, best_auc, best_state


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args) -> None:
    log.info("=" * 70)
    log.info("E2 ablation: DEAP pre-training vs. random initialisation on I-CARE")
    log.info("=" * 70)

    # ── Load I-CARE ──────────────────────────────────────────────────────────
    from src.stage3_doc.dataset_icare import ICareDataset
    log.info("\nLoading I-CARE dataset …")
    ds        = ICareDataset()
    epochs_t  = ds.epochs
    labels_t  = ds.labels
    subj_np   = ds.subject_ids.numpy()
    log.info(f"  {len(ds):,} epochs | {int(subj_np.max())+1} patients")
    log.info(f"  Class counts: {np.bincount(labels_t.numpy()).tolist()}")

    # ── Arm A: random init (read from existing checkpoint if available) ───────
    random_aucs: List[float] = []
    random_best_auc: float   = float("nan")
    if ICARE_CKPT.exists():
        ckpt = torch.load(ICARE_CKPT, map_location="cpu", weights_only=False)
        random_aucs     = ckpt.get("cv_aucs", [])
        random_best_auc = float(ckpt.get("best_auc", float("nan")))
        log.info(
            f"\n[random] Loaded existing checkpoint ({ICARE_CKPT.name}) — "
            f"skipping re-training.\n"
            f"  Fold AUCs : {[f'{a:.4f}' for a in random_aucs]}\n"
            f"  Best AUC  : {random_best_auc:.4f}"
        )
    else:
        log.info("\n[random] e2_doc_icare.pt not found — training from scratch …")
        random_aucs, random_best_auc, _ = _run_cv(
            epochs_t, labels_t, subj_np, args, init_mode="random"
        )

    # ── Arm B: DEAP pre-training ─────────────────────────────────────────────
    if not DEAP_CKPT.exists():
        log.error(
            f"DEAP checkpoint not found: {DEAP_CKPT}\n"
            "  Please run Stage 2 training first:\n"
            "    python -m src.stage2_emotion.train_emotion_encoder --dataset deap"
        )
        return

    log.info("\n[deap] Fine-tuning from DEAP pre-trained weights …")
    deap_aucs, deap_best_auc, deap_state = _run_cv(
        epochs_t, labels_t, subj_np, args, init_mode="deap"
    )

    # ── Save DEAP-transfer checkpoint ────────────────────────────────────────
    if deap_state is not None:
        OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": deap_state,
                "config": {
                    "n_eeg_ch":    N_EEG_ICARE,
                    "lorentz_dim": LORENTZ_DIM,
                    "num_classes": NUM_DOC_CLASSES,
                    "stage":       "e2_doc_icare_from_deap",
                    "init":        "deap_pretrained",
                    "geometry":    "lorentzian",
                },
                "cv_aucs":  deap_aucs,
                "best_auc": deap_best_auc,
            },
            OUT_CKPT,
        )
        log.info(f"  Saved → {OUT_CKPT}")

    # ── Summary ──────────────────────────────────────────────────────────────
    rand_mean = float(np.mean(random_aucs)) if random_aucs else float("nan")
    rand_std  = float(np.std(random_aucs))  if random_aucs else float("nan")
    deap_mean = float(np.mean(deap_aucs))   if deap_aucs   else float("nan")
    deap_std  = float(np.std(deap_aucs))    if deap_aucs   else float("nan")
    delta     = deap_mean - rand_mean

    log.info("\n" + "=" * 70)
    log.info("RESULTS SUMMARY")
    log.info("=" * 70)
    log.info(f"  Random init  (scratch)  : AUC {rand_mean:.4f} ± {rand_std:.4f}  "
             f"(best fold: {random_best_auc:.4f})")
    log.info(f"  DEAP pre-train + ft     : AUC {deap_mean:.4f} ± {deap_std:.4f}  "
             f"(best fold: {deap_best_auc:.4f})")
    log.info(f"  Δ (DEAP − random)       : {delta:+.4f}")
    if delta > 0.01:
        log.info("  → DEAP pre-training provides meaningful gain.")
    elif delta > -0.01:
        log.info("  → DEAP pre-training provides negligible gain (within noise).")
    else:
        log.info("  → DEAP pre-training slightly hurts performance (domain mismatch).")
    log.info("=" * 70)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows: List[Dict] = []
    for fold_idx, (ra, da) in enumerate(zip(random_aucs, deap_aucs), start=1):
        rows.append({
            "fold":        fold_idx,
            "random_init": f"{ra:.4f}",
            "deap_pretrain": f"{da:.4f}",
            "delta":       f"{da - ra:+.4f}",
        })
    rows.append({
        "fold": "mean ± std",
        "random_init":   f"{rand_mean:.4f} ± {rand_std:.4f}",
        "deap_pretrain": f"{deap_mean:.4f} ± {deap_std:.4f}",
        "delta":         f"{delta:+.4f}",
    })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "random_init", "deap_pretrain", "delta"])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"  CSV → {OUT_CSV}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="E2 ablation: DEAP pre-training vs. random init on I-CARE"
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr",     type=float, default=5e-4)
    p.add_argument("--batch",  type=int, default=64)
    p.add_argument("--folds",  type=int, default=5)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested (--device cuda) but torch.cuda.is_available() is False.\n"
            "Check your CUDA/driver installation or pass --device cpu."
        )
    main(args)
