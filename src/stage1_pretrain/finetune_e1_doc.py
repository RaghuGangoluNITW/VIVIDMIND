"""
Stage 1b — Fine-tune FractalSSL backbone on I-CARE for DOC classification.

Loads the pre-trained FractalSSL backbone (e1_fractalssl_tuh.pt), freezes
it for the first FREEZE_EPOCHS then trains end-to-end.  A 3-class linear
head (UWS / MCS / HC) is added on top of the frozen embeddings.

Training uses:
  • WeightedRandomSampler  — oversamples UWS / MCS to fix 10/14/76% imbalance
  • FocalLoss (gamma=2)    — focuses on hard minority-class examples
  • Cosine-annealing LR with warm-up

Output:
  results/checkpoints/e1_doc_finetuned.pt
  results/checkpoints/e1_doc_finetuned.pkl

CLI:
  python -m src.stage1_pretrain.finetune_e1_doc [options]

Options:
  --epochs   INT    Total fine-tuning epochs (default 50)
  --freeze   INT    Epochs to keep backbone frozen (default 10)
  --lr       FLOAT  Learning rate for classifier head (default 1e-3)
  --lr_bb    FLOAT  LR for backbone after unfreeze (default 1e-4)
  --folds    INT    Patient-level CV folds (default 5)
  --device   STR    "cuda" or "cpu"
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
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ICARE_DIR,
    SFREQ_ICARE,
    CKPT_ROOT,
    NUM_DOC_CLASSES,
    DOC_CLASSES,
    RANDOM_SEED,
)
from src.models.fractal_ssl import FractalSSLBackbone

log = logging.getLogger(__name__)

CKPT_PATH = CKPT_ROOT / "e1_doc_finetuned.pt"


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
# E1 DOC Classifier
# ─────────────────────────────────────────────────────────────────────────────

class E1DocClassifier(nn.Module):
    """
    FractalSSL backbone + 3-class DOC linear head.
    Backbone weights loaded from pre-trained checkpoint.
    """

    def __init__(self, backbone: FractalSSLBackbone, embed_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, NUM_DOC_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb    = self.backbone(x)
        logits = self.head(emb)
        return logits, emb

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _make_weighted_loader(
    dataset: Subset,
    labels:  np.ndarray,
    batch_size: int,
    shuffle_seed: int = 0,
) -> DataLoader:
    counts  = np.bincount(labels, minlength=NUM_DOC_CLASSES).astype(float)
    counts  = np.maximum(counts, 1.0)
    inv_f   = 1.0 / counts
    sw      = torch.tensor(inv_f[labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def finetune_e1(args) -> None:
    # ── Load I-CARE ─────────────────────────────────────────────────────────
    from src.stage3_doc.dataset_icare import ICareDataset
    log.info("Loading I-CARE dataset …")
    icare_ds = ICareDataset()
    epochs_t  = icare_ds.epochs          # (N, 19, T)  float32
    labels_t  = icare_ds.labels          # (N,)  int64
    subj_t    = icare_ds.subject_ids     # (N,)  int64

    labels_np = labels_t.numpy()
    subj_np   = subj_t.numpy()
    n_subjects = int(subj_np.max()) + 1
    log.info(f"  {len(epochs_t):,} epochs, {n_subjects} patients, "
             f"class counts: {np.bincount(labels_np).tolist()}")

    # ── Load pre-trained backbone ────────────────────────────────────────────
    e1_ckpt = CKPT_ROOT / "e1_fractalssl_tuh.pt"
    if not e1_ckpt.exists():
        raise FileNotFoundError(f"E1 checkpoint not found: {e1_ckpt}\n"
                                "Run Stage 1 first.")

    ckpt      = torch.load(e1_ckpt, map_location="cpu")
    cfg       = ckpt.get("config", {})
    n_ch      = cfg.get("n_channels", cfg.get("n_eeg_ch", 19))
    embed_dim = cfg.get("embed_dim", 128)

    backbone  = FractalSSLBackbone(n_channels=n_ch, embed_dim=embed_dim)
    # Load backbone weights from FractalSSL full model state
    full_state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    bb_state   = {k.replace("backbone.", ""): v
                  for k, v in full_state.items() if k.startswith("backbone.")}
    backbone.load_state_dict(bb_state, strict=True)
    log.info(f"  Backbone loaded from {e1_ckpt.name}  (n_ch={n_ch}, embed={embed_dim})")

    # ── Patient-level CV ─────────────────────────────────────────────────────
    patient_labels = np.array([
        int(np.bincount(labels_np[subj_np == s]).argmax())
        for s in range(n_subjects)
    ])
    subjects  = np.arange(n_subjects)
    skf       = StratifiedKFold(n_splits=args.folds, shuffle=True,
                                random_state=RANDOM_SEED)
    full_ds   = TensorDataset(epochs_t, labels_t)

    fold_aucs: List[float] = []
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

        train_loader = _make_weighted_loader(
            Subset(full_ds, train_idx), tr_labels, args.batch)
        val_loader = DataLoader(
            Subset(full_ds, val_idx), batch_size=args.batch * 2,
            shuffle=False, num_workers=0)

        # Fresh model per fold
        bb_fold = FractalSSLBackbone(n_channels=n_ch, embed_dim=embed_dim)
        bb_fold.load_state_dict(bb_state, strict=True)
        model = E1DocClassifier(bb_fold, embed_dim=embed_dim).to(args.device)
        model.freeze_backbone()

        criterion = FocalLoss(gamma=2.0, alpha=alpha_w)
        opt_head  = torch.optim.AdamW(model.head.parameters(),
                                       lr=args.lr, weight_decay=1e-4)
        opt_all   = torch.optim.AdamW(model.parameters(),
                                       lr=args.lr_bb, weight_decay=1e-4)
        sched     = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_all, T_max=max(args.epochs - args.freeze, 1), eta_min=1e-6)

        best_fold_auc = -1.0
        best_fold_state: Optional[dict] = None
        patience_cnt = 0
        PATIENCE = 10

        for ep in range(1, args.epochs + 1):
            # Unfreeze backbone after freeze_epochs
            if ep == args.freeze + 1:
                model.unfreeze_backbone()
                log.info(f"  Epoch {ep}: backbone unfrozen")

            opt = opt_all if ep > args.freeze else opt_head
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(args.device), yb.to(args.device)
                opt.zero_grad()
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if ep > args.freeze:
                sched.step()

            # Validate
            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(args.device)
                    logits, _ = model(xb)
                    all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
                    all_labels.append(yb.numpy())
            probs_np  = np.concatenate(all_probs)
            labels_vl = np.concatenate(all_labels)
            try:
                auc = roc_auc_score(labels_vl, probs_np,
                                    multi_class="ovr", average="macro")
            except Exception:
                auc = float("nan")
            acc = float((probs_np.argmax(1) == labels_vl).mean())
            log.info(f"  ep {ep:3d}/{args.epochs}  acc={acc:.4f}  auc={auc:.4f}" if not np.isnan(auc)
                     else f"  ep {ep:3d}/{args.epochs}  acc={acc:.4f}  auc=—")

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
                "n_channels": n_ch,
                "embed_dim":  embed_dim,
                "num_classes": NUM_DOC_CLASSES,
                "stage":      "e1_doc_finetuned",
                "dataset":    "I-CARE",
            },
            "cv_aucs": fold_aucs,
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
    p = argparse.ArgumentParser(description="Fine-tune E1 FractalSSL on I-CARE DOC data")
    p.add_argument("--epochs", type=int,   default=50)
    p.add_argument("--freeze", type=int,   default=10)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--lr_bb",  type=float, default=1e-4)
    p.add_argument("--batch",  type=int,   default=64)
    p.add_argument("--folds",  type=int,   default=5)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        level=logging.INFO)
    args = _parse_args()
    log.info(f"E1 DOC fine-tuning — device: {args.device}")
    finetune_e1(args)


if __name__ == "__main__":
    main()
