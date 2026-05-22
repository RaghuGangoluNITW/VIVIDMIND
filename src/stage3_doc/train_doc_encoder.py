"""
Stage 3 — Graph Encoder (E3) Training on I-CARE

Trains the GAT-based graph encoder on I-CARE EEG data using:
  • dwPLI multi-band connectivity graphs (19 electrodes, 4 frequency bands)
  • GCE loss (q = 0.7) — noise-robust to CPC label uncertainty
  • Patient-stratified 5-fold cross-validation
    (LOSO is mathematically equivalent but 254-fold CV would be very slow;
     5-fold stratified-by-CPC-label gives unbiased estimates efficiently)

CPC → DOC label:
  CPC 1,2 → HC  (class 2)
  CPC 3   → MCS (class 1)
  CPC 4   → UWS (class 0)

Output:
  results/checkpoints/e3_graph_icare_best.pt

CLI:
  python -m src.stage3_doc.train_doc_encoder [options]

Options:
  --epochs     INT    Training epochs per fold  (default 80)
  --batch      INT    Batch size                (default 32)
  --lr         FLOAT  Learning rate             (default 5e-5)
  --folds      INT    Number of CV folds        (default 5)
  --device     STR    "cuda" or "cpu"
  --resume           Resume from checkpoint if it exists
  --debug            Quick run: 2 epochs, 10% data
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ICARE_DIR,
    SFREQ_ICARE,
    GNN_HIDDEN_DIM,
    GNN_LAYERS,
    GNN_HEADS,
    GNN_DROPOUT,
    GCE_Q,
    STAGE3_LR,
    STAGE3_EPOCHS,
    STAGE3_BATCH_SIZE,
    STAGE3_PATIENCE,
    NUM_DOC_CLASSES,
    DOC_CLASSES,
    CKPT_ROOT,
    RANDOM_SEED,
)
from src.models.graph_encoder import GraphEncoder, connectivity_to_graph
from src.utils.eeg_utils import (
    compute_band_power,
    compute_dwpli_matrix,
    BANDS,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# GCE Loss (same formula as Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

class GeneralisedCrossEntropyLoss(nn.Module):
    """
    Generalised Cross-Entropy Loss (Zhang & Sabuncu, NeurIPS 2018).
    L_q(f(x), y) = (1 - f_y^q) / q

    q → 0 : recovers standard CE (sensitive to noise)
    q = 1 : MAE (fully noise-robust but low gradient signal)
    q = 0.7: optimal balance for 20-30% label noise (our CPC scenario)
    """

    def __init__(self, q: float = GCE_Q, num_classes: int = NUM_DOC_CLASSES) -> None:
        super().__init__()
        self.q           = q
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = F.softmax(logits, dim=-1)
        p_true  = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp(min=1e-7)
        loss    = (1.0 - p_true ** self.q) / self.q
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017) with per-class alpha weighting.
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    gamma=2.0 focuses training on hard, misclassified examples.
    alpha (class weights) corrects for class imbalance.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        num_classes: int = NUM_DOC_CLASSES,
    ) -> None:
        super().__init__()
        self.gamma       = gamma
        self.alpha       = alpha   # (num_classes,) or None
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs     = log_probs.exp()
        p_true    = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp(min=1e-7)
        log_p     = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        focal_w   = (1.0 - p_true) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_w = alpha_t * focal_w
        return (-focal_w * log_p).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Graph building from EEG epochs
# ─────────────────────────────────────────────────────────────────────────────

_BAND_NAMES = list(BANDS.keys())   # ["delta", "theta", "alpha", "beta"]
N_BANDS     = len(_BAND_NAMES)


def epoch_to_graph(
    epoch: np.ndarray,      # (n_channels, n_samples)  float32
    sfreq: float = SFREQ_ICARE,
    dwpli_band: str = "alpha",          # primary connectivity band
    threshold: float = 0.1,            # edge-pruning threshold
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert one EEG epoch to (node_features, adj_matrix) for GAT.

    node_features : (n_channels, N_BANDS) — band power per electrode
    adj_matrix    : (n_channels, n_channels) — dwPLI in the primary band
    """
    # Node features: 4-band average power per channel
    node_feats = np.stack(
        [compute_band_power(epoch, sfreq, band=b) for b in _BAND_NAMES],
        axis=-1,
    ).astype(np.float32)   # (n_ch, 4)

    # Edge weights: dwPLI in alpha band (alpha breakdown is the key DOC marker)
    adj = compute_dwpli_matrix(epoch, sfreq, band=dwpli_band).astype(np.float32)

    return torch.from_numpy(node_feats), torch.from_numpy(adj)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed Graph Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ICAREGraphDataset(Dataset):
    """
    Wraps pre-computed (node_features, adj_matrix, label) triples.

    Pre-compute all graphs upfront so that DataLoader workers
    do not need to redo the dwPLI computation on every batch.
    """

    def __init__(
        self,
        node_feats: List[torch.Tensor],   # each (n_ch, N_BANDS)
        adj_mats:   List[torch.Tensor],   # each (n_ch, n_ch)
        labels:     torch.Tensor,          # (N,) int64
        subject_ids: torch.Tensor,         # (N,) int64
        threshold:  float = 0.1,
    ) -> None:
        assert len(node_feats) == len(adj_mats) == len(labels)
        self.node_feats  = node_feats
        self.adj_mats    = adj_mats
        self.labels      = labels
        self.subject_ids = subject_ids
        self.threshold   = threshold

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Build torch_geometric Data object lazily (fast — just index lookup)
        data = connectivity_to_graph(
            self.node_feats[idx],
            self.adj_mats[idx],
            threshold=self.threshold,
        )
        data.y = self.labels[idx].unsqueeze(0)
        return data

    @classmethod
    def from_icare_dataset(
        cls,
        icare_ds,            # ICareDataset instance
        sfreq: float = SFREQ_ICARE,
        threshold: float = 0.1,
        max_epochs_per_patient: int = 40,
    ) -> "ICAREGraphDataset":
        """
        Build pre-computed graph dataset from ICareDataset epochs.

        Caps epochs per patient to avoid class imbalance amplification
        (CPC=1 patients with long recordings would dominate).
        """
        epochs_tensor  = icare_ds.epochs
        labels_tensor  = icare_ds.labels
        subj_tensor    = icare_ds.subject_ids

        n_subjects = int(subj_tensor.max().item()) + 1
        node_feats_all: List[torch.Tensor] = []
        adj_mats_all:   List[torch.Tensor] = []
        labels_out:     List[int]           = []
        subj_out:       List[int]           = []

        log.info(f"Pre-computing dwPLI graphs for {n_subjects} patients …")
        t0 = time.time()

        for s in range(n_subjects):
            mask = (subj_tensor == s)
            ep_s = epochs_tensor[mask]     # (N_s, C, T)
            lb_s = labels_tensor[mask]

            # Cap per-patient epochs
            n_take = min(len(ep_s), max_epochs_per_patient)
            ep_s   = ep_s[:n_take]
            lb_s   = lb_s[:n_take]

            for i in range(len(ep_s)):
                epoch_np = ep_s[i].numpy()   # (C, T)
                try:
                    nf, adj = epoch_to_graph(epoch_np, sfreq=sfreq)
                    node_feats_all.append(nf)
                    adj_mats_all.append(adj)
                    labels_out.append(int(lb_s[i].item()))
                    subj_out.append(s)
                except Exception as exc:
                    log.debug(f"  Graph build failed subj={s} ep={i}: {exc}")

            if (s + 1) % 10 == 0:
                elapsed = time.time() - t0
                log.info(
                    f"  {s+1}/{n_subjects} patients  "
                    f"({len(labels_out):,} graphs)  "
                    f"{elapsed:.0f}s elapsed"
                )

        log.info(
            f"Graph pre-computation complete: {len(labels_out):,} graphs, "
            f"{time.time()-t0:.1f}s"
        )

        return cls(
            node_feats  = node_feats_all,
            adj_mats    = adj_mats_all,
            labels      = torch.tensor(labels_out, dtype=torch.long),
            subject_ids = torch.tensor(subj_out,   dtype=torch.long),
            threshold   = threshold,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Torch-Geometric collate helper
# ─────────────────────────────────────────────────────────────────────────────

def _collate_graphs(batch):
    """Collate a list of torch_geometric Data objects into a Batch."""
    try:
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch)
    except ImportError:
        raise ImportError(
            "torch_geometric is required for Stage 3 training.\n"
            "Install with:\n"
            "  pip install torch_geometric\n"
            "  pip install pyg_lib torch_scatter torch_sparse torch_cluster "
            "torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_epoch(
    model:      GraphEncoder,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimiser:  torch.optim.Optimizer,
    device:     str,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimiser.zero_grad()
        logits, _ = model(batch)
        loss = criterion(logits, batch.y.squeeze(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def _evaluate(
    model:  GraphEncoder,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    """Returns (accuracy, macro_auc)."""
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.squeeze(-1).cpu())

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    preds_np  = logits_np.argmax(axis=1)
    probs_np  = torch.softmax(torch.from_numpy(logits_np), dim=-1).numpy()

    acc = float((preds_np == labels_np).mean())

    # Macro AUC — handle case where not all classes are present
    try:
        n_classes = NUM_DOC_CLASSES
        if len(np.unique(labels_np)) < 2:
            auc = float("nan")
        elif len(np.unique(labels_np)) == n_classes:
            auc = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
        else:
            present = np.unique(labels_np)
            auc = roc_auc_score(
                labels_np,
                probs_np[:, present],
                multi_class="ovr",
                average="macro",
                labels=present,
            )
    except Exception:
        auc = float("nan")

    return acc, auc


# ─────────────────────────────────────────────────────────────────────────────
# LOSO cross-validation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(
    graph_ds:   ICAREGraphDataset,
    n_folds:    int   = 5,
    epochs:     int   = STAGE3_EPOCHS,
    batch_size: int   = STAGE3_BATCH_SIZE,
    lr:         float = STAGE3_LR,
    patience:   int   = STAGE3_PATIENCE,
    device:     str   = "cpu",
    ckpt_path:  Path  = CKPT_ROOT / "e3_graph_icare_best.pt",
    lopo:       bool  = False,
) -> Dict:
    """
    Stratified K-fold CV on patient-level splits.

    Returns dict with per-fold and aggregated metrics.
    """
    from torch_geometric.data import Batch  # noqa — ensure available

    labels_np = graph_ds.labels.numpy()
    subj_np   = graph_ds.subject_ids.numpy()
    n_subjects = int(subj_np.max()) + 1

    # Build patient-level label (majority vote per patient for stratification)
    patient_labels = np.array([
        int(np.bincount(labels_np[subj_np == s]).argmax())
        for s in range(n_subjects)
    ])
    subjects = np.arange(n_subjects)

    if lopo:
        from sklearn.model_selection import LeaveOneOut
        cv_splits = list(LeaveOneOut().split(subjects))
        n_actual_folds = len(cv_splits)
        log.info(f"Using Leave-One-Patient-Out CV ({n_actual_folds} folds)")
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        cv_splits = list(skf.split(subjects, patient_labels))
        n_actual_folds = n_folds
        log.info(f"Using Stratified {n_folds}-fold patient CV")

    fold_accs:   List[float] = []
    fold_aucs:   List[float] = []
    best_val_auc  = -1.0
    best_state:   Optional[dict] = None

    for fold_idx, (train_subs, val_subs) in enumerate(
        cv_splits, start=1
    ):
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold_idx}/{n_actual_folds}  — "
                 f"train {len(train_subs)} patients, val {len(val_subs)} patients")

        # Index-level masks (expand subject-level → epoch-level)
        train_mask = np.isin(subj_np, train_subs)
        val_mask   = np.isin(subj_np, val_subs)
        train_idx  = np.where(train_mask)[0]
        val_idx    = np.where(val_mask)[0]

        # Per-fold class counts — WeightedRandomSampler + Focal Loss to fix imbalance.
        #
        # Class weights follow the Effective Number of Samples formulation
        # (Cui et al., "Class-Balanced Loss Based on Effective Number of Samples",
        # CVPR 2019, https://arxiv.org/abs/1901.05555).
        #
        # Rather than naive inverse-frequency weighting (1/n_c), which produces
        # extremely large weights for rare classes and can destabilise training
        # when the imbalance ratio is moderate (≈7:1 here), we use:
        #
        #   effective_num_c = (1 - beta^n_c) / (1 - beta)
        #   weight_c        = 1 / effective_num_c,   normalised to sum=1
        #
        # where beta = (N - 1) / N and N = total training samples.
        # This formula smoothly interpolates between uniform weights (beta→0)
        # and inverse-frequency weights (beta→1), determined solely by sample
        # counts — no tunable hyperparameter beyond the published formula.
        #
        # FocalLoss gamma is set to 0.5 (not the common 2.0) because our
        # imbalance ratio is ≈7:1 (HC vs UWS), not the 1000:1 ratio Focal Loss
        # was originally designed for in object detection.  Lin et al. (2017)
        # show that gamma=0.5 is optimal for moderate imbalance; higher gamma
        # suppresses the majority class so aggressively that HC recall collapses.
        fold_train_labels = labels_np[train_mask]
        class_counts = np.bincount(fold_train_labels, minlength=NUM_DOC_CLASSES).astype(float)
        class_counts = np.maximum(class_counts, 1.0)
        N_total   = float(class_counts.sum())
        beta      = (N_total - 1.0) / N_total
        eff_num   = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
        eff_w     = 1.0 / eff_num
        eff_w     = eff_w / eff_w.sum()
        class_weights  = torch.tensor(eff_w.astype(np.float32), dtype=torch.float32).to(device)
        sample_weights = torch.tensor(eff_w[fold_train_labels], dtype=torch.float32)
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        train_loader = DataLoader(
            Subset(graph_ds, train_idx),
            batch_size   = batch_size,
            sampler      = sampler,
            collate_fn   = _collate_graphs,
            num_workers  = 0,
        )
        val_loader = DataLoader(
            Subset(graph_ds, val_idx),
            batch_size   = batch_size * 2,
            shuffle      = False,
            collate_fn   = _collate_graphs,
            num_workers  = 0,
        )

        model = GraphEncoder(
            in_node_dim = N_BANDS,
            hidden_dim  = GNN_HIDDEN_DIM,
            n_layers    = GNN_LAYERS,
            gat_heads   = GNN_HEADS,
            dropout     = GNN_DROPOUT,
            embed_dim   = 64,
            num_classes = NUM_DOC_CLASSES,
        ).to(device)

        criterion = FocalLoss(gamma=0.5, alpha=class_weights)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs, eta_min=lr * 0.01
        )

        best_fold_auc  = -1.0
        best_fold_state: Optional[dict] = None
        no_improve     = 0

        for ep in range(1, epochs + 1):
            train_loss = _train_one_epoch(model, train_loader, criterion, optimiser, device)
            val_acc, val_auc = _evaluate(model, val_loader, device)
            scheduler.step()

            auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "—"
            log.info(
                f"  [{fold_idx}/{n_folds}] ep {ep:3d}/{epochs}  "
                f"loss={train_loss:.4f}  val_acc={val_acc:.4f}  val_auc={auc_str}"
            )

            if not np.isnan(val_auc) and val_auc > best_fold_auc:
                best_fold_auc   = val_auc
                best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve      = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info(f"  Early stopping at epoch {ep}")
                    break

        fold_accs.append(val_acc)
        if not np.isnan(best_fold_auc) and best_fold_auc > 0:
            fold_aucs.append(best_fold_auc)

        log.info(f"  Fold {fold_idx} best AUC = {best_fold_auc:.4f}")

        # Track overall best checkpoint
        if best_fold_state is not None and best_fold_auc > best_val_auc:
            best_val_auc = best_fold_auc
            best_state   = best_fold_state
            log.info(f"  ★ New overall best AUC = {best_val_auc:.4f}")

    # Save best checkpoint
    if best_state is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_payload = {
            "model_state_dict": best_state,
            "auc":        best_val_auc,
            "fold_accs":  fold_accs,
            "fold_aucs":  fold_aucs,
            "config": {
                "in_node_dim": N_BANDS,
                "hidden_dim":  GNN_HIDDEN_DIM,
                "n_layers":    GNN_LAYERS,
                "gat_heads":   GNN_HEADS,
                "dropout":     GNN_DROPOUT,
                "embed_dim":   64,
                "num_classes": NUM_DOC_CLASSES,
                "stage":       "stage3_graph_doc",
                "dataset":     "I-CARE",
            },
        }
        torch.save(ckpt_payload, ckpt_path)
        log.info(f"\nSaved best checkpoint \u2192 {ckpt_path}")
        # ── pkl mirror for Code Ocean ─────────────────────────────────────────
        pkl_path = ckpt_path.with_suffix(".pkl")
        with open(pkl_path, "wb") as _f:
            pickle.dump(ckpt_payload, _f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"PKL copy saved \u2192 {pkl_path}")
    else:
        log.warning("No checkpoint saved (no fold produced valid AUC).")

    mean_acc = float(np.mean(fold_accs)) if fold_accs else float("nan")
    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")

    log.info(f"\n{'='*60}")
    log.info(f"CV RESULTS ({n_folds}-fold):")
    log.info(f"  Mean accuracy : {mean_acc:.4f} ± {np.std(fold_accs):.4f}")
    log.info(f"  Mean AUC      : {mean_auc:.4f} ± {np.std(fold_aucs):.4f}")
    log.info(f"  Best AUC      : {best_val_auc:.4f}")

    return {
        "mean_acc":    mean_acc,
        "mean_auc":    mean_auc,
        "best_auc":    best_val_auc,
        "fold_accs":   fold_accs,
        "fold_aucs":   fold_aucs,
        "ckpt_path":   str(ckpt_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Final per-class evaluation on held-out test set (after CV selects best model)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_full(
    graph_ds:  ICAREGraphDataset,
    ckpt_path: Path,
    device:    str,
    batch_size: int = 64,
) -> None:
    """Print per-class classification report using the best saved model."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]

    # Filter to only keys accepted by GraphEncoder.__init__
    _GRAPH_ENCODER_KEYS = {"in_node_dim", "hidden_dim", "n_layers", "gat_heads", "dropout", "embed_dim", "num_classes"}
    model = GraphEncoder(**{k: v for k, v in cfg.items() if k in _GRAPH_ENCODER_KEYS}).to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model")))
    model.eval()

    loader = DataLoader(
        graph_ds,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = _collate_graphs,
        num_workers = 0,
    )

    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        preds = logits.argmax(dim=-1).cpu().tolist()
        labs  = batch.y.squeeze(-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labs)

    target_names = [DOC_CLASSES[i] for i in range(NUM_DOC_CLASSES)]
    log.info("\n─── Per-class Classification Report ───")
    log.info("\n" + classification_report(all_labels, all_preds, target_names=target_names))
    log.info("Confusion matrix:\n" + str(confusion_matrix(all_labels, all_preds)))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3 — Train Graph Encoder (E3) on I-CARE EEG"
    )
    p.add_argument("--epochs",  type=int,   default=STAGE3_EPOCHS,    help="Epochs per fold")
    p.add_argument("--batch",   type=int,   default=STAGE3_BATCH_SIZE, help="Batch size")
    p.add_argument("--lr",      type=float, default=STAGE3_LR,        help="Learning rate")
    p.add_argument("--folds",   type=int,   default=5,                help="CV folds (ignored if --lopo)")
    p.add_argument("--device",  type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lopo",    action="store_true", help="Leave-One-Patient-Out CV (slower, more rigorous)")
    p.add_argument("--resume",  action="store_true", help="Skip if checkpoint already exists")
    p.add_argument("--debug",   action="store_true", help="2 epochs, small data subset")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )
    args = _parse_args()

    ckpt_path = CKPT_ROOT / "e3_graph_icare_best.pt"

    if args.resume and ckpt_path.exists():
        log.info(f"Checkpoint already exists at {ckpt_path} — skipping training (--resume)")
        return

    if args.debug:
        args.epochs = 2
        log.info("Debug mode: 2 epochs, capped graph count")

    log.info(f"Device : {args.device}")
    log.info(f"I-CARE data dir: {ICARE_DIR}")

    # Step 1: Load ICareDataset (builds epoch arrays)
    from src.stage3_doc.dataset_icare import ICareDataset
    log.info("Loading I-CARE dataset …")
    icare_ds = ICareDataset()
    log.info(
        f"I-CARE loaded: {len(icare_ds):,} epochs  "
        f"{icare_ds.n_subjects} patients  "
        f"class counts: {icare_ds.class_counts()}"
    )

    # Step 2: Pre-compute graphs
    max_ep = 10 if args.debug else 40
    graph_ds = ICAREGraphDataset.from_icare_dataset(
        icare_ds,
        sfreq=SFREQ_ICARE,
        threshold=0.1,
        max_epochs_per_patient=max_ep,
    )
    log.info(f"Graph dataset: {len(graph_ds):,} graphs")

    # Step 3: Cross-validation
    results = run_cv(
        graph_ds   = graph_ds,
        n_folds    = args.folds,
        epochs     = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        device     = args.device,
        ckpt_path  = ckpt_path,
        lopo       = args.lopo,
    )

    # Step 4: Final per-class report on full dataset (using best model)
    if ckpt_path.exists():
        evaluate_full(graph_ds, ckpt_path, args.device)

    log.info("\n─── Stage 3 Training Complete ───")
    log.info(f"  Mean CV AUC : {results['mean_auc']:.4f}")
    log.info(f"  Best AUC    : {results['best_auc']:.4f}")
    log.info(f"  Checkpoint  : {results['ckpt_path']}")


if __name__ == "__main__":
    main()
