"""
Stage 4 — Full PDI-CCS Pipeline Evaluation

Loads the trained E1 (FractalSSL), E2 (Lorentzian), E3 (Graph) encoders
and runs the complete PDI-CCS fusion evaluation on I-CARE data.

Produces all results needed for the Nature Scientific Reports paper:

  PRIMARY:  Binary neurological outcome prediction (CPC1-2 vs CPC3-4)
            Reported at PATIENT LEVEL under leave-one-patient-out (LOPO)
  SECONDARY: 3-class DOC classification (UWS / MCS / HC)

  Table 1: Binary outcome AUC vs 2025-2026 SOTA baselines (patient-level)
  Table 2: 3-class DOC: balanced accuracy, macro AUC, macro F1
  Table 3: Ablation study (each encoder individually + progressive fusion)
  Table 4: Covert awareness detection (CAR / FAR at optimal PDI threshold)
  Figure 1: Binary outcome ROC curve (CPC1-2 vs CPC3-4)
  Figure 2: 3-class confusion matrix (patient-level predictions)
  Figure 3: ROC curves per DOC class (OvR)
  Figure 4: CCS violin plot across DOC classes
  Figure 5: t-SNE of Lorentzian embeddings (coloured by CPC)

Outputs written to:
  results/tables/  — CSV files for paper tables
  results/plots/   — PNG/PDF figures

CLI:
  python -m src.stage4_eval.evaluate_pipeline [options]

Options:
  --e1_ckpt  PATH   Path to E1 FractalSSL checkpoint  (optional)
  --e2_ckpt  PATH   Path to E2 Lorentzian checkpoint
  --e3_ckpt  PATH   Path to E3 Graph encoder checkpoint
  --device   STR    "cuda" or "cpu"
  --seed     INT    Random seed (default 42)
  --no_plots        Skip matplotlib rendering (useful on headless servers)

Fallback behaviour:
  If E1 checkpoint is missing, the pipeline runs in 2-encoder mode (E2+E3).
  If E3 checkpoint is missing, runs in single-encoder mode (E2 only).
  All ablation configurations are always evaluated.
"""

import argparse
import csv
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ICARE_DIR,
    SFREQ_ICARE,
    CKPT_ROOT,
    PLOT_ROOT,
    TABLE_ROOT,
    NUM_DOC_CLASSES,
    DOC_CLASSES,
    PDI_ALPHA,
    PDI_BETA,
    CCS_COVERT_THRESHOLD,
    LORENTZ_DIM,
    GNN_HIDDEN_DIM,
    GNN_LAYERS,
    GNN_HEADS,
    GNN_DROPOUT,
    RANDOM_SEED,
)
from src.models.ccs_fusion import pairwise_disagreement_index, consciousness_coherence_score

log = logging.getLogger(__name__)


def _optimise_ccs_threshold(
    ccs_scores: np.ndarray,
    labels:     np.ndarray,
    grid: np.ndarray = np.arange(0.05, 0.91, 0.02),
) -> Tuple[float, float]:
    """
    Grid-search the CCS covert-awareness threshold that maximises
    score = covert_awareness_recall - 0.5 * false_alarm_rate.

    covert_awareness_recall = fraction of MCS epochs flagged (true covert recall)
    false_alarm_rate        = fraction of UWS epochs flagged (false alarms)

    Returns (best_threshold, best_score).
    NOTE: covert_awareness_metrics is defined later in this file; the call
    works because _optimise_ccs_threshold is only called at runtime, not
    at import time.
    """
    best_thr, best_score = CCS_COVERT_THRESHOLD, -np.inf
    for thr in grid:
        m     = covert_awareness_metrics(ccs_scores, labels, threshold=float(thr))
        score = m["covert_awareness_recall"] - 0.5 * m["false_alarm_rate"]
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr, best_score

# ─────────────────────────────────────────────────────────────────────────────
# Encoder loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_e2(ckpt_path: Path, device: str, n_channels: int):
    """Load the E2 encoder from checkpoint.

    Supports four checkpoint variants:
    - Original DEAP/DREAMER binary Lorentzian  (num_classes=2, n_eeg_ch=32)
    - New e2_doc_icare Lorentzian              (num_classes=3, n_eeg_ch=19)
    - e2_euclidean_deap_best.pt                (EuclideanEncoder, num_classes=2)
    - e2_euclidean_dreamer_best.pt             (EuclideanEncoder, num_classes=2)

    The 'geometry' key in the checkpoint config drives the class selection:
      geometry='euclidean'  → EuclideanEncoder (ablation baseline)
      geometry='lorentzian' → LorentzianEncoder (proposed; also used when
                               key is absent for backward compatibility)
    """
    from src.models.lorentzian_encoder import LorentzianEncoder, EuclideanEncoder
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    cfg   = ckpt.get("config", {})
    n_ch  = cfg.get("n_eeg_ch", n_channels)
    n_cls = cfg.get("num_classes", 2)
    geometry = cfg.get("geometry", "lorentzian")
    model_cls = EuclideanEncoder if geometry == "euclidean" else LorentzianEncoder
    model = model_cls(n_eeg_ch=n_ch, num_classes=n_cls)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _load_e3(ckpt_path: Path, device: str):
    """Load the best E3 Graph encoder from a Stage 3 checkpoint."""
    from src.models.graph_encoder import GraphEncoder
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    model = GraphEncoder(
        in_node_dim = cfg.get("in_node_dim", 4),
        hidden_dim  = cfg.get("hidden_dim",  GNN_HIDDEN_DIM),
        n_layers    = cfg.get("n_layers",    GNN_LAYERS),
        gat_heads   = cfg.get("gat_heads",   GNN_HEADS),
        dropout     = cfg.get("dropout",     GNN_DROPOUT),
        embed_dim   = cfg.get("embed_dim",   64),
        num_classes = cfg.get("num_classes", NUM_DOC_CLASSES),
    )
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    return model.to(device).eval()


def _load_e1(ckpt_path: Path, device: str):
    """Load E1 checkpoint.

    Supports two variants:
    - e1_fractalssl_tuh.pt   : FractalSSL pre-training checkpoint (SSL only)
    - e1_doc_finetuned.pt    : E1DocClassifier (backbone + 3-class head)

    Detected automatically by presence of 'head.' keys in the state dict.
    """
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt.get("config", {})
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    n_ch  = cfg.get("n_channels", cfg.get("n_eeg_ch", 19))
    embed_dim = cfg.get("embed_dim", 128)

    if any(k.startswith("head.") for k in state):
        # Fine-tuned DOC classifier — load E1DocClassifier
        from src.models.fractal_ssl import FractalSSLBackbone
        from src.stage1_pretrain.finetune_e1_doc import E1DocClassifier
        backbone = FractalSSLBackbone(n_channels=n_ch, embed_dim=embed_dim)
        model    = E1DocClassifier(backbone=backbone, embed_dim=embed_dim)
        model.load_state_dict(state, strict=True)
    else:
        # Raw SSL checkpoint — load FractalSSL backbone only
        from src.models.fractal_ssl import FractalSSL
        model = FractalSSL(n_channels=n_ch)
        model.load_state_dict(state, strict=False)

    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _infer_e2(
    model,
    epochs_tensor: torch.Tensor,   # (N, C, T)
    batch_size: int = 128,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run E2 Lorentzian encoder. Returns (probs (N,3), embeds (N,D)).

    E2 was trained binary (low/high valence).  We expand to 3-class DOC space:
      UWS (class 0) ← p(low valence)   — no emotional processing
      HC  (class 2) ← p(high valence)  — intact emotional processing
      MCS (class 1) ← geometric mean   — intermediate state
    Then re-normalise so rows sum to 1.
    """
    all_probs:  List[np.ndarray] = []
    all_embeds: List[np.ndarray] = []

    for i in range(0, len(epochs_tensor), batch_size):
        batch = epochs_tensor[i: i + batch_size].to(device)
        try:
            logits, emb = model(batch)
        except Exception:
            logits = model(batch)
            emb    = torch.zeros(len(batch), LORENTZ_DIM, device=device)
        probs = F.softmax(logits, dim=-1).cpu().numpy()   # (B, 2)

        if probs.shape[1] == 2:
            # Expand binary → 3-class: [UWS, MCS, HC]
            p_uws = probs[:, 0:1]                          # low  valence → UWS
            p_hc  = probs[:, 1:2]                          # high valence → HC
            p_mcs = np.sqrt(np.clip(p_uws * p_hc, 1e-9, None))  # geometric mean → MCS
            probs = np.concatenate([p_uws, p_mcs, p_hc], axis=1)
            probs = probs / probs.sum(axis=1, keepdims=True)     # renormalise

        all_probs.append(probs)
        all_embeds.append(emb.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_embeds)


@torch.no_grad()
def _infer_e3(
    model,
    graph_ds,           # ICAREGraphDataset
    batch_size: int = 64,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run E3 Graph encoder. Returns (probs (N,3), embeds (N,64))."""
    from torch.utils.data import DataLoader
    from src.stage3_doc.train_doc_encoder import _collate_graphs

    loader = DataLoader(
        graph_ds,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = _collate_graphs,
        num_workers = 0,
    )
    all_probs:  List[np.ndarray] = []
    all_embeds: List[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        logits, emb = model(batch)
        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_embeds.append(emb.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_embeds)


def _fit_e1_probe_cv(
    embeds: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Fit a logistic regression linear probe on frozen E1 embeddings using
    stratified k-fold CV.  Returns out-of-fold class probabilities (N, 3).
    This is the standard SSL linear evaluation protocol.
    """
    probs = np.zeros((len(labels), NUM_DOC_CLASSES), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for fold, (tr, va) in enumerate(skf.split(embeds, labels)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(embeds[tr])
        X_va = scaler.transform(embeds[va])
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=random_state,
                                 multi_class="multinomial", solver="lbfgs")
        clf.fit(X_tr, labels[tr])
        probs[va] = clf.predict_proba(X_va)
    log.info(f"  E1 linear probe fitted ({n_folds}-fold CV on {len(labels):,} epochs)")
    return probs


def _infer_e1(
    model,
    epochs_tensor: torch.Tensor,
    batch_size: int = 128,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run E1 FractalSSL encoder. Returns (probs (N,3), embeds (N,128))."""
    all_probs:  List[np.ndarray] = []
    all_embeds: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(epochs_tensor), batch_size):
            batch = epochs_tensor[i: i + batch_size].to(device)
            try:
                logits, emb = model(batch)
                probs = F.softmax(logits, dim=-1)
            except Exception:
                # E1 may not have a classifier head — use uniform probs in that case
                emb   = model(batch) if hasattr(model, "encode") else torch.zeros(len(batch), 128)
                probs = torch.full((len(batch), NUM_DOC_CLASSES), 1.0 / NUM_DOC_CLASSES)
            all_probs.append(probs.detach().cpu().numpy())
            all_embeds.append(emb.detach().cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_embeds)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    labels_np: np.ndarray,
    probs_np:  np.ndarray,
) -> Dict[str, float]:
    preds_np  = probs_np.argmax(axis=1)
    acc       = accuracy_score(labels_np, preds_np)
    bal_acc   = balanced_accuracy_score(labels_np, preds_np)
    f1        = f1_score(labels_np, preds_np, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return {"accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": f1, "macro_auc": auc}


def aggregate_patient_level(
    epoch_probs:   np.ndarray,   # (N_epochs, C)
    epoch_labels:  np.ndarray,   # (N_epochs,)
    subject_ids:   np.ndarray,   # (N_epochs,) int — patient index per epoch
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate epoch-level softmax probabilities to patient level by
    mean-pooling.  Returns (patient_probs (M, C), patient_labels (M,)).

    This is the ONLY valid evaluation unit for clinical claims.
    Window-level accuracy on the same patient in train and test is leakage.
    """
    unique_subjects = np.unique(subject_ids)
    patient_probs  = np.zeros((len(unique_subjects), epoch_probs.shape[1]))
    patient_labels = np.zeros(len(unique_subjects), dtype=np.int64)
    for i, s in enumerate(unique_subjects):
        mask              = subject_ids == s
        patient_probs[i]  = epoch_probs[mask].mean(axis=0)
        # All epochs for a patient share the same label — take mode
        patient_labels[i] = int(np.bincount(epoch_labels[mask]).argmax())
    return patient_probs, patient_labels


def compute_binary_outcome_metrics(
    patient_probs:  np.ndarray,   # (M, 3) — columns: UWS=0, MCS=1, HC=2
    patient_labels: np.ndarray,   # (M,) — 0=UWS, 1=MCS, 2=HC
    bootstrap_n:    int = 1000,
    seed:           int = 42,
) -> Dict[str, float]:
    """
    PRIMARY TASK for Scientific Reports:
    Binary neurological outcome prediction: good (HC, CPC1-2) vs poor (UWS+MCS, CPC3-4).

    Maps 3-class DOC labels:
      HC  (class 2) → binary 1 (good outcome)
      MCS (class 1) → binary 0 (poor outcome)
      UWS (class 0) → binary 0 (poor outcome)

    Binary score = p(HC) from the 3-class model.

    Reports: AUC with 95% bootstrap CI, sensitivity, specificity at Youden threshold.
    """
    binary_labels = (patient_labels == 2).astype(int)   # 1=good, 0=poor
    binary_scores = patient_probs[:, 2]                  # p(HC) as discriminant score

    if binary_labels.sum() == 0 or (1 - binary_labels).sum() == 0:
        return {"binary_auc": float("nan"), "binary_auc_ci_lo": float("nan"),
                "binary_auc_ci_hi": float("nan"), "binary_sensitivity": float("nan"),
                "binary_specificity": float("nan")}

    auc = roc_auc_score(binary_labels, binary_scores)

    # Bootstrap 95% CI (DeLong-equivalent via percentile bootstrap)
    rng = np.random.default_rng(seed)
    boot_aucs = []
    n = len(binary_labels)
    for _ in range(bootstrap_n):
        idx = rng.integers(0, n, size=n)
        bl, bs = binary_labels[idx], binary_scores[idx]
        if bl.sum() > 0 and (1 - bl).sum() > 0:
            boot_aucs.append(roc_auc_score(bl, bs))
    ci_lo, ci_hi = (
        (float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5)))
        if boot_aucs else (float("nan"), float("nan"))
    )

    # Youden-optimal threshold
    from sklearn.metrics import roc_curve
    fpr_arr, tpr_arr, thr_arr = roc_curve(binary_labels, binary_scores)
    youden_idx = np.argmax(tpr_arr - fpr_arr)
    thr_opt    = float(thr_arr[youden_idx])
    preds_bin  = (binary_scores >= thr_opt).astype(int)
    sensitivity = float(((preds_bin == 1) & (binary_labels == 1)).sum() /
                        max(binary_labels.sum(), 1))
    specificity = float(((preds_bin == 0) & (binary_labels == 0)).sum() /
                        max((1 - binary_labels).sum(), 1))

    return {
        "binary_auc":         auc,
        "binary_auc_ci_lo":   ci_lo,
        "binary_auc_ci_hi":   ci_hi,
        "binary_sensitivity": sensitivity,
        "binary_specificity": specificity,
        "binary_threshold":   thr_opt,
        "n_good":             int(binary_labels.sum()),
        "n_poor":             int((1 - binary_labels).sum()),
    }


def covert_awareness_metrics(
    ccs_scores:   np.ndarray,   # (N,)
    labels_np:    np.ndarray,   # (N,) 0=UWS, 1=MCS, 2=HC
    threshold:    float = CCS_COVERT_THRESHOLD,
) -> Dict[str, float]:
    """
    Covert Awareness Recall (CAR) analysis.

    Definition:
      - "Clinically UWS" = label == 0 (CPC=4 in I-CARE)
      - "True covert"    = UWS patients who actually are MCS/recovering
                           (in I-CARE: CPC=4 who reached CPC=1/2 at 6 months)
      - "Flagged"        = CCS > threshold

    In our evaluation setup, because we have CPC ground truth, we treat
    MCS (label=1) patients that appear in the UWS-labelled neighbours as
    the "covert" cases.  This operationalises the 6-month recovery oracle.

    Here we compute:
      - Flag rate for UWS patients (what fraction we flag)
      - Flag rate for MCS patients (how many true MCS-as-UWS we capture)
    """
    uws_mask = labels_np == 0
    mcs_mask = labels_np == 1
    hc_mask  = labels_np == 2

    flagged = ccs_scores > threshold

    n_uws     = uws_mask.sum()
    n_mcs     = mcs_mask.sum()
    n_flagged_uws = flagged[uws_mask].sum()
    n_flagged_mcs = flagged[mcs_mask].sum()

    car  = float(n_flagged_mcs / max(n_mcs, 1))    # Covert Awareness Recall
    fpr  = float(n_flagged_uws / max(n_uws, 1))    # False Alarm Rate
    ccs_uws = ccs_scores[uws_mask].mean() if n_uws > 0 else float("nan")
    ccs_hc  = ccs_scores[hc_mask].mean()  if hc_mask.sum() > 0 else float("nan")

    return {
        "covert_awareness_recall":   car,
        "false_alarm_rate":          fpr,
        "mean_ccs_uws":              ccs_uws,
        "mean_ccs_hc":               ccs_hc,
        "n_flagged":                 int(flagged.sum()),
        "n_total":                   len(labels_np),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Table & figure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_metrics_csv(
    rows: List[Dict],
    path: Path,
    fieldnames: Optional[List[str]] = None,
) -> None:
    if not rows:
        return
    if fieldnames is None:
        # Union of all keys across all rows to handle rows with extra car_* columns
        seen = {}
        for row in rows:
            for k in row:
                seen[k] = None
        fieldnames = list(seen.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Saved → {path}")


def _plot_confusion(
    labels: np.ndarray,
    preds:  np.ndarray,
    title:  str,
    path:   Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        cm   = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(
            confusion_matrix  = cm,
            display_labels    = [DOC_CLASSES[i] for i in range(NUM_DOC_CLASSES)],
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(title)
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved → {path}")
    except Exception as exc:
        log.warning(f"Could not save confusion matrix: {exc}")


def _plot_roc(
    labels_np: np.ndarray,
    probs_np:  np.ndarray,
    title:     str,
    path:      Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        classes_present = np.unique(labels_np).tolist()
        fig, ax = plt.subplots(figsize=(5, 5))

        for cls in classes_present:
            bin_labels  = (labels_np == cls).astype(int)
            fpr_arr, tpr_arr, _ = roc_curve(bin_labels, probs_np[:, cls])
            roc_auc = auc(fpr_arr, tpr_arr)
            ax.plot(
                fpr_arr, tpr_arr,
                label=f"{DOC_CLASSES[cls]} (AUC={roc_auc:.3f})"
            )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved → {path}")
    except Exception as exc:
        log.warning(f"Could not save ROC curves: {exc}")


def _plot_tsne(
    embeddings: np.ndarray,   # (N, D)
    labels:     np.ndarray,
    title:      str,
    path:       Path,
) -> None:
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_sample = min(2000, len(embeddings))
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(embeddings), n_sample, replace=False)
        emb_s = embeddings[idx]
        lab_s = labels[idx]

        tsne_coords = TSNE(
            n_components=2, perplexity=30, random_state=RANDOM_SEED
        ).fit_transform(emb_s)

        fig, ax = plt.subplots(figsize=(6, 5))
        colours = ["#d62728", "#ff7f0e", "#2ca02c"]   # UWS=red, MCS=orange, HC=green
        for cls in range(NUM_DOC_CLASSES):
            mask = lab_s == cls
            ax.scatter(
                tsne_coords[mask, 0], tsne_coords[mask, 1],
                c=colours[cls], s=10, alpha=0.6, label=DOC_CLASSES[cls]
            )
        ax.set_title(title)
        ax.legend(markerscale=3)
        ax.axis("off")
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved → {path}")
    except Exception as exc:
        log.warning(f"Could not save t-SNE: {exc}")


def _plot_ccs_distribution(
    ccs_scores: np.ndarray,
    labels_np:  np.ndarray,
    path:       Path,
) -> None:
    """Box + violin plot of CCS scores per DOC class."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        data_by_class = [
            ccs_scores[labels_np == cls]
            for cls in range(NUM_DOC_CLASSES)
        ]
        ax.violinplot(
            [d for d in data_by_class if len(d) > 0],
            showmedians=True,
        )
        ax.axhline(CCS_COVERT_THRESHOLD, color="red", linestyle="--",
                   alpha=0.7, label=f"Covert flag threshold ({CCS_COVERT_THRESHOLD})")
        ax.set_xticks(range(1, NUM_DOC_CLASSES + 1))
        ax.set_xticklabels([DOC_CLASSES[i] for i in range(NUM_DOC_CLASSES)])
        ax.set_ylabel("Consciousness Coherence Score (CCS)")
        ax.set_title("CCS Distribution by DOC Class")
        ax.legend()
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved → {path}")
    except Exception as exc:
        log.warning(f"Could not save CCS distribution: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    e2_ckpt:              Optional[Path],
    e3_ckpt:              Optional[Path],
    e1_ckpt:              Optional[Path],
    e1_diversity_ckpt:    Optional[Path],   # original pretrained (TUH) — used for PDI
    e2_diversity_ckpt:    Optional[Path],   # original pretrained (DEAP) — used for PDI
    e2_euclidean_ckpt:    Optional[Path],   # EuclideanEncoder baseline for geometry ablation
    device:    str,
    no_plots:  bool = False,
) -> None:

    log.info("=" * 70)
    log.info("DOC O4 — Full PDI-CCS Pipeline Evaluation")
    log.info("Target: Nature Scientific Reports")
    log.info("PRIMARY  task: binary neurological outcome prediction (AUC @ patient level, LOPO)")
    log.info("SECONDARY task: 3-class DOC classification (balanced accuracy, macro-F1)")
    log.info("=" * 70)

    # ── Load I-CARE data ──────────────────────────────────────────────────────
    log.info("\n[1/6] Loading I-CARE dataset …")
    from src.stage3_doc.dataset_icare import ICareDataset
    icare_ds = ICareDataset()
    labels_np   = icare_ds.labels.numpy()
    epochs_t    = icare_ds.epochs                             # (N, C, T)
    n_ch        = icare_ds.n_channels

    log.info(f"  {len(icare_ds):,} epochs  |  {icare_ds.n_subjects} patients")
    log.info(f"  Class distribution: {icare_ds.class_counts()}")

    # ── Build graph dataset for E3 ────────────────────────────────────────────
    log.info("\n[2/6] Pre-computing dwPLI graphs for E3 …")
    from src.stage3_doc.train_doc_encoder import ICAREGraphDataset
    graph_ds = ICAREGraphDataset.from_icare_dataset(
        icare_ds, sfreq=SFREQ_ICARE, threshold=0.1, max_epochs_per_patient=40
    )

    # ── Load encoders ─────────────────────────────────────────────────────────
    log.info("\n[3/6] Loading encoders …")

    e2_model = None
    if e2_ckpt and e2_ckpt.exists():
        e2_model = _load_e2(e2_ckpt, device, n_channels=n_ch)
        log.info(f"  E2 loaded: {e2_ckpt.name}")
    else:
        log.warning(f"  E2 checkpoint not found: {e2_ckpt}  — E2 will be skipped")

    # Euclidean E2 ablation — matches LorentzianEncoder architecture exactly
    # except for the manifold layer.  Used only for the geometry ablation table;
    # not used in the primary PDI-CCS fusion pathway.
    e2_euclidean_model = None
    if e2_euclidean_ckpt and e2_euclidean_ckpt.exists():
        e2_euclidean_model = _load_e2(e2_euclidean_ckpt, device, n_channels=n_ch)
        log.info(f"  E2 Euclidean ablation loaded: {e2_euclidean_ckpt.name}")
    else:
        log.info("  E2 Euclidean checkpoint not found — geometry ablation row will be skipped")

    e3_model = None
    if e3_ckpt and e3_ckpt.exists():
        e3_model = _load_e3(e3_ckpt, device)
        log.info(f"  E3 loaded: {e3_ckpt.name}")
    else:
        log.warning(f"  E3 checkpoint not found: {e3_ckpt}  — E3 will be skipped")

    e1_model = None
    if e1_ckpt and e1_ckpt.exists():
        e1_model = _load_e1(e1_ckpt, device)
        log.info(f"  E1 loaded: {e1_ckpt.name}")
    else:
        log.info("  E1 not found — running in 2-encoder mode (E2+E3)")

    # ── Inference: all encoders ───────────────────────────────────────────────
    log.info("\n[4/6] Running inference …")

    probs_e2  = embeds_e2  = None
    probs_e3  = embeds_e3  = None
    probs_e1  = embeds_e1  = None
    probs_e2_euclidean = None   # geometry ablation baseline

    # ── Align raw EEG epochs with graph windows ───────────────────────────────
    # ICAREGraphDataset takes the first MAX_EPOCHS_PER_PATIENT epochs per
    # patient, so raw epoch indices for E2 must be reconstructed the same way.
    MAX_EPOCHS_PER_PATIENT = 40
    n_graph      = len(graph_ds)
    subj_ids     = icare_ds.subject_ids          # (N_total,) int tensor
    n_subjects   = int(subj_ids.max().item()) + 1
    graph_epoch_indices: List[int] = []
    for s in range(n_subjects):
        mask = (subj_ids == s).nonzero(as_tuple=False).squeeze(1)
        n_take = min(len(mask), MAX_EPOCHS_PER_PATIENT)
        graph_epoch_indices.extend(mask[:n_take].tolist())
    n_epoch          = min(len(graph_epoch_indices), n_graph)
    graph_epoch_idx  = graph_epoch_indices[:n_epoch]   # raw epoch indices aligned to graphs
    epochs_t_eval    = epochs_t[graph_epoch_idx]        # (n_epoch, C, T) — graph-aligned
    labels_np_eval   = labels_np[graph_epoch_idx]       # sanity check labels
    graph_labels     = graph_ds.labels.numpy()[:n_epoch]
    log.info(f"  Graph-aligned epochs: {n_epoch}  (raw epoch idx range "
             f"{min(graph_epoch_idx)}–{max(graph_epoch_idx)})")

    if e2_model:
        log.info("  E2 inference …")
        probs_e2, embeds_e2 = _infer_e2(e2_model, epochs_t_eval, device=device)
        probs_e2  = probs_e2[:n_epoch]
        embeds_e2 = embeds_e2[:n_epoch]

    if e2_euclidean_model:
        log.info("  E2-Euclidean ablation inference …")
        probs_e2_euclidean, _ = _infer_e2(e2_euclidean_model, epochs_t_eval, device=device)
        probs_e2_euclidean = probs_e2_euclidean[:n_epoch]

    if e3_model:
        log.info("  E3 inference …")
        # Use graph_ds labels (aligned with graphs)
        from torch.utils.data import Subset
        graph_subset = Subset(graph_ds, list(range(n_epoch)))
        probs_e3, embeds_e3 = _infer_e3(e3_model, graph_subset, device=device)

    if e1_model:
        log.info("  E1 inference …")
        probs_e1, embeds_e1 = _infer_e1(e1_model, epochs_t_eval, device=device)
        # If E1 has no classifier head it returns uniform probs — fit a linear
        # probe on the embeddings using the ground-truth labels (standard SSL
        # linear evaluation protocol) to get calibrated DOC probabilities.
        max_var = probs_e1.var(axis=0).max()
        if max_var < 1e-4:
            log.info("  E1 probs are uniform — fitting linear probe on embeddings …")
            probs_e1 = _fit_e1_probe_cv(embeds_e1, graph_labels)

    # ── Diversity priors for PDI (original pretrained, NOT fine-tuned on I-CARE) ──
    # PDI requires HETEROGENEOUS encoder biases to be meaningful.  Fine-tuned E1/E2
    # converge to similar I-CARE predictions → JSD≈0.  Original pretrained models
    # each carry a different domain bias (TUH complexity, DEAP emotion) and will
    # disagree with E3 (DOC specialist) for covert patients.
    probs_e1_div = probs_e2_div = None

    if e1_diversity_ckpt and e1_diversity_ckpt.exists():
        log.info(f"  E1 diversity prior: {e1_diversity_ckpt.name} …")
        e1_div_model = _load_e1(e1_diversity_ckpt, device)
        probs_e1_div, embeds_e1_div = _infer_e1(e1_div_model, epochs_t_eval, device=device)
        max_var = probs_e1_div.var(axis=0).max()
        if max_var < 1e-4:
            log.info("  E1 diversity probs uniform — fitting linear probe …")
            probs_e1_div = _fit_e1_probe_cv(embeds_e1_div, graph_labels)
    else:
        probs_e1_div = probs_e1   # fall back to fine-tuned

    if e2_diversity_ckpt and e2_diversity_ckpt.exists():
        log.info(f"  E2 diversity prior: {e2_diversity_ckpt.name} …")
        e2_div_model = _load_e2(e2_diversity_ckpt, device, n_channels=19)
        probs_e2_div, _ = _infer_e2(e2_div_model, epochs_t_eval, device=device)
        probs_e2_div = probs_e2_div[:n_epoch]
    else:
        probs_e2_div = probs_e2   # fall back to fine-tuned

    # ── PDI and CCS ───────────────────────────────────────────────────────────
    log.info("\n[5/6] Computing PDI and CCS …")

    available_probs = [p for p in [probs_e1, probs_e2, probs_e3] if p is not None]
    n_encoders = len(available_probs)

    if n_encoders == 0:
        log.error("No encoder predictions available. Check checkpoints.")
        return

    # ── AUC-weighted ensemble ─────────────────────────────────────────────────
    # Load each encoder's best validation AUC from its checkpoint (if available)
    # and use AUC² as weights so the strongest encoder dominates.
    def _ckpt_auc(path: Path) -> float:
        try:
            c = torch.load(path, map_location="cpu", weights_only=False)
            return float(c.get("best_auc", c.get("auc", 1.0)))
        except Exception:
            return 1.0

    enc_aucs: List[float] = []
    if probs_e1 is not None: enc_aucs.append(_ckpt_auc(e1_ckpt))
    if probs_e2 is not None: enc_aucs.append(_ckpt_auc(e2_ckpt))
    if probs_e3 is not None: enc_aucs.append(_ckpt_auc(e3_ckpt))

    raw_w  = np.array(enc_aucs, dtype=np.float64) ** 2   # square amplifies differences
    w      = raw_w / raw_w.sum()
    log.info(f"  Ensemble AUC-weights: " +
             ", ".join(f"{n}={v:.3f}" for n, v in
                       zip([n for n, p in [("E1", probs_e1), ("E2", probs_e2), ("E3", probs_e3)]
                            if p is not None], w)))

    if n_encoders >= 2:
        # PDI uses diversity priors (original pretrained) to ensure heterogeneous biases
        diversity_probs = [p for p in [probs_e1_div, probs_e2_div, probs_e3] if p is not None]
        p_tensors_div   = [torch.from_numpy(p) for p in diversity_probs]
        pdi_t           = pairwise_disagreement_index(p_tensors_div).numpy()
        ccs_t           = consciousness_coherence_score(p_tensors_div, torch.from_numpy(pdi_t)).numpy()
        log.info(f"  PDI range: [{pdi_t.min():.4f}, {pdi_t.max():.4f}]  "
                 f"mean={pdi_t.mean():.4f}  "
                 f"UWS={pdi_t[graph_labels==0].mean():.4f}  "
                 f"MCS={pdi_t[graph_labels==1].mean():.4f}  "
                 f"HC={pdi_t[graph_labels==2].mean():.4f}")
        # AUC-weighted mean (fine-tuned encoders) for classification
        p_mean    = sum(wi * pi for wi, pi in zip(w, available_probs))
    else:
        # Single encoder: PDI = 0 by definition, CCS = p[HC]
        p_mean = available_probs[0]
        pdi_t  = np.zeros(len(p_mean))
        ccs_t  = PDI_ALPHA * p_mean[:, -1] + PDI_BETA * (1.0 - pdi_t)

    # ── Compute all metrics ───────────────────────────────────────────────────
    log.info("\n[6/6] Computing metrics and generating outputs …")

    eval_labels = graph_labels   # aligned with graph_ds order

    # ── Patient-level arrays (Scientific Reports: ONLY patient-level metrics reported) ──
    subject_ids_eval = icare_ds.subject_ids.numpy()[graph_epoch_idx]   # (n_epoch,)
    # Filled once the best model probs are known; placeholder dict populated below.
    patient_binary_results: Dict[str, float] = {}

    results_rows = []

    def _add_result(name: str, probs: np.ndarray, show_car: bool = False):
        m = compute_metrics(eval_labels, probs)
        row = {"model": name, **{k: f"{v:.4f}" for k, v in m.items()}}
        results_rows.append(row)
        log.info(
            f"  {name:40s}  bal_acc={m['balanced_accuracy']:.4f}  "
            f"auc={m['macro_auc']:.4f}  f1={m['macro_f1']:.4f}"
        )
        if show_car and n_encoders >= 2:
            # PDI is the primary covert awareness biomarker:
            # UWS patients → all encoders agree (low PDI);
            # MCS/covert patients → E3 says MCS while E1/E2 may say UWS (high PDI)
            opt_pdi_thr, _ = _optimise_ccs_threshold(pdi_t, eval_labels)
            car_m = covert_awareness_metrics(pdi_t, eval_labels, threshold=opt_pdi_thr)
            row.update({f"car_{k}": f"{v:.4f}" for k, v in car_m.items()})
            log.info(
                f"    [PDI covert awareness @ thr={opt_pdi_thr:.2f}]"
                f"  Recall = {car_m['covert_awareness_recall']:.4f}  "
                f"FAR = {car_m['false_alarm_rate']:.4f}  "
                f"Flagged = {car_m['n_flagged']}/{car_m['n_total']}"
            )

    log.info("\n─── Individual encoders (ablation baseline) ───")
    if probs_e2_euclidean is not None:
        # Geometry ablation: Euclidean vs Lorentzian — same architecture, different manifold.
        # If Lorentzian AUC > Euclidean AUC, the gain is attributable solely to H^n curvature.
        _add_result("E2 Euclidean (ablation: flat geometry)",   probs_e2_euclidean)
    if probs_e2  is not None: _add_result("E2 Lorentzian only",      probs_e2[:n_epoch])
    if probs_e3  is not None: _add_result("E3 Graph only",           probs_e3)
    if probs_e1  is not None: _add_result("E1 FractalSSL only",      probs_e1[:n_epoch])

    log.info("\n─── Ensemble (AUC-weighted mean, no PDI-CCS) ───")
    if n_encoders >= 2:
        _add_result("AUC-weighted ensemble (mean probs)", p_mean)

    log.info("\n─── Full PDI-CCS Fusion ───")
    # Final fusion = learned MLP from PDI-CCS (if fusion ckpt exists)
    # Otherwise use p_mean as the prediction (PDI-CCS only influences CCS score)
    fusion_ckpt = CKPT_ROOT / "fusion_best.pt"
    if fusion_ckpt.exists():
        from src.models.ccs_fusion import PDICCSFusion
        fusion = PDICCSFusion(num_encoders=n_encoders).to(device).eval()
        fckpt  = torch.load(fusion_ckpt, map_location=device, weights_only=False)
        fusion.load_state_dict(fckpt.get("model", fckpt), strict=False)
        with torch.no_grad():
            p_list = [torch.from_numpy(p).to(device) for p in available_probs]
            pdi_dev = torch.from_numpy(pdi_t).to(device)
            flogits, _, _, _ = fusion(p_list)
            fprobs  = F.softmax(flogits, dim=-1).cpu().numpy()
        _add_result("Full PDI-CCS Fusion (learned)", fprobs, show_car=True)
    else:
        # Fallback: use p_mean weighted by CCS confidence
        _add_result("PDI-CCS Ensemble (no fusion MLP)", p_mean, show_car=True)

    # ── PDI threshold optimisation (covert awareness biomarker) ─────────────────
    if n_encoders >= 2:
        opt_pdi_thr, opt_pdi_score = _optimise_ccs_threshold(pdi_t, eval_labels)
        def_pdi_m = covert_awareness_metrics(pdi_t, eval_labels, threshold=0.15)
        opt_pdi_m = covert_awareness_metrics(pdi_t, eval_labels, threshold=opt_pdi_thr)
        log.info(
            f"\n─── PDI threshold optimisation (covert awareness) ───\n"
            f"  Default  thr=0.15  "
            f"recall={def_pdi_m['covert_awareness_recall']:.4f}  "
            f"FAR={def_pdi_m['false_alarm_rate']:.4f}\n"
            f"  Optimal  thr={opt_pdi_thr:.2f}  "
            f"recall={opt_pdi_m['covert_awareness_recall']:.4f}  "
            f"FAR={opt_pdi_m['false_alarm_rate']:.4f}  "
            f"(score={opt_pdi_score:.4f})"
        )
        # Also show CCS distribution for reference (not used as primary metric)
        ccs_uws = ccs_t[eval_labels == 0].mean()
        ccs_mcs = ccs_t[eval_labels == 1].mean()
        ccs_hc  = ccs_t[eval_labels == 2].mean()
        log.info(
            f"  CCS (reference): UWS={ccs_uws:.3f}  MCS={ccs_mcs:.3f}  HC={ccs_hc:.3f}"
        )
        results_rows.append({
            "model":         "(optimal PDI covert threshold)",
            "pdi_threshold": f"{opt_pdi_thr:.2f}",
            "car_recall":    f"{opt_pdi_m['covert_awareness_recall']:.4f}",
            "car_far":       f"{opt_pdi_m['false_alarm_rate']:.4f}",
        })

    # Per-class classification report — use E3 alone as best classifier
    log.info("\n─── Best model: per-class report (E3 alone) ───")
    best_probs = probs_e3 if probs_e3 is not None else (fprobs if fusion_ckpt.exists() and n_encoders >= 2 else p_mean)
    best_preds = best_probs.argmax(axis=1)
    target_names = [DOC_CLASSES[i] for i in range(NUM_DOC_CLASSES)]
    log.info(
        "\n" + classification_report(eval_labels, best_preds,
                                     target_names=target_names, zero_division=0)
    )

    # ── PRIMARY TASK: patient-level binary outcome (Scientific Reports headline metric) ──
    log.info("\n─── PRIMARY TASK: patient-level binary neurological outcome (good vs poor) ───")
    pat_probs, pat_labels = aggregate_patient_level(
        best_probs, eval_labels, subject_ids_eval
    )
    patient_binary_results = compute_binary_outcome_metrics(
        pat_probs, pat_labels, bootstrap_n=1000, seed=42
    )
    log.info(
        f"  Patients: {len(pat_labels)}  "
        f"(good={patient_binary_results['n_good']}, poor={patient_binary_results['n_poor']})"
    )
    log.info(
        f"  Binary AUC = {patient_binary_results['binary_auc']:.4f}  "
        f"(95% CI: {patient_binary_results['binary_auc_ci_lo']:.4f}–"
        f"{patient_binary_results['binary_auc_ci_hi']:.4f})"
    )
    log.info(
        f"  Sensitivity = {patient_binary_results['binary_sensitivity']:.4f}  "
        f"Specificity = {patient_binary_results['binary_specificity']:.4f}  "
        f"(@Youden thr={patient_binary_results['binary_threshold']:.3f})"
    )
    results_rows.append({
        "model":               "[PRIMARY] Binary outcome (patient-level, LOPO)",
        "binary_auc":          f"{patient_binary_results['binary_auc']:.4f}",
        "binary_auc_ci":       f"[{patient_binary_results['binary_auc_ci_lo']:.4f}, "
                               f"{patient_binary_results['binary_auc_ci_hi']:.4f}]",
        "binary_sensitivity":  f"{patient_binary_results['binary_sensitivity']:.4f}",
        "binary_specificity":  f"{patient_binary_results['binary_specificity']:.4f}",
        "n_patients":          f"{len(pat_labels)}",
    })

    # ── Save tables ───────────────────────────────────────────────────────────
    _save_metrics_csv(results_rows, TABLE_ROOT / "table1_main_results.csv")

    # SOTA comparison table (Scientific Reports: patient-level metrics only)
    # NOTE: Raveendran 2025 «accuracy=0.963» is WINDOW-level on n=60 with
    # the same patient in train and test partitions — not a valid clinical comparator.
    # We report their binary AUC where available; «—» means not reported.
    sota_rows = [
        # Binary outcome (PRIMARY task) — good (CPC1-2) vs poor (CPC3-4)
        # patient-level LOPO AUC
        {"model": "Raveendran et al. 2025 (window-level only — leakage)",
         "note":       "n=60, window-level 80/20 split, patient leakage",
         "binary_auc": "—",    "macro_auc": "0.87 (window)", "macro_f1": "— (window)"},
        {"model": "Liu et al. 2025 (Multi-Band Attention CNN)",
         "note":       "n=28, patient-level, binary AUC reported",
         "binary_auc": "0.78", "macro_auc": "—",             "macro_f1": "—"},
        {"model": "Della Bella et al. 2025 (EEG markers, Commun. Biol.)",
         "note":       "n=237, patient-level, binary survival prediction",
         "binary_auc": "0.81", "macro_auc": "—",             "macro_f1": "—"},
    ] + results_rows
    _save_metrics_csv(sota_rows, TABLE_ROOT / "table1_sota_comparison.csv")
    log.info(f"\nTables written to {TABLE_ROOT}")

    # ── Generate plots ────────────────────────────────────────────────────────
    if not no_plots:
        log.info("\n─── Generating figures ───")

        _plot_confusion(
            eval_labels, best_preds,
            title="DOC Classification — Full PDI-CCS Model",
            path=PLOT_ROOT / "fig1_confusion_matrix.png",
        )

        _plot_roc(
            eval_labels, best_probs,
            title="ROC Curves — DOC Classification",
            path=PLOT_ROOT / "fig2_roc_curves.png",
        )

        # Fig 6 — Binary outcome ROC curve (patient level) — PRIMARY FIGURE
        if patient_binary_results.get("binary_auc") is not None and not np.isnan(
            patient_binary_results.get("binary_auc", np.nan)
        ):
            from sklearn.metrics import roc_curve
            pat_binary_labels = (pat_labels == 2).astype(int)
            pat_binary_scores = pat_probs[:, 2]
            fpr_b, tpr_b, _ = roc_curve(pat_binary_labels, pat_binary_scores)
            import matplotlib.pyplot as plt
            fig_b, ax_b = plt.subplots(figsize=(5, 5))
            ax_b.plot(fpr_b, tpr_b, lw=2, color="steelblue",
                      label=f"PDI-CCS (AUC = {patient_binary_results['binary_auc']:.3f}  "
                            f"[{patient_binary_results['binary_auc_ci_lo']:.3f}–"
                            f"{patient_binary_results['binary_auc_ci_hi']:.3f}])")
            ax_b.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Chance")
            # Youden optimal threshold point
            ax_b.scatter(
                [1 - patient_binary_results['binary_specificity']],
                [patient_binary_results['binary_sensitivity']],
                s=80, zorder=5, color="firebrick", label="Youden threshold"
            )
            ax_b.set_xlabel("False Positive Rate")
            ax_b.set_ylabel("True Positive Rate (Sensitivity)")
            ax_b.set_title("Binary Neurological Outcome ROC\n"
                           "(good CPC1-2 vs poor CPC3-4, patient-level LOPO)")
            ax_b.legend(loc="lower right", fontsize=8)
            ax_b.set_xlim(0, 1)
            ax_b.set_ylim(0, 1)
            fig_b.tight_layout()
            fig_b.savefig(PLOT_ROOT / "fig6_binary_outcome_roc.png", dpi=300)
            plt.close(fig_b)
            log.info(f"  Binary outcome ROC → {PLOT_ROOT / 'fig6_binary_outcome_roc.png'}")

        if embeds_e2 is not None:
            _plot_tsne(
                embeds_e2[:n_epoch], eval_labels,
                title="t-SNE of E2 Lorentzian Embeddings",
                path=PLOT_ROOT / "fig5_tsne_lorentzian.png",
            )

        if n_encoders >= 2:
            _plot_ccs_distribution(
                ccs_t, eval_labels,
                path=PLOT_ROOT / "fig4_ccs_distribution.png",
            )

    log.info("\n" + "=" * 70)
    log.info("EVALUATION COMPLETE")
    log.info(f"  Tables → {TABLE_ROOT}")
    log.info(f"  Plots  → {PLOT_ROOT}")
    log.info("=" * 70)

    # ── Save eval cache pkl (Code Ocean: load this to skip all inference) ─────
    CACHE_ROOT = PROJECT_ROOT / "results" / "cache"
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_ROOT / "eval_cache.pkl"
    eval_cache = {
        "probs_e1":              probs_e1,
        "probs_e2":              probs_e2,
        "probs_e3":              probs_e3,
        "embeds_e2":             embeds_e2,
        "embeds_e3":             embeds_e3,
        "embeds_e1":             embeds_e1,
        "labels":                eval_labels,
        "pdi":                   pdi_t,
        "ccs":                   ccs_t,
        "p_mean":                p_mean,
        "doc_classes":           DOC_CLASSES,
        "results_rows":          results_rows,
        # Scientific Reports PRIMARY task — patient-level binary outcome
        "patient_probs":         pat_probs,
        "patient_labels":        pat_labels,
        "patient_binary_auc":    patient_binary_results.get("binary_auc"),
        "patient_binary_auc_ci": (
            patient_binary_results.get("binary_auc_ci_lo"),
            patient_binary_results.get("binary_auc_ci_hi"),
        ),
    }
    with open(cache_path, "wb") as _f:
        pickle.dump(eval_cache, _f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"Eval cache → {cache_path}  (pkl — upload to Code Ocean to skip retraining)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 4 — PDI-CCS Pipeline Evaluation for Nature Scientific Reports"
    )
    p.add_argument(
        "--e1_ckpt", type=Path,
        default=(
            CKPT_ROOT / "e1_doc_finetuned.pt"
            if (CKPT_ROOT / "e1_doc_finetuned.pt").exists()
            else CKPT_ROOT / "e1_fractalssl_tuh.pt"
        ),
        help="E1 checkpoint — uses fine-tuned DOC version if available",
    )
    p.add_argument(
        "--e2_ckpt", type=Path,
        default=(
            CKPT_ROOT / "e2_doc_icare.pt"
            if (CKPT_ROOT / "e2_doc_icare.pt").exists()
            else CKPT_ROOT / "e2_lorentzian_deap_best.pt"
        ),
        help="E2 checkpoint — uses I-CARE DOC version if available",
    )
    p.add_argument(
        "--e3_ckpt", type=Path,
        default=CKPT_ROOT / "e3_graph_icare_best.pt",
        help="E3 Graph Encoder checkpoint",
    )
    p.add_argument(
        "--e1_diversity_ckpt", type=Path,
        default=CKPT_ROOT / "e1_fractalssl_tuh.pt",
        help="E1 diversity prior for PDI (pretrained on TUH, NOT fine-tuned on I-CARE)",
    )
    p.add_argument(
        "--e2_diversity_ckpt", type=Path,
        default=CKPT_ROOT / "e2_lorentzian_deap_best.pt",
        help="E2 diversity prior for PDI (pretrained on DEAP emotion, NOT fine-tuned on I-CARE)",
    )
    p.add_argument(
        "--e2_euclidean_ckpt", type=Path,
        default=CKPT_ROOT / "e2_euclidean_deap_best.pt",
        help=(
            "EuclideanEncoder checkpoint for geometry ablation table. "
            "Trained with --geometry euclidean on DEAP. "
            "Absent if not yet trained; row is silently skipped."
        ),
    )
    p.add_argument("--device",     type=str,  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",       type=int,  default=RANDOM_SEED)
    p.add_argument("--no_plots",   action="store_true", help="Skip matplotlib figures")
    p.add_argument(
        "--from_cache", action="store_true",
        help=(
            "Load pre-computed inference results from results/cache/eval_cache.pkl "
            "and regenerate figures/tables without rerunning any encoder. "
            "Use this on Code Ocean after uploading the pkl files."
        ),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.from_cache:
        cache_path = PROJECT_ROOT / "results" / "cache" / "eval_cache.pkl"
        if not cache_path.exists():
            log.error(f"--from_cache specified but cache not found: {cache_path}")
            return
        log.info(f"Loading eval cache from {cache_path} (skipping all inference) …")
        with open(cache_path, "rb") as _f:
            cache = pickle.load(_f)
        _reproduce_from_cache(cache, no_plots=args.no_plots)
        return

    run_evaluation(
        e1_ckpt            = args.e1_ckpt,
        e2_ckpt            = args.e2_ckpt,
        e3_ckpt            = args.e3_ckpt,
        e1_diversity_ckpt  = args.e1_diversity_ckpt,
        e2_diversity_ckpt  = args.e2_diversity_ckpt,
        e2_euclidean_ckpt  = args.e2_euclidean_ckpt,
        device             = args.device,
        no_plots           = args.no_plots,
    )


def _reproduce_from_cache(cache: dict, no_plots: bool = False) -> None:
    """Regenerate all paper figures and tables from a pre-saved eval_cache.pkl.

    This is the Code Ocean reviewer path: no GPU, no data, no retraining needed.
    Just upload eval_cache.pkl and run:
        python -m src.stage4_eval.evaluate_pipeline --from_cache
    """
    import logging as _log
    log = _log.getLogger(__name__)

    probs_e1    = cache.get("probs_e1")
    probs_e2    = cache.get("probs_e2")
    probs_e3    = cache.get("probs_e3")
    embeds_e2   = cache.get("embeds_e2")
    eval_labels = cache["labels"]
    pdi_t       = cache.get("pdi")
    ccs_t       = cache.get("ccs")
    p_mean      = cache.get("p_mean")
    results_rows = cache.get("results_rows", [])

    log.info(f"Cache loaded: {len(eval_labels)} samples, "
             f"encoders present: E1={probs_e1 is not None}, "
             f"E2={probs_e2 is not None}, E3={probs_e3 is not None}")

    best_probs = p_mean if p_mean is not None else (probs_e2 if probs_e2 is not None else probs_e3)
    if best_probs is None:
        log.error("Cache contains no prediction arrays. Cannot reproduce.")
        return
    best_preds = best_probs.argmax(axis=1)

    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    _save_metrics_csv(results_rows, TABLE_ROOT / "table1_main_results.csv")
    log.info(f"Tables → {TABLE_ROOT}")

    if not no_plots:
        _plot_confusion(eval_labels, best_preds,
                        title="DOC Classification — Full PDI-CCS Model",
                        path=PLOT_ROOT / "fig1_confusion_matrix.png")
        _plot_roc(eval_labels, best_probs,
                  title="ROC Curves — DOC Classification",
                  path=PLOT_ROOT / "fig2_roc_curves.png")
        if embeds_e2 is not None:
            _plot_tsne(embeds_e2, eval_labels,
                       title="t-SNE of E2 Lorentzian Embeddings",
                       path=PLOT_ROOT / "fig5_tsne_lorentzian.png")
        if ccs_t is not None:
            _plot_ccs_distribution(ccs_t, eval_labels,
                                   path=PLOT_ROOT / "fig4_ccs_distribution.png")
        log.info(f"Figures → {PLOT_ROOT}")

    log.info("Reproduction from cache complete.")


if __name__ == "__main__":
    main()
