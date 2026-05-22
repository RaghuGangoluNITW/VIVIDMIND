"""
PDI-CCS Fusion Module

Combines E1 (FractalSSL), E2 (LorentzianEncoder), E3 (GraphEncoder) to produce:

  1. CCS  — Consciousness Coherence Score  (final DOC classification probability)
  2. PDI  — Pairwise Disagreement Index    (covert awareness indicator)

Mathematical formulation
────────────────────────
Given K encoders with class-probability outputs p_1 ... p_K ∈ Δ^(C-1):

  Mean prediction:
      p̄ = (1/K) Σ_i p_i                                     (B, C)

  Pairwise Disagreement Index (Jensen–Shannon divergence averaged):
      PDI = (1 / C(K,2)) Σ_{i<j} JSD(p_i ‖ p_j)            (B,)

      JSD(p ‖ q) = ½ KL(p ‖ m) + ½ KL(q ‖ m),  m = (p+q)/2

  Consciousness Coherence Score:
      CCS = α · p̄[highest_class] + β · (1 – PDI)            (B,)

  Covert awareness flag (VS patients only):
      flag = (PDI > τ_PDI) AND (CRS-R label == VS)

Interpretation
──────────────
• A VS patient with HIGH PDI = encoders disagree.  Since E2 (emotion) or
  E3 (connectivity) may pick up residual awareness signals invisible to
  behavioural CRS-R assessment, high PDI in a labelled-VS patient is a
  clinically meaningful flag for covert awareness (Chennu 2014 vs-fMRI cases).

• CCS integrates mean confidence AND inter-encoder coherence.
  Fully unresponsive patients will have low p̄[conscious] AND low PDI
  (encoders consistently agree = UWS/Coma).

References
──────────
Chen et al. (2020). A Mathematical Framework for Transformer Circuits. arXiv.
Menon et al. (2023). Covert Consciousness — a review. Nature Reviews Neurology.
Chennu et al. (2014). PLOS Comp Bio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from src.config import (
    PDI_ALPHA,
    PDI_BETA,
    CCS_COVERT_THRESHOLD,
    NUM_DOC_CLASSES,
    LORENTZ_DIM,
    DEVICE,
)
from src.models.lorentzian_encoder import LorentzianEncoder
from src.utils.lorentz_utils import lorentz_dist


# ─────────────────────────────────────────────────────────────────────────────
# JSD + PDI
# ─────────────────────────────────────────────────────────────────────────────

def _eps_clamp(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return p.clamp(min=eps)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon divergence between two probability distributions.
    p, q : (B, C)  — probability rows (sum = 1)
    returns: (B,)
    """
    m    = 0.5 * (p + q)
    p, q, m = _eps_clamp(p), _eps_clamp(q), _eps_clamp(m)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def pairwise_disagreement_index(probs: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute PDI = mean JSD over all encoder pairs.

    probs : list of K tensors, each (B, C)
    returns: (B,)  PDI in [0, log(2)] — normalised to [0, 1]
    """
    K     = len(probs)
    assert K >= 2, "Need at least 2 encoders for PDI"
    n_pairs = 0
    total   = torch.zeros(probs[0].size(0), device=probs[0].device)

    for i in range(K):
        for j in range(i + 1, K):
            total   += js_divergence(probs[i], probs[j])
            n_pairs += 1

    # Normalise by log(2) (max JSD for binary) so PDI ∈ [0, 1]
    return (total / n_pairs) / (torch.log(torch.tensor(2.0, device=total.device)) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# CCS
# ─────────────────────────────────────────────────────────────────────────────

def consciousness_coherence_score(
    probs:       List[torch.Tensor],
    pdi:         torch.Tensor,
    alpha:       float = PDI_ALPHA,
    beta:        float = PDI_BETA,
    top_class:   int   = -1,           # index of highest-consciousness class; -1 = last
) -> torch.Tensor:
    """
    CCS = α · p̄[conscious] + β · (1 – PDI)

    probs    : list of K tensors (B, C)
    pdi      : (B,)
    top_class: which column is "most conscious" (default = last = HC in DOC_CLASSES)

    returns  : (B,)  CCS ∈ [0, 1]
    """
    p_bar         = torch.stack(probs, dim=0).mean(dim=0)   # (B, C)
    p_conscious   = p_bar[:, top_class]                     # (B,)
    ccs           = alpha * p_conscious + beta * (1.0 - pdi)
    return ccs.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperbolic distance-based fusion (optional, uses Lorentz embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def lorentz_ensemble_disagreement(
    embeddings: List[torch.Tensor],   # list of K tensors (B, D+1) on H^n
) -> torch.Tensor:
    """
    Compute mean pairwise Lorentzian geodesic distance between encoder embeddings.

    High distance → encoders map the same EEG to distant points on H^n
    → structural disagreement in hyperbolic representation space.

    Returns: (B,)  mean geodesic distance
    """
    K     = len(embeddings)
    n_pairs = 0
    total   = torch.zeros(embeddings[0].size(0), device=embeddings[0].device)

    for i in range(K):
        for j in range(i + 1, K):
            # embeddings[j] might come from a different dimension — handle gracefully
            if embeddings[i].shape == embeddings[j].shape:
                total   += lorentz_dist(embeddings[i], embeddings[j])
                n_pairs += 1

    return total / max(n_pairs, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Learnable fusion head
# ─────────────────────────────────────────────────────────────────────────────

class PDICCSFusion(nn.Module):
    """
    Learnable fusion combinator.

    Concatenates the softmax outputs of all available encoders + PDI value,
    passes through a small MLP to produce final DOC class logits AND CCS.

    Inputs
    ------
    probs_list   : list of K tensors (B, C)  — encoder probabilities
    embeddings   : optional list of K Lorentz embeddings (B, D+1) for H^n PDI

    Outputs
    -------
    logits       : (B, num_classes)  — final DOC prediction
    ccs          : (B,)              — Consciousness Coherence Score
    pdi          : (B,)              — PDI
    covert_flags : (B,) bool         — VS patients with high PDI
    """

    def __init__(
        self,
        num_encoders:  int = 3,
        num_classes:   int = NUM_DOC_CLASSES,
        alpha:         float = PDI_ALPHA,
        beta:          float = PDI_BETA,
    ) -> None:
        super().__init__()
        in_dim    = num_encoders * num_classes + 1  # +1 for PDI feature
        self.mlp  = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
        self.alpha = alpha
        self.beta  = beta

    def forward(
        self,
        probs_list:  List[torch.Tensor],
        embeddings:  Optional[List[torch.Tensor]] = None,
        doc_labels:  Optional[torch.Tensor]        = None,   # CRS-R labels if known
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # ── PDI ──────────────────────────────────────────────────────────────
        pdi = pairwise_disagreement_index(probs_list)           # (B,)

        # ── CCS ──────────────────────────────────────────────────────────────
        ccs = consciousness_coherence_score(
            probs_list, pdi, self.alpha, self.beta
        )                                                       # (B,)

        # ── Covert awareness flag ─────────────────────────────────────────────
        # vs_label = 0 in DOC_CLASSES {"VS":0, "MCS":1, "HC":2}
        if doc_labels is not None:
            vs_mask      = (doc_labels == 0)
            covert_flags = vs_mask & (ccs > CCS_COVERT_THRESHOLD)
        else:
            covert_flags = (ccs > CCS_COVERT_THRESHOLD)

        # ── Final logits (learnable fusion) ──────────────────────────────────
        cat_probs = torch.cat(probs_list, dim=-1)               # (B, K*C)
        feat      = torch.cat([cat_probs, pdi.unsqueeze(-1)], dim=-1)  # (B, K*C+1)
        logits    = self.mlp(feat)                              # (B, C)

        return logits, ccs, pdi, covert_flags


# ─────────────────────────────────────────────────────────────────────────────
# CCS-weighted loss (used when fine-tuning fusion on Chennu 2014)
# ─────────────────────────────────────────────────────────────────────────────

class CCSWeightedGCELoss(nn.Module):
    """
    GCE loss re-weighted by CCS confidence.
    Samples where CCS is high (clear cases) get higher weight.
    Samples where CCS is ambiguous get down-weighted (they may be covert).
    """

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q

    def forward(
        self,
        logits:  torch.Tensor,   # (B, C)
        targets: torch.Tensor,   # (B,)
        ccs:     torch.Tensor,   # (B,)  — confidence weights
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        p_y   = probs[torch.arange(len(targets)), targets]
        gce   = (1.0 - p_y ** self.q) / self.q
        # CCS as per-sample weight (scale to mean=1 to not change loss magnitude)
        w     = ccs / (ccs.mean() + 1e-8)
        return (w * gce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    B, C, K = 8, 3, 3

    # Simulate K encoder probability outputs
    probs = [torch.softmax(torch.randn(B, C), dim=-1) for _ in range(K)]

    # PDI
    pdi = pairwise_disagreement_index(probs)
    print(f"PDI range: [{pdi.min().item():.3f}, {pdi.max().item():.3f}]")

    # CCS
    ccs = consciousness_coherence_score(probs, pdi)
    print(f"CCS range: [{ccs.min().item():.3f}, {ccs.max().item():.3f}]")

    # Full fusion
    fusion = PDICCSFusion(num_encoders=K, num_classes=C)
    labels = torch.randint(0, C, (B,))
    logits, ccs, pdi, flags = fusion(probs, doc_labels=labels)
    print(f"Fusion logits: {logits.shape}")
    print(f"Covert flags : {flags.sum().item()} of {B} samples flagged")

    # CCS-weighted loss
    loss_fn = CCSWeightedGCELoss()
    loss    = loss_fn(logits, labels, ccs)
    print(f"CCS-GCE loss : {loss.item():.4f}")
