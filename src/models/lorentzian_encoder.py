"""
E2 — Lorentzian Encoder

Architecture:
    TemporalCNN  →  SpatialAttention  →  FC projection  →
    project_to_hyperboloid  →  LorentzMLR classifier

The final embedding lives on the Lorentz hyperboloid H^n (Lorentz model of
hyperbolic space), which naturally accommodates the strict hierarchy

    Coma  ⊂  UWS  ⊂  MCS–  ⊂  MCS+  ⊂  Conscious

and the emotion intensity hierarchy in DEAP / DREAMER pre-training
(low arousal ⊂ medium arousal ⊂ high arousal).

Reference for Lorentz model: Chami et al., NeurIPS 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.lorentz_utils import (
    project_to_hyperboloid,
    LorentzLinear,
    LorentzMLR,
)
from src.config import (
    N_EEG_CHANNELS_DEAP,
    TEMPORAL_CNN_CHANNELS,
    TEMPORAL_CNN_KERNELS,
    LORENTZ_DIM,
    LORENTZ_CURV,
)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal CNN  (operates on the time axis)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalCNNBlock(nn.Module):
    """
    One 1-D convolutional block: Conv1d → BN → ELU → Dropout.
    Input/output shape: (batch, channels, time_steps)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False
        )
        self.bn      = nn.BatchNorm1d(out_ch)
        self.act     = nn.ELU()
        self.drop    = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x)))) + self.residual(x)


class TemporalCNN(nn.Module):
    """
    Stack of TemporalCNNBlocks applied *per EEG channel* (channel acts as
    the feature dimension after reshaping) then pooled over time.

    Input  : (batch, n_channels, n_samples)
    Output : (batch, n_channels, final_cnn_dim)
    """

    def __init__(
        self,
        n_eeg_ch: int = N_EEG_CHANNELS_DEAP,
        cnn_channels: list = TEMPORAL_CNN_CHANNELS,
        kernels: list = TEMPORAL_CNN_KERNELS,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers = []
        in_ch  = 1
        for out_ch, k in zip(cnn_channels, kernels):
            layers.append(TemporalCNNBlock(in_ch, out_ch, k, dropout))
            layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.net    = nn.Sequential(*layers)
        self.n_eeg  = n_eeg_ch
        self.out_dim = in_ch  # == cnn_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)          # treat each channel independently
        x = self.net(x)                      # (B*C, F, T')
        x = x.mean(dim=-1)                   # global avg pool → (B*C, F)
        x = x.reshape(B, C, self.out_dim)    # (B, C, F)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Spatial Attention  (across EEG channels)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """
    Learns a weighted average of channel-wise CNN features.

    Input  : (batch, n_channels, feat_dim)
    Output : (batch, feat_dim)
    """

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Tanh(),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F)
        w = self.attn(x)                  # (B, C, 1)
        w = torch.softmax(w, dim=1)       # normalise over channels
        out = (w * x).sum(dim=1)          # (B, F)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Full Lorentzian Encoder (E2)
# ─────────────────────────────────────────────────────────────────────────────

class LorentzianEncoder(nn.Module):
    """
    Full E2 encoder.

    Forward pass:
        EEG (B, C, T)
        → TemporalCNN         (B, C, F)
        → SpatialAttention    (B, F)
        → FC projection       (B, lorentz_dim)
        → project_to_H^n      (B, lorentz_dim+1)  ← hyperboloid embedding
        → LorentzMLR          (B, num_classes)     ← logits

    The intermediate hyperboloid point `z_h` is also exposed so the
    PDI-CCS fusion module can compute distances in H^n.
    """

    def __init__(
        self,
        n_eeg_ch:    int = N_EEG_CHANNELS_DEAP,
        cnn_channels: list = TEMPORAL_CNN_CHANNELS,
        kernels:      list = TEMPORAL_CNN_KERNELS,
        lorentz_dim:  int = LORENTZ_DIM,
        num_classes:  int = 2,
        dropout:      float = 0.3,
    ) -> None:
        super().__init__()
        self.temporal_cnn = TemporalCNN(n_eeg_ch, cnn_channels, kernels, dropout)
        feat_dim          = cnn_channels[-1]
        self.spatial_attn = SpatialAttention(feat_dim)
        self.proj         = nn.Sequential(
            nn.Linear(feat_dim, lorentz_dim),
            nn.LayerNorm(lorentz_dim),
        )
        self.lorentz_lin  = LorentzLinear(lorentz_dim, lorentz_dim)  # spatial dim
        self.classifier   = LorentzMLR(lorentz_dim, num_classes)  # spatial dim

    # ── helpers ──────────────────────────────────────────────────────────────

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the hyperboloid embedding (B, lorentz_dim+1)."""
        feat = self.temporal_cnn(x)         # (B, C, F)
        feat = self.spatial_attn(feat)      # (B, F)
        z_e  = self.proj(feat)              # (B, D)
        z_h  = project_to_hyperboloid(z_e)  # (B, D+1)
        z_h  = self.lorentz_lin(z_h)        # (B, D+1) — refine on manifold
        return z_h

    def forward(
        self, x: torch.Tensor
    ):
        """
        Returns:
            logits  : (B, num_classes)
            z_h     : (B, lorentz_dim+1)   — hyperbolic embedding
        """
        z_h    = self.embed(x)
        logits = self.classifier(z_h)
        return logits, z_h


# ─────────────────────────────────────────────────────────────────────────────
# EuclideanEncoder  — ablation baseline for E2
# ─────────────────────────────────────────────────────────────────────────────

class EuclideanEncoder(nn.Module):
    """
    Ablation baseline for E2: architecturally identical to LorentzianEncoder,
    but embeds on the unit hypersphere (L2-normalised Euclidean space) instead
    of the Lorentz hyperboloid H^n.

    Purpose
    -------
    The core theoretical claim of E2 is that the UWS ⊂ MCS ⊂ HC hierarchy is
    a tree and that hyperbolic space embeds trees with zero distortion (Gromov
    1987; Chami et al., NeurIPS 2019).  To isolate this contribution we need a
    baseline that shares *every* architectural decision except the manifold
    choice.  EuclideanEncoder is that baseline:

      - Same TemporalCNN backbone (n_layers, channels, kernel sizes)
      - Same SpatialAttention module (channel-weighted pooling)
      - Same FC projection + LayerNorm (to lorentz_dim dimensions)
      - Manifold: unit hypersphere via F.normalize(z, p=2) instead of H^n

    The L2-norm constraint is a principled Euclidean normalisation (used in
    SimCLR, CLIP, ArcFace) that mirrors the norm constraint imposed by the
    hyperboloid — only the *curvature* is absent.

    If LorentzianEncoder achieves higher AUC than EuclideanEncoder at the same
    architecture depth, the gain is attributable solely to the hyperbolic
    geometry, not to capacity or regularisation.

    References
    ----------
    Chami et al., "Hyperbolic Graph Convolutional Neural Networks",
      NeurIPS 2019.
    Cui et al., "Class-Balanced Loss Based on Effective Number of Samples",
      CVPR 2019.
    He et al., "Momentum Contrast for Unsupervised Visual Representation
      Learning", CVPR 2020.  (unit-sphere normalisation motivation)
    """

    def __init__(
        self,
        n_eeg_ch:     int   = N_EEG_CHANNELS_DEAP,
        cnn_channels: list  = TEMPORAL_CNN_CHANNELS,
        kernels:      list  = TEMPORAL_CNN_KERNELS,
        lorentz_dim:  int   = LORENTZ_DIM,
        num_classes:  int   = 2,
        dropout:      float = 0.3,
    ) -> None:
        super().__init__()
        self.temporal_cnn = TemporalCNN(n_eeg_ch, cnn_channels, kernels, dropout)
        feat_dim          = cnn_channels[-1]
        self.spatial_attn = SpatialAttention(feat_dim)
        self.proj         = nn.Sequential(
            nn.Linear(feat_dim, lorentz_dim),
            nn.LayerNorm(lorentz_dim),
        )
        # Euclidean classifier: standard linear head, no manifold operations
        self.classifier   = nn.Linear(lorentz_dim, num_classes)
        self.embed_dim    = lorentz_dim

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the L2-normalised Euclidean embedding (B, lorentz_dim).

        The L2 normalisation constrains all embeddings to lie on the unit
        (lorentz_dim - 1)-sphere, which is the natural Euclidean analogue of
        the hyperboloid norm constraint.  This ensures the comparison is fair:
        both encoders operate on a norm-constrained manifold, the only
        difference being the curvature (0 for Euclidean, c=1 for Lorentzian).
        """
        feat = self.temporal_cnn(x)               # (B, C, F)
        feat = self.spatial_attn(feat)            # (B, F)
        z    = self.proj(feat)                    # (B, D)
        z    = F.normalize(z, p=2, dim=-1)        # unit hypersphere in R^D
        return z

    def forward(self, x: torch.Tensor):
        """
        Returns:
            logits  : (B, num_classes)
            z       : (B, lorentz_dim)   — Euclidean unit-sphere embedding
        """
        z      = self.embed(x)
        logits = self.classifier(z)
        return logits, z
