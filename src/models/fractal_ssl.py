"""
E1 — FractalSSL Encoder  (Stage 1 scaffold)

Pre-trains on large unlabelled clinical EEG (TUH corpus) using
self-supervised contrastive learning with fractal augmentations.

STATUS: Scaffold — data pending (TUH EEG access requested from NEDC).
        Architecture and augmentation code is complete and unit-testable
        on random data.

Novel augmentation rationale
────────────────────────────
EEG from DOC patients has altered fractal (self-similar) structure:
Higuchi FD drops in UWS vs conscious (Olejarczyk et al. 2022).
We generate positive pairs by applying *different* fractal transformations
that preserve pathological scale-invariance, forcing the encoder to learn
representations invariant to recording-device noise but sensitive to
neural fractal dimension — a clinically validated DOC biomarker.

Reference: FractalSSL builds on SimCLR (Chen et al. 2020) with
           Fractional Brownian Motion (fBm) augmentations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from src.config import (
    FRACTAL_DIM_RANGE,
    FRACTAL_PROJ_DIM,
    SSL_TEMPERATURE,
    TEMPORAL_CNN_CHANNELS,
    TEMPORAL_CNN_KERNELS,
    RANDOM_SEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fractal Augmentations
# ─────────────────────────────────────────────────────────────────────────────

def _fgn_factor(n: int, hurst: float) -> np.ndarray:
    """
    Fast approximate Fractional Gaussian Noise via spectral method.
    Used to create spectrally-matched noise for augmentation.
    """
    rng    = np.random.default_rng()
    f      = np.fft.rfftfreq(n)[1:]          # skip DC
    power  = f ** (-(2 * hurst + 1) / 2.0)
    phases = rng.uniform(0, 2 * np.pi, len(f))
    coeff  = power * np.exp(1j * phases)
    noise  = np.fft.irfft(np.concatenate([[0], coeff]), n=n)
    return noise.astype(np.float32)


class FractalAugment:
    """
    EEG fractal augmentation.

    Applies additive fractional Gaussian noise at a random Hurst exponent
    sampled from FRACTAL_DIM_RANGE.  The noise is channel-independent,
    scaled to 10% of the channel RMS amplitude so it does not destroy signal.
    """

    def __init__(
        self,
        hurst_range: Tuple[float, float] = FRACTAL_DIM_RANGE,
        noise_scale:  float = 0.1,
    ) -> None:
        self.hurst_range = hurst_range
        self.noise_scale = noise_scale

    def __call__(self, epoch: np.ndarray) -> np.ndarray:
        """epoch : (n_channels, n_samples)  → augmented epoch (same shape)"""
        hurst  = np.random.uniform(*self.hurst_range)
        n      = epoch.shape[-1]
        result = np.empty_like(epoch)
        for ch in range(epoch.shape[0]):
            rms   = np.sqrt(np.mean(epoch[ch] ** 2)) + 1e-8
            noise = _fgn_factor(n, hurst)
            result[ch] = epoch[ch] + self.noise_scale * rms * noise
        return result


class FractalAugmentTorch(nn.Module):
    """Differentiable wrapper — operates on (B, C, T) tensors."""

    def __init__(self, **kwargs):
        super().__init__()
        self._aug = FractalAugment(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply per sample in batch (numpy round-trip, not differentiable)
        x_np  = x.cpu().numpy()
        out   = np.stack([self._aug(xi) for xi in x_np], axis=0)
        return torch.from_numpy(out).to(x.device)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder backbone (shared with E2 but without Lorentzian head)
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.res(x)


class FractalSSLBackbone(nn.Module):
    """
    Shared temporal CNN backbone for FractalSSL.
    Input : (B, C, T)
    Output: (B, embed_dim)
    """

    def __init__(
        self,
        n_channels: int,
        cnn_channels: list = TEMPORAL_CNN_CHANNELS,
        kernels:      list = TEMPORAL_CNN_KERNELS,
        embed_dim:    int  = 128,
        dropout:      float = 0.3,
    ) -> None:
        super().__init__()
        layers = []
        in_ch  = n_channels
        for out_ch, k in zip(cnn_channels, kernels):
            layers.append(_ConvBlock1D(in_ch, out_ch, k, dropout))
            layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.conv   = nn.Sequential(*layers)
        self.proj   = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_ch, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


# ─────────────────────────────────────────────────────────────────────────────
# Projection head (SimCLR style)
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = FRACTAL_PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# NT-Xent contrastive loss
# ─────────────────────────────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float = SSL_TEMPERATURE) -> torch.Tensor:
    """
    Normalised Temperature-scaled Cross-Entropy loss (Chen et al. 2020).

    z1, z2 : (B, proj_dim)  — L2-normalised projection of two views
    """
    B     = z1.size(0)
    z     = torch.cat([z1, z2], dim=0)          # (2B, D)
    sim   = z @ z.T / temp                       # (2B, 2B)

    # Mask out diagonal (self-similarity)
    mask  = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim   = sim.masked_fill(mask, float("-inf"))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)
    loss   = F.cross_entropy(sim, labels)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Full FractalSSL model
# ─────────────────────────────────────────────────────────────────────────────

class FractalSSL(nn.Module):
    """
    Full FractalSSL model: backbone + projection head + augmentor.

    During pre-training, call forward(x) which:
      1. Creates two fractal-augmented views of x
      2. Encodes both with the shared backbone
      3. Projects with MLP head
      4. Returns NT-Xent loss

    During fine-tuning / transfer:
      - Call embed(x) to get (B, embed_dim) representations
      - Throw away the projection head
    """

    def __init__(
        self,
        n_channels: int,
        embed_dim:  int = 128,
        proj_dim:   int = FRACTAL_PROJ_DIM,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = FractalSSLBackbone(n_channels, embed_dim=embed_dim, dropout=dropout)
        self.proj_head = ProjectionHead(embed_dim, proj_dim)
        self.augment   = FractalAugmentTorch()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.augment(x)
        x2 = self.augment(x)
        z1 = self.proj_head(self.backbone(x1))
        z2 = self.proj_head(self.backbone(x2))
        return nt_xent_loss(z1, z2)


# ─────────────────────────────────────────────────────────────────────────────
# Scaffold: pre-training entry point (requires TUH data)
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(n_channels: int, tuh_loader, device: str, epochs: int = 200) -> FractalSSL:
    """
    Pre-train FractalSSL on TUH EEG.

    tuh_loader : DataLoader yielding (batch_eeg, _) tuples
                 batch_eeg shape: (B, n_channels, T)
    """
    model     = FractalSSL(n_channels).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    model.train()
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for x, _ in tuh_loader:
            x = x.to(device)
            optimiser.zero_grad()
            loss = model(x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        if ep % 10 == 0:
            print(f"[FractalSSL] Epoch {ep}/{epochs}  loss={epoch_loss/len(tuh_loader):.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    B, C, T = 8, 32, 512
    x = torch.randn(B, C, T)
    model = FractalSSL(n_channels=C)
    model.train()
    loss = model(x)
    print(f"FractalSSL NT-Xent loss (random data): {loss.item():.4f}")
    emb = model.embed(x)
    print(f"Embedding shape: {emb.shape}")  # (8, 128)
