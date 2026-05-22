"""
Stage 1 — FractalSSL Pre-training on TUH EEG

Trains the FractalSSL encoder (E1) on unlabelled clinical EEG using
self-supervised contrastive learning with fractal augmentations.

The pre-trained backbone weights are saved as:
    results/checkpoints/e1_fractalssl_tuh.pt

These weights are then used to initialise the emotion encoder (E2) and
the DOC classifier (Stage 3) via transfer learning.

Usage:
    python -m src.stage1_pretrain.train_fractalssl
        [--recordings 500]   # number of TUH recordings to use
        [--epochs 200]
        [--batch_size 64]
        [--device cuda]
        [--resume]           # resume from checkpoint if it exists
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEVICE,
    STAGE2_BATCH_SIZE,
    STAGE2_LR,
    STAGE2_WEIGHT_DECAY,
    CKPT_ROOT,
    RANDOM_SEED,
)
from src.models.fractal_ssl import FractalSSL
from src.stage1_pretrain.dataset_tuh import get_tuh_loader, N_TUH_CHANNELS

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

CKPT_PATH = CKPT_ROOT / "e1_fractalssl_tuh.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_fractalssl(args) -> FractalSSL:
    torch.manual_seed(RANDOM_SEED)

    # ── Data ─────────────────────────────────────────────────────────────────
    log.info(f"Loading TUH EEG corpus (up to {args.recordings} recordings) …")
    loader, n_channels = get_tuh_loader(
        max_recordings = args.recordings,
        batch_size     = args.batch_size,
        num_workers    = 0,
    )
    log.info(f"  {len(loader.dataset):,} epochs — {len(loader)} batches/epoch")
    log.info(f"  Channels: {n_channels}  Device: {args.device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FractalSSL(n_channels=n_channels).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  FractalSSL parameters: {n_params:,}")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    if args.resume and CKPT_PATH.exists():
        state = torch.load(CKPT_PATH, map_location=args.device)
        model.load_state_dict(state["model"])
        start_epoch = state.get("epoch", 0) + 1
        log.info(f"  Resumed from epoch {start_epoch - 1}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = STAGE2_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max   = args.epochs,
        eta_min = 1e-6,
    )
    # Advance scheduler state if resuming
    for _ in range(start_epoch - 1):
        scheduler.step()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float("inf")
    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss    = 0.0
        n_batches     = 0

        for x, _ in loader:
            x    = x.to(args.device)
            optimiser.zero_grad()
            loss = model(x)           # NT-Xent on two fractal views
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:03d}/{args.epochs}  "
                f"loss={avg_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_payload = {
                "model":      model.state_dict(),
                "epoch":      epoch,
                "loss":       best_loss,
                "config": {
                    "n_channels":    n_channels,
                    "proj_dim":      128,
                    "stage":         "stage1_fractalssl",
                },
            }
            torch.save(ckpt_payload, CKPT_PATH)
            # also save as pkl for Code Ocean reproducibility
            import pickle
            pkl_path = CKPT_PATH.with_suffix(".pkl")
            with open(pkl_path, "wb") as _f:
                pickle.dump(ckpt_payload, _f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"Pre-training complete.  Best loss={best_loss:.4f}")
    log.info(f"Backbone weights saved → {CKPT_PATH}")
    log.info(f"PKL copy               → {CKPT_PATH.with_suffix('.pkl')}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1: FractalSSL pre-training on TUH EEG"
    )
    p.add_argument("--recordings", type=int,   default=500,
                   help="Max TUH recordings to load (default: 500)")
    p.add_argument("--epochs",     type=int,   default=200,
                   help="Pre-training epochs (default: 200)")
    p.add_argument("--batch_size", type=int,   default=STAGE2_BATCH_SIZE,
                   help=f"Batch size (default: {STAGE2_BATCH_SIZE})")
    p.add_argument("--lr",         type=float, default=STAGE2_LR,
                   help=f"Learning rate (default: {STAGE2_LR})")
    p.add_argument("--device",     default=DEVICE,
                   help=f"Compute device (default: {DEVICE})")
    p.add_argument("--resume",     action="store_true",
                   help="Resume from existing checkpoint if present")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log.info(f"Stage 1 FractalSSL pre-training — device: {args.device}")
    train_fractalssl(args)
