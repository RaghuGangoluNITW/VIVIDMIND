# Checkpoint Inventory

All checkpoints are PyTorch state-dicts saved with `torch.save(model.state_dict(), path)`.
Load with `model.load_state_dict(torch.load(path, map_location="cpu"))`.

## E1 — FractalSSL Encoder

| File | Stage | Description |
|------|-------|-------------|
| `e1_fractalssl_tuh.pt` | Stage 1 pre-training | E1 backbone after SimCLR self-supervised pre-training on TUH EEG (2,905 recordings, 100 epochs). No class labels used. |
| `e1_doc_finetuned.pt` | Stage 1 fine-tune | E1 fine-tuned on I-CARE with a 3-class linear head (CPC 1–2 / CPC 3 / CPC 4). 50 epochs, lr=1e-4. |

## E2 — Lorentzian Encoder

| File | Stage | Description |
|------|-------|-------------|
| `e2_lorentzian_deap_best.pt` | Stage 2 pre-training | E2 Lorentz encoder pre-trained on DEAP binary valence (best val epoch). |
| `e2_lorentzian_dreamer_best.pt` | Stage 2 pre-training | E2 Lorentz encoder pre-trained on DREAMER binary valence (best val epoch). |
| `e2_euclidean_deap_best.pt` | Ablation | Identical architecture with Euclidean geometry (Poincaré ball replaced by flat R^n). Used for the Lorentz vs. Euclidean ablation (ΔmacroAUC = +0.347). |
| `e2_doc_icare.pt` | Stage 2 fine-tune | E2 Lorentzian fine-tuned on I-CARE (3 classes). Random weight initialisation baseline. |
| `e2_doc_icare_from_deap.pt` | Stage 2 fine-tune | E2 Lorentzian fine-tuned on I-CARE starting from `e2_lorentzian_deap_best.pt`. Used in cross-domain transfer ablation (Table 2 in the paper). |

## E3 — Graph-GAT Encoder

| File | Stage | Description |
|------|-------|-------------|
| `e3_graph_icare_best.pt` | Stage 3 | Graph-GAT encoder trained on I-CARE dwPLI connectivity graphs (best LOPO fold). K=3 GAT layers, M=4 attention heads, z∈R^64. |

## Notes

- `.pkl` mirror files are **not** included in this release; use `.pt` files only.
- Checkpoints were trained with PyTorch ≥ 2.1 and geoopt ≥ 0.5. Loading with older versions is not guaranteed.
- All I-CARE checkpoints were produced under leave-one-patient-out (LOPO) protocol; no patient appears in both the training set of the saved checkpoint and any test fold.
