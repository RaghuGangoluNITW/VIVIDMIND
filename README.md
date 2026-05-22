# VIVIDMIND

**Visible Interpretable Vigilance Inference for Disordered Minds via Intelligent Neural Decoding**

Official implementation for the paper:

> Gangolu R., Kadambari K.V. (2025). *VIVIDMIND: Visible Interpretable Vigilance Inference for Disordered Minds via Intelligent Neural Decoding.* Scientific Reports (under review).

---

## Overview

VIVIDMIND is a three-stage multi-geometry EEG representation learning framework for binary neurological outcome prediction in cardiac-arrest survivors with disorders of consciousness (DOC). It jointly models:

- **E1 – FractalSSL**: Fractal temporal self-similarity (SimCLR pre-trained on TUH EEG)
- **E2 – Lorentzian**: Hierarchical DOC-state geometry on the Lorentz hyperboloid (pre-trained on DEAP/DREAMER emotion EEG)
- **E3 – Graph-GAT**: Functional connectivity breakdown via dwPLI + Graph Attention Networks

A **PDI-CCS fusion** module computes cross-encoder disagreement as a continuous covert-awareness biomarker.

### Key Results (I-CARE, n = 55, LOPO cross-validation)

| Metric | Value |
|--------|-------|
| Patient-level binary AUC | **0.8798** (95 % CI: 0.7748–0.9678) |
| Sensitivity (Youden-optimal) | **89.7 %** (35 / 39) |
| Specificity | **75.0 %** (12 / 16) |
| E2 Lorentzian vs. Euclidean gain | **+0.347 macro-AUC** |
| Covert Awareness Recall (CAR) | **98.8 %** at PDI threshold 0.11 |

---

## Repository Structure

```
vividmind_release/
├── README.md
├── requirements.txt
├── generate_all_figures.py      # Reproduce all paper figures
│
├── src/
│   ├── config.py                # All paths & hyperparameters  edit here for your machine
│   ├── models/
│   │   ├── fractal_ssl.py       # E1: FractalSSL backbone + SimCLR head
│   │   ├── lorentzian_encoder.py# E2: Lorentz hyperboloid encoder
│   │   ├── graph_encoder.py     # E3: Graph-GAT encoder
│   │   └── ccs_fusion.py        # PDI-CCS fusion module
│   ├── stage1_pretrain/
│   │   ├── dataset_tuh.py       # TUH EEG data loader
│   │   ├── train_fractalssl.py  # Stage 1: FractalSSL pre-training on TUH
│   │   └── finetune_e1_doc.py   # Stage 1b: E1 fine-tune on I-CARE
│   ├── stage2_emotion/
│   │   ├── dataset_deap.py      # DEAP data loader
│   │   ├── dataset_dreamer.py   # DREAMER data loader
│   │   ├── train_emotion_encoder.py  # Stage 2: E2 pre-training on DEAP/DREAMER
│   │   ├── finetune_e2_doc.py   # Stage 2b: E2 fine-tune on I-CARE
│   │   └── ablation_e2_transfer.py  # 5-fold transfer ablation (Table 2)
│   ├── stage3_doc/
│   │   ├── dataset_icare.py     # I-CARE data loader (PhysioNet)
│   │   ├── dataset_sleepedf.py  # SleepEDF loader (auxiliary)
│   │   └── train_doc_encoder.py # Stage 3: E3 Graph-GAT training on I-CARE
│   ├── stage4_eval/
│   │   └── evaluate_pipeline.py # Stage 4: LOPO evaluation, SOTA table, all metrics
│   └── utils/
│       ├── eeg_utils.py         # Preprocessing, dwPLI, graph construction
│       └── lorentz_utils.py     # Lorentz manifold ops (exp/log maps, Fréchet mean)
│
├── checkpoints/
│   ├── README.md                # Checkpoint inventory and loading instructions
│   ├── e1_fractalssl_tuh.pt     # E1 backbone after TUH SSL pre-training
│   ├── e1_doc_finetuned.pt      # E1 fine-tuned on I-CARE (3-class)
│   ├── e2_lorentzian_deap_best.pt    # E2 after DEAP pre-training
│   ├── e2_lorentzian_dreamer_best.pt # E2 after DREAMER pre-training
│   ├── e2_euclidean_deap_best.pt     # E2 Euclidean ablation checkpoint
│   ├── e2_doc_icare.pt          # E2 Lorentzian fine-tuned on I-CARE
│   ├── e2_doc_icare_from_deap.pt# E2 from DEAP init, fine-tuned on I-CARE
│   └── e3_graph_icare_best.pt   # E3 Graph-GAT trained on I-CARE
│
├── results/
│   ├── tables/
│   │   ├── table1_main_results.csv    # Primary outcome + ablation metrics
│   │   ├── table1_sota_comparison.csv # SOTA comparison data
│   │   └── ablation_e2_transfer.csv   # E2 transfer 5-fold results
│   └── logs/
│       ├── training_log.txt           # Full training run log
│       └── training_log_eval.txt      # Evaluation log with per-class counts
│
└── data/
    └── README.md                # Dataset access and directory layout instructions
```

---

## Prerequisites

### Hardware
- GPU with ≥ 8 GB VRAM (tested on NVIDIA RTX 3060)
- Evaluation only (no training): any machine with ≥ 16 GB RAM

### Python Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**Key dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥ 2.1.0 | Deep learning |
| torch-geometric | ≥ 2.4.0 | Graph-GAT (E3) |
| geoopt | ≥ 0.5.0 | Lorentz manifold ops (E2) |
| mne | ≥ 1.5.0 | EEG preprocessing |
| scikit-learn | ≥ 1.3.0 | Metrics, cross-validation |

> **Note on torch-geometric**: install the version matching your CUDA release. See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Data Access

**No EEG data is distributed in this repository.** All datasets require individual registration. See [data/README.md](data/README.md) for full instructions.

| Dataset | Purpose | Access |
|---------|---------|--------|
| I-CARE | Primary evaluation (n=55) | PhysioNet Credentialed licence |
| DEAP | E2 emotion pre-training | Keele University request form |
| DREAMER | E2 emotion pre-training | Contact authors (Katsigiannis et al.) |
| TUH EEG | E1 SSL pre-training | NEDC/Temple University account |

Once downloaded, update the paths at the top of `src/config.py`:

```python
ICARE_DIR  = Path("path/to/i-care/training")
DEAP_DIR   = Path("path/to/DEAP/data_preprocessed_matlab")
DREAMER_MAT= Path("path/to/DREAMER/DREAMER.mat")
TUH_DIR    = Path("path/to/tuh_eeg")
```

---

## Reproducing Results

### Option A  Evaluate with pre-trained checkpoints (fastest)

This skips all training and directly reproduces Table 1 (main results), Table 2 (transfer ablation), and the SOTA comparison using the provided checkpoints.

```bash
python -m src.stage4_eval.evaluate_pipeline --device cuda
```

Expected output (matches paper Table 1):
```
[PRIMARY] Binary outcome (patient-level, LOPO) AUC = 0.8798 [0.7748, 0.9678]
Sensitivity = 0.8974, Specificity = 0.7500, n_patients = 55
```

### Option B  Full training pipeline (reproduces from scratch)

Run stages in order. Each stage saves checkpoints to `results/checkpoints/`.

#### Stage 1  FractalSSL pre-training (E1 on TUH EEG)

```bash
python -m src.stage1_pretrain.train_fractalssl --epochs 100 --device cuda
python -m src.stage1_pretrain.finetune_e1_doc  --epochs 50  --device cuda
```

#### Stage 2  Lorentzian encoder pre-training (E2 on DEAP/DREAMER)

```bash
python -m src.stage2_emotion.train_emotion_encoder --dataset deap    --epochs 100 --device cuda
python -m src.stage2_emotion.train_emotion_encoder --dataset dreamer --epochs 100 --device cuda
python -m src.stage2_emotion.finetune_e2_doc       --epochs 80       --device cuda
```

#### Stage 3  Graph-GAT encoder training (E3 on I-CARE)

```bash
python -m src.stage3_doc.train_doc_encoder --epochs 80 --device cuda
```

#### Stage 4  LOPO evaluation

```bash
python -m src.stage4_eval.evaluate_pipeline --device cuda
```

#### Transfer ablation (Table 2)

```bash
python -m src.stage2_emotion.ablation_e2_transfer --epochs 80 --folds 5 --device cuda
```

### Option C  Reproduce all paper figures

```bash
python generate_all_figures.py
```

Figures are saved to `results/plots/`.

---

## Checkpoint Loading Example

```python
import torch
from src.models.lorentzian_encoder import LorentzianEncoder
from src.models.graph_encoder import GraphGATEncoder
from src.models.fractal_ssl import FractalSSLEncoder

e1 = FractalSSLEncoder()
e1.load_state_dict(torch.load("checkpoints/e1_doc_finetuned.pt", map_location="cpu"))
e1.eval()

e2 = LorentzianEncoder()
e2.load_state_dict(torch.load("checkpoints/e2_doc_icare.pt", map_location="cpu"))
e2.eval()

e3 = GraphGATEncoder()
e3.load_state_dict(torch.load("checkpoints/e3_graph_icare_best.pt", map_location="cpu"))
e3.eval()
```

---

## Verifying Results Match the Paper

After running `evaluate_pipeline.py`, check `results/tables/table1_main_results.csv`.
Key values that must match:

| Field | Expected value |
|-------|---------------|
| `binary_auc` | 0.8798 |
| `binary_auc_ci` | [0.7748, 0.9678] |
| `binary_sensitivity` | 0.8974 |
| `binary_specificity` | 0.7500 |
| `n_patients` | 55 |
| `E2 Lorentzian only` macro_auc | 0.7263 |
| `E2 Euclidean` macro_auc | 0.3788 |
| `E3 Graph only` macro_auc | 0.8824 |
| `pdi_threshold` | 0.11 |
| `car_recall` | 0.9882 |

---



---

## License

The source code in this repository is released under the **MIT License**.  
Pre-trained checkpoints are released for research purposes only and may only be used with datasets for which you have obtained the relevant data use agreement (I-CARE, DEAP, DREAMER, TUH EEG).

---

## Contact

Raghu Gangolu  rb22csr1p02@student.nitw.ac.in
Department of Computer Science and Engineering, NIT Warangal, India
