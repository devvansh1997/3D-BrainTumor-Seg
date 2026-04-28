# 3D Brain Tumor Segmentation: CNN vs Transformer

CAP 5516 — Medical Image Computing, Spring 2026.

A controlled comparison between **DynUNet** (nnU-Net-style CNN) and **SwinUNETR**
(Vision Transformer) on the MSD Task 01 brain tumor dataset, plus a
**boundary-aware loss** ablation that closes the CNN's HD95 gap to the Transformer.

---

## Key Results (Test Set, 74 cases)

| Model                       | Mean Dice ↑ | Mean HD95 ↓ | ED HD95 ↓ | Params | Inference |
|-----------------------------|:-----------:|:-----------:|:---------:|:------:|:---------:|
| DynUNet (baseline)          | **0.741**   | 6.94 mm     | 9.88 mm   | 16.5 M | 0.008 s/vol |
| SwinUNETR                   | 0.732       | **6.19 mm** | 7.27 mm   | 62.2 M | 0.122 s/vol |
| **DynUNet + BoundaryLoss**  | 0.722       | 6.50 mm     | **7.10 mm** | 16.5 M | 0.008 s/vol |

The boundary-aware loss reduces DynUNet's ED HD95 by **28%** without adding
parameters or inference cost.

---

## Project Structure

```
.
├── configs/                  # YAML configs (local debug + HPC for both models)
├── data/
│   ├── README.md             # Dataset download + layout instructions
│   └── raw/                  # NIfTI volumes (gitignored)
├── results/
│   ├── checkpoints/          # Best validation checkpoints
│   ├── logs/                 # TensorBoard event files
│   └── figures/              # Report figures (4 PNGs)
├── report/
│   ├── main.tex              # Final report
│   └── references.bib
├── scripts/
│   ├── hpc_job.sh            # SLURM training script
│   ├── verify_data.py        # Validate BraTS2021 folder layout
│   └── visualize_results.py  # Generate report figures from TensorBoard logs
├── src/
│   ├── dataset.py            # MSD + BraTS2021 dataset loaders
│   ├── transforms.py         # Preprocessing pipeline
│   ├── losses.py             # BoundaryAwareLoss (DiceCE + HausdorffDT)
│   ├── train.py              # Training loop with throttled boundary loss
│   ├── evaluate.py           # Test-time Dice / HD95 / inference timing
│   ├── utils.py              # Checkpoint I/O, seeding
│   └── models/
│       ├── baseline.py       # DynUNet builder
│       └── transformer.py    # SwinUNETR builder
├── run.py                    # CLI entrypoint
└── requirements.txt
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset (see data/README.md for full instructions)
#    BraTS2021 from Kaggle (local) OR MSD Task 01 (HPC default path)

# 3. Verify layout
python scripts/verify_data.py --data-dir data/raw
```

---

## Run

### Local debug (4 cases, 3 epochs — sanity check)
```bash
python run.py --config configs/dynunet.yaml   --debug
python run.py --config configs/swinunetr.yaml --debug
```

### Full training on HPC
```bash
sbatch scripts/hpc_job.sh dynunet
sbatch scripts/hpc_job.sh swinunetr
```

### Test-only evaluation
```bash
python run.py --config configs/dynunet_hpc.yaml --eval-only
```

### Generate report figures
```bash
python scripts/visualize_results.py
# Outputs to results/figures/
```

---

## Boundary-Aware Loss (Original Contribution)

`BoundaryAwareLoss` in `src/losses.py` combines `DiceCELoss` with MONAI's
`HausdorffDTLoss` (weight $\lambda=0.5$). Computing 3D distance transforms
every step is too slow for full training, so the boundary term is applied
**every $k=10$ steps** in the training loop (`src/train.py`), with plain
DiceCE filling the gaps. This is ${\sim}5\times$ faster per epoch than
applying boundary loss every step, while still producing the 28% ED HD95
improvement.

Enable it via the config:
```yaml
training:
  boundary_loss_weight: 0.5    # 0 = disabled (plain DiceCE)
  boundary_loss_freq:   10     # apply HausdorffDT every N steps
```

---

## External Resources

- **MONAI** (v1.3) — provides DynUNet, SwinUNETR, DiceCELoss, HausdorffDTLoss,
  sliding-window inference, and the data transforms.
- **PyTorch** (≥2.0) — backend.
- **MSD Task 01: Brain Tumour** — public dataset
  ([Antonelli et al.\ 2022](https://www.nature.com/articles/s41467-022-30695-9)).
- **BraTS 2021** (optional, local-only) — Kaggle mirror at
  `dschettler8845/brats-2021-task1`.

The model architectures, dataset, and loss function components are taken from
MONAI. The original contributions in this repository are the
`BoundaryAwareLoss` module, the throttled training loop, the experimental
design, and all reported analysis.

---

## Reproducibility

- Random seed fixed at 42 for splits, weight init, and CUDA RNG.
- Train / val / test split: 70 / 15 / 15 (deterministic).
- Hardware used: NVIDIA A100 80 GB (HPC).
- Best validation checkpoint per model is saved automatically and used for the
  final test evaluation.
