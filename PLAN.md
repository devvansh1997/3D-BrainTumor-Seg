# Project Plan: 3D Brain Tumor Segmentation

**Course:** CAP 5516 — Medical Image Computing, Spring 2026  
**Goal:** Systematic comparison of CNN-based (nnU-Net / Dynamic U-Net) vs Transformer-based (UNETR / SwinUNETR) architectures on the UPenn-GBM dataset.

---

## Repo Structure

```
3D-BrainTumor-Seg/
├── data/
│   ├── raw/                  # Downloaded UPenn-GBM NIfTI volumes
│   └── processed/            # Cached preprocessed tensors (optional)
├── src/
│   ├── dataset.py            # MONAI CacheDataset, data splits, fold logic
│   ├── transforms.py         # Preprocessing + augmentation pipelines
│   ├── models/
│   │   ├── baseline.py       # DynUNet (nnU-Net-style) via MONAI
│   │   └── transformer.py    # UNETR / SwinUNETR via MONAI
│   ├── losses.py             # DiceCELoss, per-class weighting
│   ├── train.py              # Training loop w/ checkpointing + W&B logging
│   ├── evaluate.py           # DSC, HD95 per class (ET, NCR/NET, ED)
│   └── utils.py              # Visualization, seed fixing, config loading
├── configs/
│   ├── baseline.yaml         # DynUNet hyperparameters
│   └── transformer.yaml      # UNETR / SwinUNETR hyperparameters
├── notebooks/
│   └── eda.ipynb             # Dataset exploration, class distributions
├── scripts/
│   ├── download_data.sh      # TCIA download instructions / NBIA Data Retriever
│   └── run_experiment.sh     # Single-command training entrypoint
├── results/
│   ├── checkpoints/          # Best model weights per architecture
│   ├── logs/                 # Training curves
│   └── figures/              # Segmentation overlays, comparison plots
├── requirements.txt
└── README.md
```

---

## Phase 1 — Data & Environment Setup (Weeks 1–2)

### 1.1 Environment
- [ ] Create conda/venv environment
- [ ] Install: `monai`, `torch`, `nibabel`, `SimpleITK`, `numpy`, `matplotlib`, `wandb`, `pyyaml`, `tqdm`
- [ ] Pin versions in `requirements.txt`

### 1.2 Dataset (UPenn-GBM via TCIA)
- [ ] Download dataset using NBIA Data Retriever
- [ ] Verify all 4 modalities present per case: T1, T1ce, T2, FLAIR
- [ ] Verify label maps contain: ET (label 3), NCR/NET (label 1), ED (label 2)
- [ ] Inspect volume shapes, voxel spacings, intensity ranges in `eda.ipynb`

### 1.3 Preprocessing Pipeline (`src/transforms.py`)
- [ ] Load NIfTI → MONAI `LoadImaged`
- [ ] Stack 4 modalities as channels (`ConcatItemsd` or `NormalizeIntensityd`)
- [ ] Z-score normalization per modality (non-zero voxel mask)
- [ ] Crop foreground (`CropForegroundd`)
- [ ] Define train/val/test split (e.g., 70/15/15 or 5-fold CV)

### 1.4 Augmentation (training only)
- Random 3D rotation, random flipping (all axes)
- Random intensity scaling and shift
- `RandSpatialCropd` for patch sampling (e.g., 128×128×128)

---

## Phase 2 — CNN Baseline: DynUNet / nnU-Net (Weeks 3–4, first half)

### 2.1 Model (`src/models/baseline.py`)
- [ ] Use MONAI `DynUNet` with nnU-Net-style kernel/stride configuration
- [ ] Deep supervision heads (standard nnU-Net practice)
- [ ] Configure: input channels=4, output classes=4 (BG + 3 regions)

### 2.2 Training (`src/train.py`)
- [ ] Loss: `DiceCELoss` with softmax output
- [ ] Optimizer: SGD with momentum=0.99, poly LR schedule (or Adam)
- [ ] Patch size: 128×128×128; batch size: 2
- [ ] 1000 epochs (or epoch = 250 steps); save best checkpoint by mean Dice
- [ ] Log: train/val loss, per-class Dice, epoch time, GPU memory peak

---

## Phase 3 — Transformer Model: UNETR / SwinUNETR (Weeks 3–4, second half)

### 3.1 Model (`src/models/transformer.py`)
- [ ] Implement `SwinUNETR` from MONAI (primary transformer model)
- [ ] Optionally also implement `UNETR` for ablation
- [ ] Match input/output spec to baseline (4 channels in, 4 classes out)

### 3.2 Training (controlled match to baseline)
- [ ] Same loss, same augmentation, same patch size, same number of steps
- [ ] Same optimizer family; tune LR separately for fair comparison
- [ ] Use pretrained SwinUNETR SSL weights if available (note in report)

---

## Phase 4 — Training & Hyperparameter Tuning (Weeks 5–6)

- [ ] Full training run for DynUNet on all folds / full split
- [ ] Full training run for SwinUNETR on all folds / full split
- [ ] Hyperparameter sweep (LR, patch size, batch size) — log with W&B
- [ ] Track per-epoch: train loss, val Dice (ET, NCR/NET, ED), GPU mem, time

---

## Phase 5 — Evaluation & Analysis (Weeks 7–8)

### 5.1 Segmentation Metrics (`src/evaluate.py`)
- [ ] Mean Dice (overall)
- [ ] Per-class Dice: ET, NCR/NET, ED
- [ ] HD95 per class (use `monai.metrics.HausdorffDistanceMetric`)
- [ ] Run on held-out test set with best checkpoint per architecture

### 5.2 Computational Metrics
- [ ] Training time per epoch (logged during training)
- [ ] Inference latency: average over test set (sliding window inference)
- [ ] Peak GPU memory: `torch.cuda.max_memory_allocated()`

### 5.3 Analysis
- [ ] Statistical significance (Wilcoxon signed-rank test on Dice scores)
- [ ] Qualitative: side-by-side segmentation overlays for representative cases
- [ ] Failure case analysis: where does each model struggle?

---

## Phase 6 — Report & Presentation (Final)

- [ ] Methods: dataset, preprocessing, architectures, training protocol
- [ ] Results table: Dice (per-class) + HD95 + compute metrics
- [ ] Discussion: CNN vs Transformer trade-offs on GBM subregions
- [ ] Conclusion & future work

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | MONAI | Purpose-built for 3D medical segmentation |
| CNN model | MONAI `DynUNet` | Closest open-source nnU-Net equivalent |
| Transformer model | `SwinUNETR` | SOTA, hierarchical, memory-efficient |
| Loss | `DiceCELoss` | Standard for class-imbalanced segmentation |
| Patch size | 128³ | Balances context and GPU memory |
| Evaluation | Sliding window inference | Handles full-volume test-time inference |

---

## Evaluation Summary Table (target format for report)

| Model | ET Dice | NCR Dice | ED Dice | Mean Dice | HD95 (ET) | Train Time/epoch | Inference Time | GPU Mem |
|---|---|---|---|---|---|---|---|---|
| DynUNet | — | — | — | — | — | — | — | — |
| SwinUNETR | — | — | — | — | — | — | — | — |
| UNETR (opt.) | — | — | — | — | — | — | — | — |
