"""
Segmentation mask visualization — runs on the same dataset the model was trained on.

Loads N test cases from the configured dataset (uses the same seed=42 split
as training), runs sliding-window inference with both DynUNet+BL and SwinUNETR
checkpoints, and saves one figure per case.

Usage on HPC:
    python scripts/visualize_seg.py configs/dynunet_hpc.yaml          # default 2 cases
    python scripts/visualize_seg.py configs/dynunet_hpc.yaml --num 3  # pick 3
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from monai.inferers import sliding_window_inference

sys.path.insert(0, os.path.abspath("."))
from src.dataset    import get_dataloaders
from src.transforms import get_val_transforms
from src.models.baseline    import build_dynunet
from src.models.transformer import build_swinunetr
from src.utils      import load_checkpoint

os.makedirs("results/figures", exist_ok=True)

DYNUNET_CKPT = "results/checkpoints/dynunet_best.pth"
SWIN_CKPT    = "results/checkpoints/swinunetr_best.pth"

CLASS_COLORS = {
    1: (1.0, 0.2, 0.2, 0.65),   # NCR/NET — red
    2: (1.0, 0.9, 0.1, 0.55),   # ED      — yellow
    3: (0.1, 0.9, 0.9, 0.70),   # ET      — cyan
}
CLASS_LABELS = {1: "NCR/NET", 2: "ED", 3: "ET"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to YAML config (e.g. configs/dynunet_hpc.yaml)")
    p.add_argument("--num", type=int, default=2, help="Number of test cases to visualise")
    return p.parse_args()


def build_swin_cfg(base_cfg):
    """Build a config dict for SwinUNETR with default feature_size=48."""
    return {
        "model_params": {
            "in_channels":  base_cfg["model_params"]["in_channels"],
            "out_channels": base_cfg["model_params"]["out_channels"],
            "feature_size": 48,
        },
        "training": base_cfg["training"],
    }


def make_overlay(base_slice, mask_slice):
    mn, mx = base_slice.min(), base_slice.max()
    gray   = (base_slice - mn) / (mx - mn + 1e-8)
    rgba   = np.stack([gray, gray, gray, np.ones_like(gray)], axis=-1)
    for cls_id, color in CLASS_COLORS.items():
        mask = mask_slice == cls_id
        for c in range(3):
            rgba[mask, c] = rgba[mask, c] * (1 - color[3]) + color[c] * color[3]
    return rgba


@torch.no_grad()
def run_inference(model, image, patch_size, sw_bs, device):
    x   = image.unsqueeze(0).to(device)
    out = sliding_window_inference(x, patch_size, sw_bs, model)
    return out.argmax(dim=1).squeeze(0).cpu()


def plot_case(case_tag, image, label, pred_dyn, pred_swin, fig_path):
    t1ce = image[1].numpy() if image.shape[0] >= 2 else image[0].numpy()

    tumour_voxels = (label > 0).numpy()
    z_sums = tumour_voxels.sum(axis=(0, 1))
    if z_sums.max() == 0:
        slices = [t1ce.shape[2] // 2 + i for i in (-10, 0, 10)]
    else:
        top    = np.argsort(z_sums)[::-1][:20]
        step   = max(1, len(top) // 3)
        slices = sorted([top[0], top[step], top[2 * step]])

    cols  = ["T1ce", "Ground Truth", "DynUNet\n+ BoundaryLoss", "SwinUNETR"]
    preds = [None, label.numpy(), pred_dyn.numpy(), pred_swin.numpy()]

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle(f"Segmentation Visualisation — {case_tag}",
                 fontsize=14, fontweight="bold")

    for row, z in enumerate(slices):
        t1ce_sl = t1ce[:, :, z]
        for col, (title, mask) in enumerate(zip(cols, preds)):
            ax = axes[row][col]
            if mask is None:
                ax.imshow(t1ce_sl.T, cmap="gray", origin="lower")
            else:
                ax.imshow(make_overlay(t1ce_sl, mask[:, :, z]), origin="lower")
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            if col == 0:
                ax.text(-0.08, 0.5, f"z={z}", transform=ax.transAxes,
                        fontsize=10, va="center", ha="right")

    patches = [mpatches.Patch(color=CLASS_COLORS[i][:3], label=CLASS_LABELS[i])
               for i in CLASS_COLORS]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size   = cfg["training"]["patch_size"]
    sw_bs        = cfg["training"].get("sw_batch_size", 2)
    data_format  = cfg["data"].get("data_format", "msd")
    print(f"Device: {device}  |  data_format: {data_format}  |  patch: {patch_size}")

    # Use the same split as training (seed=42 deterministic)
    _, _, test_loader = get_dataloaders(
        cfg,
        get_val_transforms(data_format),   # train transforms unused for viz
        get_val_transforms(data_format),
    )

    # Build models with deep_supervision=True (matches saved checkpoints)
    dyn_cfg = dict(cfg)
    dyn_cfg["model_params"] = dict(cfg["model_params"])
    dyn_cfg["model_params"]["deep_supervision"] = True

    dyn  = build_dynunet(dyn_cfg).to(device).eval()
    swin = build_swinunetr(build_swin_cfg(cfg)).to(device).eval()
    load_checkpoint(dyn,  None, DYNUNET_CKPT, device)
    load_checkpoint(swin, None, SWIN_CKPT,    device)

    n_done = 0
    for i, batch in enumerate(test_loader):
        if n_done >= args.num:
            break
        image = batch["image"][0]   # [4, H, W, D]
        label = batch["label"][0].long()
        if label.ndim == 4:
            label = label.squeeze(0)

        # Skip cases with no tumour voxels (rare but possible)
        if (label > 0).sum() == 0:
            continue

        case_tag = f"test_case_{i:03d}"
        print(f"\nProcessing {case_tag}  shape={image.shape}")

        print("  → DynUNet+BL inference...")
        pred_dyn  = run_inference(dyn,  image, patch_size, sw_bs, device)
        print("  → SwinUNETR inference...")
        pred_swin = run_inference(swin, image, patch_size, sw_bs, device)

        fig_path = f"results/figures/fig5_seg_{case_tag}.png"
        plot_case(case_tag, image, label, pred_dyn, pred_swin, fig_path)
        n_done += 1

    print(f"\nDone. {n_done} figure(s) saved to results/figures/")


if __name__ == "__main__":
    main()
