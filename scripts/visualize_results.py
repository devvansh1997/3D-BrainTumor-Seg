"""
Generate all report figures from hardcoded test results.
Outputs to results/figures/

Usage:
    python scripts/visualize_results.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/figures", exist_ok=True)

# ── Test Results ────────────────────────────────────────────────────────────
CLASSES = ["NCR/NET", "ED", "ET"]

RESULTS = {
    "DynUNet\n(baseline)": {
        "dice": [0.7943, 0.5977, 0.8313],
        "hd95": [6.60,   9.88,   4.33],
        "mean_dice": 0.7411,
        "mean_hd95": 6.94,
        "infer_s":   0.008,
        "gpu_mb":    10322,
        "color":     "#4C72B0",
    },
    "SwinUNETR": {
        "dice": [0.7932, 0.5867, 0.8145],
        "hd95": [7.41,   7.27,   3.89],
        "mean_dice": 0.7315,
        "mean_hd95": 6.19,
        "infer_s":   0.122,
        "gpu_mb":    17028,
        "color":     "#DD8452",
    },
    "DynUNet\n+ BoundaryLoss": {
        "dice": [0.7858, 0.5799, 0.8013],
        "hd95": [6.39,   7.10,   6.03],
        "mean_dice": 0.7223,
        "mean_hd95": 6.50,
        "infer_s":   0.008,
        "gpu_mb":    10322,
        "color":     "#55A868",
    },
}

MODELS   = list(RESULTS.keys())
COLORS   = [RESULTS[m]["color"] for m in MODELS]

# ── Training curves (from real TensorBoard event files) ─────────────────────
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def _load_tb(path, tag):
    ea = EventAccumulator(path)
    ea.Reload()
    s = ea.Scalars(tag)
    return [(x.step + 1, x.value) for x in s]   # step is 0-indexed

_BASE_FILE = "results/logs/dynunet/events.out.tfevents.1776780001.evc22.3281765.0"
_SWIN_FILE = "results/logs/swinunetr/events.out.tfevents.1776714712.evc32.2929074.0"
_BL_FILE   = "results/logs/dynunet/events.out.tfevents.1777318940.evc36.3347328.0"

_dynunet_base_val  = _load_tb(_BASE_FILE, "val/mean_dice")
_swin_val          = _load_tb(_SWIN_FILE, "val/mean_dice")
_dynunet_bl_val    = _load_tb(_BL_FILE,   "val/mean_dice")

_dynunet_base_loss = _load_tb(_BASE_FILE, "train/loss")
_swin_loss         = _load_tb(_SWIN_FILE, "train/loss")
_dynunet_bl_loss   = _load_tb(_BL_FILE,   "train/loss")


# ── Figure 1: Per-class Dice bar chart ──────────────────────────────────────
def plot_dice_bars():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(CLASSES))
    w = 0.25

    for ax, metric, ylabel, title in [
        (axes[0], "dice", "Dice Score",       "Per-class Dice Score"),
        (axes[1], "hd95", "HD95 (mm)",        "Per-class HD95 (mm)"),
    ]:
        for j, (model, res) in enumerate(RESULTS.items()):
            vals = res[metric]
            bars = ax.bar(x + (j - 1) * w, vals, w,
                          label=model.replace("\n", " "),
                          color=res["color"], edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                fmt = f"{v:.3f}" if metric == "dice" else f"{v:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.005 if metric == "dice" else 0.1),
                        fmt, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "dice":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0, 13)
            # highlight ED bar for boundary loss model
            ax.annotate("−28%\nvs baseline", xy=(x[1] + w, 7.10),
                        xytext=(x[1] + w + 0.35, 9.5),
                        arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                        fontsize=8, color="green", fontweight="bold")

    fig.suptitle("DynUNet vs SwinUNETR vs DynUNet+BoundaryLoss — Test Set",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = "results/figures/fig1_per_class_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: Mean metrics summary bar chart ────────────────────────────────
def plot_mean_summary():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    labels  = [m.replace("\n", "\n") for m in MODELS]
    colors  = COLORS

    metrics = [
        ("mean_dice",  "Mean Dice Score",  axes[0], None),
        ("mean_hd95",  "Mean HD95 (mm)",   axes[1], None),
        ("infer_s",    "Inference (s/vol)", axes[2], None),
    ]

    for key, ylabel, ax, _ in metrics:
        vals = [RESULTS[m][key] for m in MODELS]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.5)
        for bar, v in zip(bars, vals):
            fmt = f"{v:.4f}" if key == "mean_dice" else (f"{v:.2f}" if key == "mean_hd95" else f"{v:.3f}")
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ypad = max(vals) * 0.15
        ax.set_ylim(0, max(vals) + ypad)

    fig.suptitle("Summary — Mean Metrics on Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = "results/figures/fig2_mean_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: Training curves (val Dice + train loss) ───────────────────────
def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    curve_cfg = [
        (_dynunet_base_val,  _dynunet_base_loss,  "DynUNet (baseline)",      "#4C72B0", "--"),
        (_swin_val,          _swin_loss,           "SwinUNETR",               "#DD8452", "-"),
        (_dynunet_bl_val,    _dynunet_bl_loss,     "DynUNet + BoundaryLoss",  "#55A868", "-"),
    ]

    for val_data, loss_data, label, color, ls in curve_cfg:
        ve, vd = zip(*val_data)
        le, ld = zip(*loss_data)
        axes[0].plot(ve, vd, linestyle=ls, color=color, linewidth=2, label=label, alpha=0.9)
        axes[1].plot(le, ld, linestyle=ls, color=color, linewidth=2, label=label, alpha=0.9)

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Validation Dice", fontsize=12)
    axes[0].set_title("Validation Dice During Training", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, linestyle="--")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0.1, 0.82)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Training Loss", fontsize=12)
    axes[1].set_title("Training Loss During Training", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, linestyle="--")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = "results/figures/fig3_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Figure 4: HD95 improvement spotlight ────────────────────────────────────
def plot_hd95_spotlight():
    """Horizontal bar chart highlighting ED HD95 improvement."""
    fig, ax = plt.subplots(figsize=(9, 4))

    model_labels = [m.replace("\n", " ") for m in MODELS]
    ed_hd95      = [RESULTS[m]["hd95"][1] for m in MODELS]  # ED class
    colors_local = COLORS

    bars = ax.barh(model_labels, ed_hd95, color=colors_local,
                   edgecolor="white", height=0.5)
    for bar, v in zip(bars, ed_hd95):
        ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f} mm", va="center", fontsize=11, fontweight="bold")

    ax.axvline(ed_hd95[1], color="#DD8452", linestyle=":", alpha=0.6, linewidth=1.5,
               label=f"SwinUNETR ED HD95 = {ed_hd95[1]:.2f} mm")
    ax.set_xlabel("HD95 (mm) — lower is better", fontsize=11)
    ax.set_title("ED (Peritumoral Edema) HD95 — Key Novelty Result", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 13)

    # annotation
    ax.annotate("Boundary loss closes\ngap to SwinUNETR",
                xy=(ed_hd95[2], 2), xytext=(10.5, 1.6),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                fontsize=9, color="green", fontweight="bold")

    plt.tight_layout()
    path = "results/figures/fig4_ed_hd95_spotlight.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_dice_bars()
    plot_mean_summary()
    plot_training_curves()
    plot_hd95_spotlight()
    print("\nAll figures saved to results/figures/")
