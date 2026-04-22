import torch.nn as nn
from monai.losses import DiceCELoss, HausdorffDTLoss


class BoundaryAwareLoss(nn.Module):
    """DiceCE + weighted HausdorffDT loss. Targets DynUNet's HD95 weakness."""
    def __init__(self, boundary_weight: float = 0.5):
        super().__init__()
        self.dice_ce  = DiceCELoss(to_onehot_y=True, softmax=True)
        self.hd_loss  = HausdorffDTLoss(include_background=False, to_onehot_y=True, softmax=True)
        self.w = boundary_weight

    def forward(self, pred, target):
        return self.dice_ce(pred, target) + self.w * self.hd_loss(pred, target)


def get_loss_fn(config: dict = None):
    """
    Returns BoundaryAwareLoss if boundary_loss_weight > 0 in config, else plain DiceCELoss.
    Only DynUNet configs set this weight — SwinUNETR uses standard DiceCE unchanged.
    """
    weight = 0.0
    if config:
        weight = config.get("training", {}).get("boundary_loss_weight", 0.0)

    if weight > 0:
        print(f"[LOSS] BoundaryAwareLoss  (DiceCE + {weight} × HausdorffDT)")
        return BoundaryAwareLoss(boundary_weight=weight)

    return DiceCELoss(to_onehot_y=True, softmax=True)


def compute_loss(loss_fn, outputs, labels):
    """
    Handles three DynUNet deep supervision output formats across MONAI versions:
      - list/tuple of tensors  [5D, 5D, ...]
      - stacked 6D tensor      [B, num_heads, C, H, W, D]
      - plain 5D tensor        [B, C, H, W, D]
    Deep supervision weights: 1, 0.5, 0.25, ...
    """
    if isinstance(outputs, (list, tuple)):
        return sum(0.5**i * loss_fn(o, labels) for i, o in enumerate(outputs))
    if outputs.ndim == 6:
        n = outputs.shape[1]
        return sum(0.5**i * loss_fn(outputs[:, i], labels) for i in range(n))
    return loss_fn(outputs, labels)
