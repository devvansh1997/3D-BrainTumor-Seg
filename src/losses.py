from monai.losses import DiceCELoss


def get_loss_fn() -> DiceCELoss:
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
        # stacked deep supervision: [B, num_heads, C, H, W, D]
        n = outputs.shape[1]
        return sum(0.5**i * loss_fn(outputs[:, i], labels) for i in range(n))
    return loss_fn(outputs, labels)
