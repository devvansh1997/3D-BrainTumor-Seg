from monai.losses import DiceCELoss


def get_loss_fn() -> DiceCELoss:
    return DiceCELoss(to_onehot_y=True, softmax=True)


def compute_loss(loss_fn, outputs, labels):
    """
    Handles DynUNet deep supervision (list of predictions) and normal outputs.
    Deep supervision weights: 1, 0.5, 0.25, ...
    """
    if isinstance(outputs, (list, tuple)):
        return sum(0.5**i * loss_fn(o, labels) for i, o in enumerate(outputs))
    return loss_fn(outputs, labels)
