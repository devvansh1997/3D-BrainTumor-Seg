import time
from typing import Dict

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from tqdm import tqdm

CLASS_NAMES = ["NCR/NET", "ED", "ET"]   # exclude background (index 0)


def evaluate(model, loader, device, config, split: str = "test") -> Dict[str, float]:
    """
    Run full evaluation on a DataLoader.
    Returns per-class and mean DSC + HD95, plus mean inference time per volume.
    """
    n_cls      = config["model_params"]["out_channels"]
    patch_size = config["training"]["patch_size"]
    sw_bs      = config["training"].get("sw_batch_size", 2)

    post_pred  = AsDiscrete(argmax=True, to_onehot=n_cls)
    post_label = AsDiscrete(to_onehot=n_cls)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")

    model.eval()
    infer_times = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{split}] evaluating"):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            t0 = time.perf_counter()
            outputs = sliding_window_inference(inputs, patch_size, sw_bs, model)
            infer_times.append(time.perf_counter() - t0)

            preds  = [post_pred(i)  for i in decollate_batch(outputs)]
            labels_dec = [post_label(i) for i in decollate_batch(labels)]

            dice_metric(y_pred=preds, y=labels_dec)
            hd95_metric(y_pred=preds, y=labels_dec)

    dice_scores = dice_metric.aggregate()   # shape [3]: NCR/NET, ED, ET
    hd95_scores = hd95_metric.aggregate()
    dice_metric.reset()
    hd95_metric.reset()

    results: Dict[str, float] = {}
    for i, name in enumerate(CLASS_NAMES):
        results[f"dice_{name}"] = dice_scores[i].item()
        results[f"hd95_{name}"] = hd95_scores[i].item()
    results["mean_dice"] = dice_scores.mean().item()
    results["mean_hd95"] = hd95_scores.mean().item()
    results["mean_infer_s"] = sum(infer_times) / len(infer_times) if infer_times else 0.0

    _print_results(results, split)
    return results


def _print_results(results: Dict[str, float], split: str):
    print(f"\n{'='*50}")
    print(f"  Results [{split}]")
    print(f"{'='*50}")
    for name in CLASS_NAMES:
        print(f"  {name:<10}  Dice: {results[f'dice_{name}']:.4f}   HD95: {results[f'hd95_{name}']:.2f} mm")
    print(f"  {'Mean':<10}  Dice: {results['mean_dice']:.4f}   HD95: {results['mean_hd95']:.2f} mm")
    print(f"  Inference: {results['mean_infer_s']:.3f} s/vol")
    print(f"{'='*50}\n")
