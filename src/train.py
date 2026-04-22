import os
import time

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.evaluate import evaluate
from src.losses import compute_loss, get_loss_fn
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import load_checkpoint, peak_gpu_mb, save_checkpoint, set_seed


def _build_model(config):
    name = config["model"]
    if name == "dynunet":
        from src.models.baseline import build_dynunet
        return build_dynunet(config)
    if name == "swinunetr":
        from src.models.transformer import build_swinunetr
        return build_swinunetr(config)
    if name == "unetr":
        from src.models.transformer import build_unetr
        return build_unetr(config)
    raise ValueError(f"Unknown model: {name}")


def train(config: dict, eval_only: bool = False):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {config['model']}  |  Debug: {config['debug']['enabled']}")

    patch_size   = config["training"]["patch_size"]
    max_epochs   = config["training"]["max_epochs"]
    val_interval = config["training"]["val_interval"]
    sw_bs        = config["training"].get("sw_batch_size", 2)
    n_cls        = config["model_params"]["out_channels"]
    ckpt_dir     = config["output"]["checkpoint_dir"]
    log_dir      = config["output"]["log_dir"]
    model_tag    = config["model"]
    data_format  = config["data"].get("data_format", "brats2021")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        config,
        get_train_transforms(patch_size, data_format),
        get_val_transforms(data_format),
    )

    # Model
    model = _build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    if eval_only:
        ckpt_path = os.path.join(ckpt_dir, f"{model_tag}_best.pth")
        load_checkpoint(model, None, ckpt_path, device)
        evaluate(model, test_loader, device, config, split="test")
        return

    loss_fn   = get_loss_fn(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    post_pred  = AsDiscrete(argmax=True, to_onehot=n_cls)
    post_label = AsDiscrete(to_onehot=n_cls)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    writer = SummaryWriter(log_dir=log_dir)
    best_dice = 0.0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        t_epoch = time.perf_counter()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = compute_loss(loss_fn, outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(train_loader)
        epoch_time  = time.perf_counter() - t_epoch

        writer.add_scalar("train/loss",       epoch_loss,  epoch)
        writer.add_scalar("train/epoch_s",    epoch_time,  epoch)
        writer.add_scalar("train/gpu_mem_mb", peak_gpu_mb(), epoch)

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_in  = val_batch["image"].to(device)
                    val_lbl = val_batch["label"].to(device)
                    val_out = sliding_window_inference(val_in, patch_size, sw_bs, model)
                    val_preds = [post_pred(i)  for i in decollate_batch(val_out)]
                    val_lbls  = [post_label(i) for i in decollate_batch(val_lbl)]
                    dice_metric(y_pred=val_preds, y=val_lbls)

            scores    = dice_metric.aggregate()   # [3]: NCR/NET, ED, ET
            mean_dice = scores.mean().item()
            dice_metric.reset()

            writer.add_scalar("val/mean_dice",    mean_dice,    epoch)
            writer.add_scalar("val/dice_NCR_NET", scores[0].item(), epoch)
            writer.add_scalar("val/dice_ED",      scores[1].item(), epoch)
            writer.add_scalar("val/dice_ET",      scores[2].item(), epoch)

            print(
                f"Epoch {epoch+1:4d} | loss {epoch_loss:.4f} | "
                f"val Dice {mean_dice:.4f} "
                f"(NCR {scores[0]:.3f}  ED {scores[1]:.3f}  ET {scores[2]:.3f}) | "
                f"{epoch_time:.1f}s | GPU {peak_gpu_mb():.0f} MB"
            )

            if mean_dice > best_dice:
                best_dice = mean_dice
                save_checkpoint(model, optimizer, epoch, ckpt_dir, model_tag)

    writer.close()
    print(f"\nTraining done. Best val Dice: {best_dice:.4f}")

    # Final test evaluation using best checkpoint
    if len(test_loader.dataset) > 0:
        ckpt_path = os.path.join(ckpt_dir, f"{model_tag}_best.pth")
        load_checkpoint(model, None, ckpt_path, device)
        evaluate(model, test_loader, device, config, split="test")
