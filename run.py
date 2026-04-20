"""
Usage:
  # Debug run (local machine — tiny data, 3 epochs, 64^3 patches)
  python run.py --config configs/dynunet.yaml --debug
  python run.py --config configs/swinunetr.yaml --debug

  # Full run (HPC)
  python run.py --config configs/dynunet.yaml
  python run.py --config configs/swinunetr.yaml

  # Eval only (load best checkpoint)
  python run.py --config configs/dynunet.yaml --eval-only
"""

import argparse

import yaml


def _apply_debug_overrides(config: dict):
    d = config["debug"]
    t = config["training"]
    t["max_epochs"]   = d["max_epochs"]
    t["patch_size"]   = d["patch_size"]
    t["batch_size"]   = d["batch_size"]
    t["val_interval"] = d["val_interval"]
    t["cache_rate"]   = d["cache_rate"]
    t["num_workers"]  = d["num_workers"]
    t["sw_batch_size"] = d.get("sw_batch_size", 1)

    mp = config["model_params"]
    if "deep_supervision" in d:
        mp["deep_supervision"] = d["deep_supervision"]
    if "feature_size" in d:
        mp["feature_size"] = d["feature_size"]

    config["output"]["checkpoint_dir"] = "results/debug/checkpoints"
    config["output"]["log_dir"]        = f"results/debug/logs/{config['model']}"


def main():
    parser = argparse.ArgumentParser(description="3D Brain Tumor Segmentation")
    parser.add_argument("--config",    required=True,       help="Path to config YAML")
    parser.add_argument("--debug",     action="store_true", help="Override config with debug settings")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; load best checkpoint and evaluate")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.debug:
        config["debug"]["enabled"] = True
        _apply_debug_overrides(config)

    from src.train import train
    train(config, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
