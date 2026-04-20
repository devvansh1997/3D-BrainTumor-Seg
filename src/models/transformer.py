from monai.networks.nets import SwinUNETR, UNETR


def build_swinunetr(config: dict) -> SwinUNETR:
    p = config["model_params"]

    return SwinUNETR(
        in_channels=p["in_channels"],
        out_channels=p["out_channels"],
        feature_size=p.get("feature_size", 48),
        use_checkpoint=p.get("use_checkpoint", True),
        spatial_dims=3,
    )


def build_unetr(config: dict) -> UNETR:
    p = config["model_params"]
    patch_size = tuple(config["training"]["patch_size"])

    return UNETR(
        in_channels=p["in_channels"],
        out_channels=p["out_channels"],
        img_size=patch_size,
        feature_size=p.get("feature_size", 16),
        hidden_size=p.get("hidden_size", 768),
        mlp_dim=p.get("mlp_dim", 3072),
        num_heads=p.get("num_heads", 12),
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )
