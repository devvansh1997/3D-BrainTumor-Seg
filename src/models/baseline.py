from monai.networks.nets import DynUNet


def build_dynunet(config: dict) -> DynUNet:
    p = config["model_params"]
    kernels = [tuple(k) for k in p["kernels"]]
    strides = [tuple(s) for s in p["strides"]]

    return DynUNet(
        spatial_dims=3,
        in_channels=p["in_channels"],
        out_channels=p["out_channels"],
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=p.get("deep_supervision", False),
        deep_supr_num=p.get("deep_supr_num", 2),
    )
