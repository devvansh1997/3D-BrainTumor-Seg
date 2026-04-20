from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    CropForegroundd,
    NormalizeIntensityd,
    Lambdad,
    RandSpatialCropd,
    SpatialPadd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

_IMG = ["image"]
_ALL = ["image", "label"]

# BraTS2021 label 4 (ET) → 3 for contiguous 0-3 class indices
_BRATS_LABEL_REMAP = Lambdad(keys=["label"], func=lambda x: x.where(x != 4, x.new_tensor(3)))


def _channel_first(data_format: str):
    """
    Return the EnsureChannelFirstd transforms appropriate for the data format.

    BraTS2021: image is a list of 4 separate 3D files → LoadImaged stacks them
               to [4,H,W,D] already; label is [H,W,D] and needs a channel dim.
    MSD:       image is a single 4D NIfTI [H,W,D,4] (channels-last NIfTI
               convention); label is [H,W,D].
    """
    if data_format == "msd":
        return [
            EnsureChannelFirstd(keys=["image"], channel_dim=-1),          # [H,W,D,4]→[4,H,W,D]
            EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"), # [H,W,D]→[1,H,W,D]
        ]
    else:  # brats2021
        return [
            EnsureChannelFirstd(keys=_ALL),   # image already [4,H,W,D]; label [H,W,D]→[1,H,W,D]
            _BRATS_LABEL_REMAP,               # remap label 4 → 3
        ]


def get_train_transforms(patch_size, data_format: str = "brats2021"):
    return Compose([
        LoadImaged(keys=_ALL, image_only=False),
        *_channel_first(data_format),
        EnsureTyped(keys=_ALL, dtype=None),
        NormalizeIntensityd(keys=_IMG, nonzero=True, channel_wise=True),
        CropForegroundd(keys=_ALL, source_key="image"),
        SpatialPadd(keys=_ALL, spatial_size=patch_size),
        RandSpatialCropd(keys=_ALL, roi_size=patch_size, random_size=False),
        RandFlipd(keys=_ALL, prob=0.5, spatial_axis=0),
        RandFlipd(keys=_ALL, prob=0.5, spatial_axis=1),
        RandFlipd(keys=_ALL, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=_ALL, prob=0.5, max_k=3),
        RandScaleIntensityd(keys=_IMG, factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=_IMG, offsets=0.1, prob=0.5),
        EnsureTyped(keys=_ALL),
    ])


def get_val_transforms(data_format: str = "brats2021"):
    return Compose([
        LoadImaged(keys=_ALL, image_only=False),
        *_channel_first(data_format),
        EnsureTyped(keys=_ALL, dtype=None),
        NormalizeIntensityd(keys=_IMG, nonzero=True, channel_wise=True),
        CropForegroundd(keys=_ALL, source_key="image"),
        EnsureTyped(keys=_ALL),
    ])
