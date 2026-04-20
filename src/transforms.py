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

_ALL = ["image", "label"]
_IMG = ["image"]

# BraTS uses label 4 for enhancing tumor; remap to 3 for contiguous 0-3 classes.
_LABEL_REMAP = Lambdad(keys=["label"], func=lambda x: x.where(x != 4, x.new_tensor(3)))


def get_train_transforms(patch_size):
    return Compose([
        LoadImaged(keys=_ALL, image_only=False),
        EnsureChannelFirstd(keys=_ALL),
        EnsureTyped(keys=_ALL, dtype=None),
        _LABEL_REMAP,
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


def get_val_transforms():
    return Compose([
        LoadImaged(keys=_ALL, image_only=False),
        EnsureChannelFirstd(keys=_ALL),
        EnsureTyped(keys=_ALL, dtype=None),
        _LABEL_REMAP,
        NormalizeIntensityd(keys=_IMG, nonzero=True, channel_wise=True),
        CropForegroundd(keys=_ALL, source_key="image"),
        EnsureTyped(keys=_ALL),
    ])
