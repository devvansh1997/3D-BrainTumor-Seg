import re
from pathlib import Path
from typing import List, Dict

from monai.data import CacheDataset, Dataset, DataLoader

# BraTS 2021 naming (one sub-folder per patient):
#   BraTS2021_XXXXX/BraTS2021_XXXXX_t1.nii.gz
#   BraTS2021_XXXXX/BraTS2021_XXXXX_t1ce.nii.gz
#   BraTS2021_XXXXX/BraTS2021_XXXXX_t2.nii.gz
#   BraTS2021_XXXXX/BraTS2021_XXXXX_flair.nii.gz
#   BraTS2021_XXXXX/BraTS2021_XXXXX_seg.nii.gz
#
# Label convention: 0=BG, 1=NCR/NET, 2=ED, 4=ET
# We remap 4→3 in transforms.py so the model sees 0/1/2/3.

_PATTERNS = {
    "t1":    re.compile(r".*_t1\.nii(\.gz)?$",    re.IGNORECASE),
    "t1ce":  re.compile(r".*_t1ce\.nii(\.gz)?$",  re.IGNORECASE),
    "t2":    re.compile(r".*_t2\.nii(\.gz)?$",     re.IGNORECASE),
    "flair": re.compile(r".*_flair\.nii(\.gz)?$",  re.IGNORECASE),
    "seg":   re.compile(r".*_seg\.nii(\.gz)?$",    re.IGNORECASE),
}


def discover_cases(data_dir: str) -> List[Dict]:
    """
    Walk data_dir expecting one sub-folder per BraTS patient.
    Returns only cases that have all 5 files.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir not found: {root}")

    cases = []
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue

        found = {k: None for k in _PATTERNS}
        for f in patient_dir.iterdir():
            for key, pat in _PATTERNS.items():
                if pat.match(f.name) and found[key] is None:
                    found[key] = str(f)

        missing = [k for k, v in found.items() if v is None]
        if missing:
            print(f"[SKIP] {patient_dir.name}: missing {missing}")
            continue

        cases.append({
            "image": [found["t1"], found["t1ce"], found["t2"], found["flair"]],
            "label": found["seg"],
            "case_id": patient_dir.name,
        })

    print(f"[DATA] Found {len(cases)} usable cases in {root}")
    return cases


def get_dataloaders(config: dict, transforms_train, transforms_val):
    cases = discover_cases(config["data"]["data_dir"])
    if not cases:
        raise RuntimeError(f"No valid cases found in {config['data']['data_dir']}")

    if config["debug"]["enabled"]:
        n_use = config["debug"]["num_samples"]
        cases = cases[:n_use]
        print(f"[DEBUG] Capped to {len(cases)} cases.")

    splits = config["data"]["train_val_test_split"]
    n = len(cases)
    n_train = max(1, int(n * splits[0]))
    n_val   = max(1, int(n * splits[1]))

    train_cases = cases[:n_train]
    val_cases   = cases[n_train : n_train + n_val]
    test_cases  = cases[n_train + n_val :]

    print(f"[DATA] train={len(train_cases)}  val={len(val_cases)}  test={len(test_cases)}")

    cache_rate  = config["training"]["cache_rate"]
    num_workers = config["training"].get("num_workers", 4)
    batch_size  = config["training"]["batch_size"]

    def make_ds(data, tfm):
        if cache_rate > 0:
            return CacheDataset(data=data, transform=tfm, cache_rate=cache_rate, num_workers=num_workers)
        return Dataset(data=data, transform=tfm)

    train_ds = make_ds(train_cases, transforms_train)
    val_ds   = make_ds(val_cases,   transforms_val)
    test_ds  = make_ds(test_cases,  transforms_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
