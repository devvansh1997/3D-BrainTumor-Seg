import json
import re
from pathlib import Path
from typing import List, Dict

from monai.data import CacheDataset, Dataset, DataLoader

# ---------- BraTS2021 (local) — one subdir per patient, 4 separate files ----------
_BRATS_PATTERNS = {
    "t1":    re.compile(r".*_t1\.nii(\.gz)?$",    re.IGNORECASE),
    "t1ce":  re.compile(r".*_t1ce\.nii(\.gz)?$",  re.IGNORECASE),
    "t2":    re.compile(r".*_t2\.nii(\.gz)?$",     re.IGNORECASE),
    "flair": re.compile(r".*_flair\.nii(\.gz)?$",  re.IGNORECASE),
    "seg":   re.compile(r".*_seg\.nii(\.gz)?$",    re.IGNORECASE),
}


def _discover_brats2021(root: Path) -> List[Dict]:
    cases = []
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        found = {k: None for k in _BRATS_PATTERNS}
        for f in patient_dir.iterdir():
            for key, pat in _BRATS_PATTERNS.items():
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
    return cases


# ---------- MSD Task01 (HPC) — imagesTr/ + labelsTr/, single 4D NIfTI ----------

def _discover_msd(root: Path) -> List[Dict]:
    json_path = root / "dataset.json"
    with open(json_path) as f:
        meta = json.load(f)

    cases = []
    for entry in meta["training"]:
        # Paths in dataset.json are relative: "./imagesTr/BRATS_001.nii.gz"
        img_path = root / entry["image"].lstrip("./")
        lbl_path = root / entry["label"].lstrip("./")
        if not img_path.exists() or not lbl_path.exists():
            print(f"[SKIP] Missing: {img_path.name}")
            continue
        cases.append({
            "image": str(img_path),
            "label": str(lbl_path),
            "case_id": img_path.stem,
        })
    return cases


# ---------- public API ----------

def discover_cases(data_dir: str, data_format: str = "brats2021") -> List[Dict]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir not found: {root}")

    if data_format == "msd":
        cases = _discover_msd(root)
    else:
        cases = _discover_brats2021(root)

    print(f"[DATA] {len(cases)} cases found ({data_format}) in {root}")
    return cases


def get_dataloaders(config: dict, transforms_train, transforms_val):
    data_format = config["data"].get("data_format", "brats2021")
    cases = discover_cases(config["data"]["data_dir"], data_format)
    if not cases:
        raise RuntimeError(f"No valid cases found in {config['data']['data_dir']}")

    if config["debug"]["enabled"]:
        cases = cases[: config["debug"]["num_samples"]]
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
