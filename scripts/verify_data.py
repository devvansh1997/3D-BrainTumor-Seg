"""
Verify dataset integrity. Auto-detects format from directory structure.

Usage:
    python scripts/verify_data.py --data-dir data/raw
    python scripts/verify_data.py --data-dir /lustre/fs1/.../Task01_BrainTumour/
"""

import argparse
import json
import re
import sys
from pathlib import Path

_BRATS_PATTERNS = {
    "t1":    re.compile(r".*_t1\.nii(\.gz)?$",    re.IGNORECASE),
    "t1ce":  re.compile(r".*_t1ce\.nii(\.gz)?$",  re.IGNORECASE),
    "t2":    re.compile(r".*_t2\.nii(\.gz)?$",     re.IGNORECASE),
    "flair": re.compile(r".*_flair\.nii(\.gz)?$",  re.IGNORECASE),
    "seg":   re.compile(r".*_seg\.nii(\.gz)?$",    re.IGNORECASE),
}


def verify_msd(root: Path):
    json_path = root / "dataset.json"
    with open(json_path) as f:
        meta = json.load(f)

    print(f"Format : MSD  |  dataset: {meta.get('name')}  |  declared training cases: {meta['numTraining']}")
    print(f"Labels : {meta.get('labels')}\n")

    missing = complete = 0
    for entry in meta["training"]:
        img = root / entry["image"].lstrip("./")
        lbl = root / entry["label"].lstrip("./")
        if img.exists() and lbl.exists():
            complete += 1
        else:
            missing += 1
            print(f"  [!!] Missing  img={img.exists()}  lbl={lbl.exists()}  — {img.name}")

    print(f"\nSummary: {complete} complete | {missing} missing")
    if missing:
        sys.exit(1)


def verify_brats2021(root: Path):
    print("Format : BraTS2021 (per-patient subdirs)\n")
    complete = incomplete = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        found = {k: None for k in _BRATS_PATTERNS}
        for f in d.iterdir():
            for key, pat in _BRATS_PATTERNS.items():
                if pat.match(f.name):
                    found[key] = f.name
        missing = [k for k, v in found.items() if v is None]
        if not missing:
            complete += 1
        else:
            incomplete += 1
            print(f"  [!!] {d.name}  missing: {missing}")

    print(f"\nSummary: {complete} complete | {incomplete} incomplete")
    if incomplete:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        print(f"[ERROR] Not found: {root}")
        sys.exit(1)

    if (root / "dataset.json").exists():
        verify_msd(root)
    else:
        verify_brats2021(root)


if __name__ == "__main__":
    main()
