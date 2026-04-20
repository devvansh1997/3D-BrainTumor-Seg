"""
Verify BraTS 2021 data structure.

Usage:
    python scripts/verify_data.py
    python scripts/verify_data.py --data-dir data/raw
"""

import argparse
import re
import sys
from pathlib import Path

_PATTERNS = {
    "t1":    re.compile(r".*_t1\.nii(\.gz)?$",    re.IGNORECASE),
    "t1ce":  re.compile(r".*_t1ce\.nii(\.gz)?$",  re.IGNORECASE),
    "t2":    re.compile(r".*_t2\.nii(\.gz)?$",     re.IGNORECASE),
    "flair": re.compile(r".*_flair\.nii(\.gz)?$",  re.IGNORECASE),
    "seg":   re.compile(r".*_seg\.nii(\.gz)?$",    re.IGNORECASE),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        print(f"[ERROR] Not found: {root}")
        sys.exit(1)

    patient_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not patient_dirs:
        print(f"[ERROR] No sub-folders found in {root}")
        sys.exit(1)

    complete = incomplete = 0
    for d in patient_dirs:
        found = {k: None for k in _PATTERNS}
        for f in d.iterdir():
            for key, pat in _PATTERNS.items():
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


if __name__ == "__main__":
    main()
