"""
Download the UPenn-GBM dataset from the data-nih/tcia GitHub release.

Usage:
    python scripts/download_data.py              # download all files
    python scripts/download_data.py --seg-only   # download only cases that have segmentation
    python scripts/download_data.py --n 4        # download first N complete cases (good for debug)

Files land in data/raw/ as:
    sub-002_T1w.nii.gz
    sub-002_T2w.nii.gz
    sub-002_FLAIR.nii.gz
    sub-002_ce-gd_T1w.nii.gz
    sub-002_seg.nii.gz
    ...
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

RELEASE_API = "https://api.github.com/repos/data-nih/tcia/releases/tags/upenn-gbm"
OUT_DIR = Path("data/raw")

# Subjects confirmed to have segmentation in this release
SUBJECTS_WITH_SEG = {"sub-002", "sub-006", "sub-008", "sub-009", "sub-011", "sub-013", "sub-014"}


def fetch_assets():
    req = urllib.request.Request(RELEASE_API, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    return [(a["name"], a["browser_download_url"]) for a in data["assets"]]


def subject_of(filename: str) -> str:
    parts = filename.split("_")
    return parts[0] if parts[0].startswith("sub-") else ""


def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  [↓]    {dest.name}")
    urllib.request.urlretrieve(url, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg-only", action="store_true",
                        help="Only download subjects that have segmentation masks")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N complete (segmented) subjects")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Fetching asset list from GitHub...")
    assets = fetch_assets()
    print(f"Found {len(assets)} files in release.\n")

    # Decide which subjects to include
    target_subjects = SUBJECTS_WITH_SEG if (args.seg_only or args.n) else None
    if args.n:
        target_subjects = set(sorted(SUBJECTS_WITH_SEG)[: args.n])

    downloaded = 0
    skipped = 0
    for name, url in assets:
        sid = subject_of(name)
        if not sid:
            continue  # e.g. sub-177_dwi.sz — no seg, skip
        if target_subjects and sid not in target_subjects:
            skipped += 1
            continue
        download_file(url, out / name)
        downloaded += 1

    print(f"\nDone. {downloaded} files downloaded to {out}/  ({skipped} skipped)")
    print("Run  python scripts/verify_data.py  to confirm all files are present.")


if __name__ == "__main__":
    main()
