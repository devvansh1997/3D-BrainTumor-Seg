# Dataset Setup — BraTS 2021 Task 1

1,251 glioblastoma cases, same 4 modalities (T1, T1ce, T2, FLAIR), same 3 tumor subregions.
Freely available on Kaggle — no registration gatekeeping.

---

## Download (Kaggle CLI — easiest)

```bash
# 1. Install kaggle client
pip install kaggle

# 2. Get your API key:
#    kaggle.com → profile icon → Settings → API → "Create New Token"
#    Move the downloaded kaggle.json to:
#      Windows:  C:\Users\<you>\.kaggle\kaggle.json
#      Linux/Mac: ~/.kaggle/kaggle.json

# 3. Download (~5 GB zip)
kaggle datasets download -d dschettler8845/brats-2021-task1 -p data/

# 4. Unzip into data/raw/
#    Windows PowerShell:
Expand-Archive data/brats-2021-task1.zip -DestinationPath data/raw/
#    Linux/Mac:
unzip data/brats-2021-task1.zip -d data/raw/
```

If the Kaggle slug above doesn't work, search "BraTS 2021 Task 1" on kaggle.com
and use the slug shown in the URL: `kaggle.com/datasets/<slug>`.

---

## Expected folder structure

One sub-folder per patient, all files inside it:

```
data/raw/
  BraTS2021_00000/
    BraTS2021_00000_flair.nii.gz
    BraTS2021_00000_t1.nii.gz
    BraTS2021_00000_t1ce.nii.gz
    BraTS2021_00000_t2.nii.gz
    BraTS2021_00000_seg.nii.gz     ← segmentation mask
  BraTS2021_00001/
    ...
```

---

## Segmentation label map

| Value | Region | Remapped to |
|-------|--------|-------------|
| 0 | Background | 0 |
| 1 | Necrotic / Non-Enhancing Tumor Core (NCR/NET) | 1 |
| 2 | Peritumoral Edema (ED) | 2 |
| **4** | **Enhancing Tumor (ET)** | **3** |

> BraTS skips label 3 — the pipeline remaps 4→3 automatically in `transforms.py`.

---

## Verify

```bash
python scripts/verify_data.py --data-dir data/raw
```

Expected:
```
  [OK]  BraTS2021_00000
  [OK]  BraTS2021_00001
  ...
Summary: 1251 complete | 0 incomplete
```

---

## Run

```bash
# Debug (4 cases, 3 epochs — sanity check)
python run.py --config configs/dynunet.yaml --debug
python run.py --config configs/swinunetr.yaml --debug

# Full (all 1251 cases, 300 epochs)
python run.py --config configs/dynunet.yaml
python run.py --config configs/swinunetr.yaml
```

---

## Dataset stats

| Property | Value |
|---|---|
| Training cases | 1,251 |
| Modalities | T1, T1ce, T2, FLAIR |
| Format | NIfTI (.nii.gz), skull-stripped, co-registered |
| Voxel spacing | 1 mm isotropic |
| Resolution | 240 × 240 × 155 |
| Total size | ~5 GB |
