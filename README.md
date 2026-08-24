# RIED-Net: CT → MRI (3D, paired)

This is a fork of the official implementation of [**Deep Residual Inception Encoder-Decoder Network for Amyloid PET Harmonization**](https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.12564) [Alzheimer's &amp; Dementia], adapted to translate **3D CT volumes into MRI volumes** on **paired** NIfTI data, using [MONAI](https://monai.io)'s 3D UNet family instead of the original 2D-slice pipeline.

[Jay Shah](https://www.public.asu.edu/~jgshah1/)<sup>1,2</sup>,
Fei Gao<sup>1,2</sup>,
[Baoxin Li](https://www.public.asu.edu/~bli24/)<sup>1,2</sup>,
Valentina Ghisays<sup>3</sup>,
Ji Luo<sup>3</sup>,
Yinghua Chen<sup>3</sup>,
Wendy Lee<sup>3</sup>,
Yuxiang Zhou<sup>4</sup>,
[Tammie L.S. Benzinger](https://scholar.google.com/citations?user=fr-fkIwAAAAJ&hl=en)<sup>5</sup>,
[Eric M. Reiman](https://scholar.google.com/citations?user=I-Khl7AAAAAJ&hl=en)<sup>3</sup>,
[Kewei Chen](https://scholar.google.com/citations?user=d83ZIzEAAAAJ&hl=en)<sup>3</sup>,
[Yi Su](https://scholar.google.com/citations?user=vdZKSEIAAAAJ&hl=en)<sup>1,2,3</sup>,
[Teresa Wu](https://labs.engineering.asu.edu/wulab/person/teresa-wu-2/)<sup>1,2</sup>

<sup>1</sup>ASU-Mayo Center for Innovative Imaging,
<sup>2</sup>Arizona State University,
<sup>3</sup>Banner Alzheimer's Institute,
<sup>4</sup>Dept of Radiology, Mayo Clinic, Arizona,
<sup>5</sup>Mallinckrodt Inst. of Radiology, Washington University

---

<p align="center">
<img src="imgs/ried_results.jpg" width=62% height=62%
class="center">
</p>
Visual comparison of synthetic images generated using RIED-Net to real Pittsburgh Compound-B (PIB) data for the OASIS (A &amp; B) and GAAIN (C &amp; D) datasets, from the original amyloid-PET work this codebase is based on.

## What changed in this fork

- **3D, not 2D-slice-based.** The original repo looped over 2D axial slices; this fork feeds whole 3D volumes (trained on random 3D patches, evaluated with sliding-window inference) through MONAI's `spatial_dims=3` `UNet` / `BasicUNet` / `AttentionUnet`.
- **CT → MRI instead of PET tracer harmonization.** Input is CT, target is MRI.
- **Two separate source folders, matched automatically.** You keep your CT and MRI volumes in two different folders (they don't need matching filenames) and `prepare_data.py` pairs them up.
- **Paired augmentation bug fixed.** The original `data_loaders.py` built a MONAI transform pipeline but never actually applied it to the returned tensors, and (had it been wired up) applied fully independent randomness to the image and target, which would have misaligned them. This fork applies geometric augmentations (crop/flip/rotate) identically to the CT/MRI pair via MONAI dictionary transforms, and restricts intensity augmentation (noise, bias field, contrast, etc.) to the CT input only, so the MRI target stays a clean label.
- **No TensorFlow dependency.** `logger.py` now uses `torch.utils.tensorboard` instead of `tensorflow`.
- **`generate.py` also scores outputs** (MAE/PSNR/SSIM) against the real paired MRI when the split provides one, and writes a `metrics_<split>.csv`.

## Quick start
```bash
./run.sh            # prepare -> train -> generate, all in one go, on the demo data
./run.sh prepare    # or run steps individually:
./run.sh train
./run.sh generate
```
`run.sh` handles the `module load` / `source activate` for you. Override any default via env vars, e.g.:
```bash
CT_DIR=/path/to/real/CT MRI_DIR=/path/to/real/MRI EPOCHS=200 MODEL_NAME=attunet ./run.sh
```
See the top of [run.sh](run.sh) for the full list of overridable variables.

## Installation

```bash
module load mamba/latest
source activate fazle_env        # or: mamba create -n riednet -c conda-forge python=3.11
pip install torch monai nibabel pandas scikit-image tensorboard
```
Packages used in the current version of this code:
```
monai==1.6.0
torch==2.12.0+cu132
nibabel==5.4.2
pandas
tensorboard
```

## Dataset

Put your paired CT and MRI 3D NIfTI volumes in two separate folders (any filenames — they don't need to match). This repo ships a 5-case demo pair set (BraTS GLI volumes standing in for CT/MRI) at:
```
CT/    # 5 demo *.nii.gz volumes (stand-in for CT)
MRI/   # 5 demo *.nii.gz volumes, paired 1:1 with CT/ by case
```
`prepare_data.py` matches files between the two folders (exact filename first, then greedy filename-similarity matching — e.g. `BraTS-GLI-00033-000-t2f.nii.gz` <-> `BraTS-GLI-00033-000-t1c.nii.gz`), then splits into train/val/test folds. It does **not** copy or symlink volumes — each fold is just a CSV of absolute paths, so your original CT/MRI folders are untouched:

```bash
python prepare_data.py --ct_dir ./CT --mri_dir ./MRI --out_dir ./dataset \
    --fold_name fold1 --val_frac 0.2 --test_frac 0.2 --seed 42
```
produces
```
dataset/fold1/
  ├── train/pairs.csv   (image=CT path, target=MRI path, case_id)
  ├── val/pairs.csv
  ├── test/pairs.csv
  └── split_summary.json
```

## Training
```bash
python train.py --dataset ./dataset/fold1 --epochs 100 --batch_size 1 --model_name resunet
```
Key options:
- `--model_name {unet,resunet,attunet}` — MONAI `BasicUNet` / `UNet` (residual) / `AttentionUnet`, all 3D.
- `--patch_size 96 96 96` — random 3D patch size used for training (GPU memory + augmentation diversity); validation/test always run full-volume via sliding-window inference.
- `--target_shape 240 240 160` — canonical shape every volume is pad/cropped to before patching. Match this to your data (must be ≥ `--patch_size` in every dimension).
- `--ct_normalize {percentile,hu_window}` — `percentile` (default) is a robust per-volume min-max normalization that works for any input, including the demo data (which isn't literally in Hounsfield units). Use `hu_window` (with `--ct_window lo hi`, e.g. `-1000 1000`) for real clinical CT.
- `--loss {l1_ssim,l1}` — default combines L1 with an SSIM term for structural sharpness.

Checkpoints (`latest.ckpt`, `best.ckpt` by val SSIM, plus periodic snapshots every `--save_freq` epochs) and TensorBoard logs are written under `./runs/<model_name>_<timestamp>/`.

## Evaluation / Generation
```bash
python generate.py --model_name resunet --checkpoint ./runs/resunet_.../weights/best.ckpt \
    --dataset ./dataset/fold1 --split test --output_dir ./results
```
Writes `results/syn_<case>.nii.gz` synthetic MRI volumes (CT's affine/header, in the padded/cropped canonical shape) and, since the `test` split has paired real MRI, `results/metrics_test.csv` with per-case MAE/PSNR/SSIM.

## Citation
Please consider citing RIED-Net if this repository is useful for your work.
```
@article{shah2022deep,
  title={Deep residual inception encoder-decoder network for amyloid PET harmonization},
  author={Shah, Jay and Gao, Fei and Li, Baoxin and Ghisays, Valentina and Luo, Ji and Chen, Yinghua and Lee, Wendy and Zhou, Yuxiang and Benzinger, Tammie LS and Reiman, Eric M and others},
  journal={Alzheimer's \& Dementia},
  volume={18},
  number={12},
  pages={2448--2457},
  year={2022},
  publisher={Wiley Online Library}
}
```
## Acknowledgments
This research has been supported partially by NIH grants R01AG031581, R01AG069453, P30AG019610, and Arizona Department of Health Services (ADHS) and the State of Arizona, ADHS Grant No. CTR040636. This is a [patent-pending](https://patentscope.wipo.int/search/en/detail.jsf?docId=WO2023101959&_cid=P21-LPD5OU-06483-1) technology.
