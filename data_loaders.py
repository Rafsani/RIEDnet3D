"""Paired 3D NIfTI dataset for CT -> MRI translation.

Each split (train/val/test) is described by a ``pairs.csv`` produced by
``prepare_data.py`` with columns ``image`` (CT path), ``target`` (MRI path)
and ``case_id``. Volumes are loaded whole, intensity-normalized per-volume,
and resized (pad/crop) to a common canonical shape so batches can be
stacked. Geometric augmentations (crop/flip/rotate) are applied identically
to the CT/MRI pair via MONAI's dictionary transforms so the pair stays
spatially aligned; intensity augmentations are applied to the CT input only,
so the MRI target always stays a clean ground truth.
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai import transforms as T
from torch.utils import data


def load_nifti(path):
    """Load a NIfTI volume as a float32 numpy array, plus the nibabel image
    (kept around so callers can reuse its affine/header, e.g. when saving
    synthetic output)."""
    img = nib.load(path)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    return vol, img


def normalize_percentile(vol, lo_pct=0.5, hi_pct=99.5, eps=1e-6):
    """Robust per-volume min-max normalization to [0, 1] using percentile
    clipping (robust to hot/cold outlier voxels). Works for any modality,
    including demo data that isn't literally in Hounsfield units."""
    lo, hi = np.percentile(vol, (lo_pct, hi_pct))
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / max(hi - lo, eps)).astype(np.float32)


def normalize_hu_window(vol, lo=-1000.0, hi=1000.0):
    """Clinical CT windowing: clip to a Hounsfield-unit window then scale to
    [0, 1]. Use this (via --ct_normalize hu_window) for real CT data where
    HU values are meaningful; the demo BraTS stand-in data is not real CT,
    so it defaults to percentile normalization instead."""
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo)).astype(np.float32)


class PairedNiftiDataset(data.Dataset):
    """Loads paired (CT, MRI) 3D volumes listed in a pairs.csv."""

    def __init__(self, pairs_csv, transform=None, target_shape=(240, 240, 160),
                 ct_normalize="percentile", ct_window=(-1000.0, 1000.0)):
        self.pairs = pd.read_csv(pairs_csv)
        self.transform = transform
        self.target_shape = tuple(target_shape)
        self.ct_normalize = ct_normalize
        self.ct_window = ct_window
        self._pad_crop = T.ResizeWithPadOrCrop(spatial_size=self.target_shape)

    def __len__(self):
        return len(self.pairs)

    def _normalize_ct(self, vol):
        if self.ct_normalize == "hu_window":
            return normalize_hu_window(vol, *self.ct_window)
        return normalize_percentile(vol)

    def __getitem__(self, index):
        row = self.pairs.iloc[index]
        ct_path, mri_path = row["image"], row["target"]

        ct_vol, _ = load_nifti(ct_path)
        mri_vol, _ = load_nifti(mri_path)

        ct_vol = self._normalize_ct(ct_vol)
        mri_vol = normalize_percentile(mri_vol)

        ct = self._pad_crop(torch.from_numpy(ct_vol).unsqueeze(0))
        mri = self._pad_crop(torch.from_numpy(mri_vol).unsqueeze(0))

        sample = {"image": ct, "target": mri}
        if self.transform is not None:
            sample = self.transform(sample)

        name = os.path.basename(ct_path)
        return sample["image"].float(), sample["target"].float(), name, ct_path


def get_train_transforms(patch_size):
    return T.Compose([
        T.RandSpatialCropd(keys=["image", "target"], roi_size=patch_size, random_size=False),
        T.RandFlipd(keys=["image", "target"], prob=0.5, spatial_axis=0),
        T.RandFlipd(keys=["image", "target"], prob=0.5, spatial_axis=1),
        T.RandFlipd(keys=["image", "target"], prob=0.5, spatial_axis=2),
        T.RandRotate90d(keys=["image", "target"], prob=0.5, spatial_axes=(0, 1)),
        # Intensity-only augmentations perturb the CT input, not the MRI
        # target, so the network still learns to map towards a clean label.
        T.RandGaussianNoised(keys=["image"], prob=0.2, std=0.02),
        T.RandStdShiftIntensityd(keys=["image"], prob=0.2, factors=0.1),
        T.RandScaleIntensityd(keys=["image"], prob=0.2, factors=0.1),
        T.RandAdjustContrastd(keys=["image"], prob=0.2),
        T.RandBiasFieldd(keys=["image"], prob=0.15),
        T.RandGaussianSmoothd(keys=["image"], prob=0.15),
    ])


def get_loader(pairs_csv, mode="train", batch_size=1, patch_size=(96, 96, 96),
               target_shape=(240, 240, 160), ct_normalize="percentile",
               ct_window=(-1000.0, 1000.0), num_workers=4):
    """Build and return a DataLoader over a pairs.csv split."""
    transform = get_train_transforms(patch_size) if mode == "train" else None

    dataset = PairedNiftiDataset(
        pairs_csv,
        transform=transform,
        target_shape=target_shape,
        ct_normalize=ct_normalize,
        ct_window=ct_window,
    )

    drop_last = mode == "train" and len(dataset) > batch_size
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


if __name__ == "__main__":
    loader = get_loader("./dataset/fold1/train/pairs.csv", mode="train", patch_size=(96, 96, 96))
    for ct, mri, name, path in loader:
        print(name, ct.shape, mri.shape, ct.min().item(), ct.max().item())
        break
