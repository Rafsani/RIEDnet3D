"""2.5D slice-wise data pipeline for CT -> MRI translation.

Independent of the 3D pipeline in ``data_loaders.py`` (only the intensity
normalization helpers are shared). Reads the same ``pairs.csv`` splits
produced by ``prepare_data.py``.

How slice order is kept
-----------------------
Every volume is reoriented to the closest canonical (RAS) orientation on load,
so the axis order is identical for every case:

    axis 0 -> sagittal slices, axis 1 -> coronal slices, axis 2 -> axial slices

A training sample is addressed as (case, axis, k). The CT input is the stack of
slices k-context..k+context and the MRI target is the stack k-o..k+o
(o = out_slices // 2), both taken at the same indices of the same volume, so
the pair is aligned by construction. Indices past the volume edge are clamped
(the border slice is repeated).

At inference ``predict_volume`` loops k = 0..D-1 and writes each predicted
slice back into position k of a preallocated output volume. Because the
network predicts ``out_slices`` neighbouring slices, every position is
predicted several times and the predictions are averaged, which smooths the
volume along the slice axis. With several axes (multi-view) the per-axis
reconstructions are averaged too.
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai import transforms as T
from torch.utils import data

from data_loaders import normalize_hu_window, normalize_percentile

AXIS_NAMES = {0: "sagittal", 1: "coronal", 2: "axial"}


def load_canonical(path):
    """Load a NIfTI volume reoriented to closest-canonical (RAS) as float32,
    plus the reoriented nibabel image (its affine/header are what synthetic
    outputs are saved with)."""
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    return vol, img


def slice_offsets(n):
    """Offsets of an n-slice stack centred on the current slice."""
    half = n // 2
    return np.arange(-half, n - half)


def take_stack(vol, axis, k, offsets):
    """Return slices k+offsets of ``vol`` along ``axis`` as a (len(offsets), H, W)
    array, clamping indices at the volume border."""
    n = vol.shape[axis]
    idx = np.clip(k + offsets, 0, n - 1)
    return np.moveaxis(np.take(vol, idx, axis=axis), axis, 0)


def load_cases(pairs_csv, ct_normalize="percentile", ct_window=(-1000.0, 1000.0)):
    """Load every (CT, MRI) pair in a pairs.csv into memory, normalized per
    volume (never per slice, so intensities stay consistent across slices)."""
    pairs = pd.read_csv(pairs_csv)
    cases = []
    for _, row in pairs.iterrows():
        ct_vol, ct_img = load_canonical(row["image"])
        if ct_normalize == "hu_window":
            ct_vol = normalize_hu_window(ct_vol, *ct_window)
        else:
            ct_vol = normalize_percentile(ct_vol)

        mri_vol = None
        if "target" in row and isinstance(row["target"], str) and row["target"]:
            mri_vol, _ = load_canonical(row["target"])
            if mri_vol.shape != ct_vol.shape:
                raise ValueError(f"CT/MRI shape mismatch for {row['image']}: "
                                 f"{ct_vol.shape} vs {mri_vol.shape}. Volumes must be co-registered.")
            mri_vol = normalize_percentile(mri_vol)

        cases.append({
            "name": os.path.basename(row["image"]).split(".")[0],
            "ct_path": row["image"],
            "ct": ct_vol,
            "mri": mri_vol,
            "affine": ct_img.affine,
            "header": ct_img.header,
        })
    return cases


class PairedSliceDataset(data.Dataset):
    """2.5D training samples: (2*context+1)-slice CT stack -> out_slices MRI stack."""

    def __init__(self, pairs_csv, axes=(2,), context=2, out_slices=3, transform=None,
                 min_foreground=0.01, ct_normalize="percentile", ct_window=(-1000.0, 1000.0)):
        self.cases = load_cases(pairs_csv, ct_normalize, ct_window)
        self.in_offsets = slice_offsets(2 * context + 1)
        self.out_offsets = slice_offsets(out_slices)
        self.transform = transform

        # Index every slice position, skipping (near-)empty background slices
        # so the model doesn't spend most of its updates learning to output zeros.
        self.index = []
        for c, case in enumerate(self.cases):
            fg = (case["ct"] > 0.05) | (case["mri"] > 0.05)
            for axis in axes:
                other = tuple(a for a in range(3) if a != axis)
                fg_frac = fg.mean(axis=other)
                for k in range(case["ct"].shape[axis]):
                    if fg_frac[k] >= min_foreground:
                        self.index.append((c, axis, k))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        c, axis, k = self.index[i]
        case = self.cases[c]
        sample = {
            "image": torch.from_numpy(take_stack(case["ct"], axis, k, self.in_offsets).copy()),
            "target": torch.from_numpy(take_stack(case["mri"], axis, k, self.out_offsets).copy()),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample["image"].float(), sample["target"].float()


def get_train_transforms_2d(slice_size):
    """In-plane geometric augmentation applied identically to input and target
    stacks; intensity augmentation on the CT input only. Slices are padded /
    randomly cropped to ``slice_size`` so stacks from any axis can be batched."""
    keys = ["image", "target"]
    tfms = [
        T.SpatialPadd(keys=keys, spatial_size=slice_size),
        T.RandSpatialCropd(keys=keys, roi_size=slice_size, random_size=False),
        T.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        T.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
    ]
    if slice_size[0] == slice_size[1]:
        tfms.append(T.RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)))
    tfms += [
        T.RandGaussianNoised(keys=["image"], prob=0.2, std=0.02),
        T.RandStdShiftIntensityd(keys=["image"], prob=0.2, factors=0.1),
        T.RandScaleIntensityd(keys=["image"], prob=0.2, factors=0.1),
        T.RandAdjustContrastd(keys=["image"], prob=0.2),
        T.RandBiasFieldd(keys=["image"], prob=0.15),
        T.RandGaussianSmoothd(keys=["image"], prob=0.15),
    ]
    return T.Compose(tfms)


def get_slice_loader(pairs_csv, axes=(2,), context=2, out_slices=3, slice_size=(256, 256),
                     batch_size=16, min_foreground=0.01, ct_normalize="percentile",
                     ct_window=(-1000.0, 1000.0), num_workers=4):
    dataset = PairedSliceDataset(
        pairs_csv, axes=axes, context=context, out_slices=out_slices,
        transform=get_train_transforms_2d(slice_size), min_foreground=min_foreground,
        ct_normalize=ct_normalize, ct_window=ct_window,
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) > batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def predict_volume(model, ct_vol, axes, context, out_slices, device, batch_size=16, divisor=16):
    """Synthesize a full volume slice by slice and reassemble it in order.

    For each axis: slide over k = 0..D-1, predict ``out_slices`` neighbouring
    MRI slices from the CT stack around k, and add each prediction into its
    slice position of an accumulator. Positions predicted more than once are
    averaged; the per-axis volumes are then averaged (multi-view fusion).
    Returns a numpy array with the same shape/order as ``ct_vol``.
    """
    model.eval()
    ct = torch.from_numpy(ct_vol).to(device)
    in_off = torch.as_tensor(slice_offsets(2 * context + 1), device=device)
    out_off = slice_offsets(out_slices)
    fused = torch.zeros_like(ct)

    for axis in axes:
        vol = ct.movedim(axis, 0)  # (D, H, W), slice index first
        D, H, W = vol.shape
        pad_h, pad_w = (-H) % divisor, (-W) % divisor
        acc = torch.zeros_like(vol)
        cnt = torch.zeros(D, device=device)

        for start in range(0, D, batch_size):
            ks = torch.arange(start, min(start + batch_size, D), device=device)
            idx = (ks[:, None] + in_off[None, :]).clamp(0, D - 1)
            x = F.pad(vol[idx], (0, pad_w, 0, pad_h))  # (B, C_in, H', W')
            y = model(x)[..., :H, :W]  # (B, out_slices, H, W)

            for j, o in enumerate(out_off):
                t = ks + int(o)
                valid = (t >= 0) & (t < D)
                acc.index_add_(0, t[valid], y[valid, j])
                cnt.index_add_(0, t[valid], torch.ones(int(valid.sum()), device=device))

        fused += (acc / cnt[:, None, None]).movedim(0, axis)

    return (fused / len(axes)).cpu().numpy()


if __name__ == "__main__":
    loader = get_slice_loader("./dataset/fold1/train/pairs.csv", axes=(2,), batch_size=4, num_workers=0)
    print("slices:", len(loader.dataset))
    for x, y in loader:
        print(x.shape, y.shape, x.min().item(), x.max().item())
        break
