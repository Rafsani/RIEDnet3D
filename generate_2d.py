"""Synthesize MRI volumes from CT with a slice-wise (2.5D) checkpoint from
train_2d.py, reassembling the predicted slices into 3D volumes, and score them
against the paired real MRI when the split provides one.

Outputs are saved in the CT's own voxel grid (no pad/crop), in closest-
canonical (RAS) orientation with the matching affine, so they overlay
correctly on the source CT in any NIfTI viewer.
"""
import argparse
import csv
import os

import nibabel as nib
import numpy as np
import torch

from data_loaders_2d import AXIS_NAMES, load_cases, predict_volume
from train_2d import build_model_2d, volume_metrics


def main():
    parser = argparse.ArgumentParser(description="RIED-Net 2.5D slice-wise CT -> MRI generation")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to a .ckpt saved by train_2d.py")
    parser.add_argument("-d", "--dataset", default="./dataset/fold1", type=str,
                         help="Fold directory produced by prepare_data.py")
    parser.add_argument("--split", default="test", type=str, choices=["train", "val", "test"])
    parser.add_argument("--axes", default=None, type=int, nargs="+", choices=[0, 1, 2],
                         help="Override the slicing axes stored in the checkpoint "
                              "(e.g. 0 1 2 for multi-view fusion with a model trained on all views)")
    parser.add_argument("--infer_batch", default=16, type=int)
    parser.add_argument("--output_dir", default="./results/2d", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    axes = tuple(sorted(set(args.axes))) if args.axes else tuple(cfg["axes"])

    model = build_model_2d(cfg["model_name"], 2 * cfg["context"] + 1, cfg["out_slices"], device)
    model.load_state_dict(ckpt["model"])
    print("Config:", cfg, "| inference axes:", [AXIS_NAMES[a] for a in axes])

    cases = load_cases(os.path.join(args.dataset, args.split, "pairs.csv"),
                       cfg["ct_normalize"], tuple(cfg["ct_window"]))

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for case in cases:
        y_hat = predict_volume(model, case["ct"], axes, cfg["context"], cfg["out_slices"],
                               device, args.infer_batch)
        out_path = os.path.join(args.output_dir, f"syn_{case['name']}.nii.gz")
        nib.save(nib.Nifti1Image(y_hat.astype(np.float32), case["affine"], case["header"]), out_path)

        row = {"case": case["name"], "output": out_path}
        if case["mri"] is not None:
            row.update(volume_metrics(y_hat, case["mri"], device))
            print(f"{case['name']}: " + " ".join(f"{k}={v:.4f}" for k, v in row.items()
                                                 if k not in ("case", "output")))
        else:
            print(f"{case['name']}: saved {out_path}")
        rows.append(row)

    if rows and "ssim" in rows[0]:
        metrics_csv = os.path.join(args.output_dir, f"metrics_{args.split}.csv")
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        keys = [k for k in rows[0] if k not in ("case", "output")]
        print("\nMean " + " ".join(f"{k}={np.mean([r[k] for r in rows]):.4f}" for k in keys))
        print(f"Per-case metrics written to {metrics_csv}")


if __name__ == "__main__":
    main()
