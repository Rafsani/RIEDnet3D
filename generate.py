"""Run a trained RIED-Net checkpoint to synthesize MRI volumes from CT, and
score them against the paired real MRI when the split provides one (e.g. the
held-out test fold from prepare_data.py).

Output volumes are saved in the padded/cropped canonical shape used during
training (see --target_shape), reusing the source CT's affine/header. If you
need voxel-exact alignment back to the original CT geometry, resample the
synthetic output against the original CT file afterwards.
"""
import argparse
import csv
import os

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from calculate_metrics import MAE, PSNR, SSIM3D
from data_loaders import get_loader
from train import build_model


def run(model, loader, patch_size, device, out_dir, score=True):
    model.eval()
    l1 = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    ssim_fn = SSIM3D(window_size=7).to(device)

    rows = []
    with torch.no_grad():
        for x, y, name, ct_path in loader:
            x = x.to(device)
            y_hat = sliding_window_inference(x, roi_size=patch_size, sw_batch_size=1,
                                              predictor=model, overlap=0.25)

            case_name = name[0].split(".")[0]
            ct_img = nib.load(ct_path[0])
            out_np = y_hat.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            syn_img = nib.Nifti1Image(out_np, ct_img.affine, ct_img.header)
            out_path = os.path.join(out_dir, f"syn_{case_name}.nii.gz")
            nib.save(syn_img, out_path)

            row = {"case": case_name, "output": out_path}
            if score:
                y = y.to(device)
                data_range = (torch.max(y) - torch.min(y)).item()
                row["mae"] = MAE(y_hat, y, l1)
                row["psnr"] = PSNR(y, y_hat, mse, data_range).item()
                row["ssim"] = ssim_fn(y, y_hat).item()
                print(f"{case_name}: MAE={row['mae']:.4f} PSNR={row['psnr']:.2f} SSIM={row['ssim']:.4f}")
            else:
                print(f"{case_name}: saved {out_path}")

            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="RIED-Net 3D CT -> MRI generation")
    parser.add_argument("--model_name", default="resunet", type=str, choices=["unet", "resunet", "attunet"])
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to a .ckpt saved by train.py")
    parser.add_argument("-d", "--dataset", default="./dataset/fold1", type=str,
                         help="Fold directory produced by prepare_data.py")
    parser.add_argument("--split", default="test", type=str, choices=["train", "val", "test"])
    parser.add_argument("--patch_size", default=(96, 96, 96), type=int, nargs=3)
    parser.add_argument("--target_shape", default=(240, 240, 160), type=int, nargs=3)
    parser.add_argument("--ct_normalize", default="percentile", choices=["percentile", "hu_window"])
    parser.add_argument("--ct_window", default=(-1000.0, 1000.0), type=float, nargs=2)
    parser.add_argument("--output_dir", default="./results", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    pairs_csv = os.path.join(args.dataset, args.split, "pairs.csv")
    loader = get_loader(
        pairs_csv, mode=args.split, batch_size=1, target_shape=tuple(args.target_shape),
        ct_normalize=args.ct_normalize, ct_window=tuple(args.ct_window), num_workers=2,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    rows = run(model, loader, tuple(args.patch_size), device, args.output_dir)

    if rows and "ssim" in rows[0]:
        metrics_csv = os.path.join(args.output_dir, f"metrics_{args.split}.csv")
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "output", "mae", "psnr", "ssim"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nMean MAE={np.mean([r['mae'] for r in rows]):.4f} "
              f"PSNR={np.mean([r['psnr'] for r in rows]):.2f} "
              f"SSIM={np.mean([r['ssim'] for r in rows]):.4f}")
        print(f"Per-case metrics written to {metrics_csv}")


if __name__ == "__main__":
    main()
