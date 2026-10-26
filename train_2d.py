"""Train RIED-Net slice-wise (2.5D) to translate CT -> MRI, then rebuild 3D
volumes from the predicted slices.

Independent of the 3D pipeline in ``train.py``: separate model builder,
checkpoints (default ``./runs/2d``) and CLI. The network sees a stack of
2*context+1 neighbouring CT slices and predicts out_slices neighbouring MRI
slices. Validation reconstructs full volumes (see
``data_loaders_2d.predict_volume``) and scores them in 3D, so the numbers are
directly comparable with the 3D model. Per-view SSIM (sagittal/coronal/axial)
and the through-plane error expose slice-to-slice flicker/banding.
"""
import argparse
import datetime
import json
import os

import numpy as np
import torch
from monai.networks.nets import AttentionUnet, BasicUNet, UNet

from calculate_metrics import PSNR, SSIM, SSIM3D
from data_loaders_2d import AXIS_NAMES, get_slice_loader, load_cases, predict_volume
from logger import Logger


def get_n_params(model):
    return sum(p.numel() for p in model.parameters())


def build_model_2d(model_name, in_channels, out_channels, device):
    if model_name == "unet":
        model = BasicUNet(
            spatial_dims=2,
            features=(32, 64, 128, 256, 512, 32),
            in_channels=in_channels,
            out_channels=out_channels,
        )
    elif model_name == "resunet":
        model = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif model_name == "attunet":
        model = AttentionUnet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    return model.to(device)


def through_plane_loss(y_hat, y):
    """L1 between the slice-to-slice differences of the prediction and of the
    real MRI. Penalizes jumps between neighbouring predicted slices that the
    real scan doesn't have, without blurring real anatomical edges.
    y_hat, y: (B, S, H, W) with S consecutive slices in order."""
    d_hat = y_hat[:, 1:] - y_hat[:, :-1]
    d = y[:, 1:] - y[:, :-1]
    return (d_hat - d).abs().mean()


def make_criterion(loss_name, tp_weight, device):
    l1 = torch.nn.L1Loss()
    ssim = SSIM(window_size=7).to(device)

    def criterion(y_hat, y):
        loss = l1(y_hat, y)
        if loss_name == "l1_ssim":
            loss = loss + 0.5 * (1 - ssim(y_hat, y))
        if tp_weight > 0:
            loss = loss + tp_weight * through_plane_loss(y_hat, y)
        return loss

    return criterion


def volume_metrics(y_hat, y, device):
    """3D MAE/PSNR/SSIM on a reconstructed volume, plus per-view 2D SSIM and
    the through-plane error along each axis (flicker/banding indicators).
    y_hat, y: numpy arrays of identical shape."""
    y_hat_t = torch.from_numpy(y_hat).to(device)
    y_t = torch.from_numpy(y).to(device)
    data_range = (y_t.max() - y_t.min()).item()

    m = {
        "mae": torch.nn.functional.l1_loss(y_hat_t, y_t).item(),
        "psnr": PSNR(y_t, y_hat_t, torch.nn.MSELoss(), data_range).item(),
        "ssim": SSIM3D(window_size=7).to(device)(y_t[None, None], y_hat_t[None, None]).item(),
    }
    ssim2d = SSIM(window_size=7).to(device)
    for axis, view in AXIS_NAMES.items():
        a = y_t.movedim(axis, 0)[:, None]
        b = y_hat_t.movedim(axis, 0)[:, None]
        m[f"ssim_{view}"] = ssim2d(a, b).item()
        m[f"tp_err_{view}"] = through_plane_loss(b.transpose(0, 1), a.transpose(0, 1)).item()
    return m


def train_one_epoch(model, loader, criterion, opt, epoch, device):
    model.train()
    losses = []
    for it, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if (it + 1) % 20 == 0 or it + 1 == len(loader):
            print(f"Epoch {epoch + 1}, Iter {it + 1}/{len(loader)}: loss = {np.mean(losses):.4f}")

    return {"loss": losses}


def evaluate(model, cases, axes, context, out_slices, device, epoch, infer_batch=16, mode="val"):
    metrics = {}
    for case in cases:
        y_hat = predict_volume(model, case["ct"], axes, context, out_slices, device, infer_batch)
        for k, v in volume_metrics(y_hat, case["mri"], device).items():
            metrics.setdefault(k, []).append(v)

    log = f"{mode.upper()} @ epoch {epoch + 1}: "
    log += " ".join(f"{k}={np.mean(v):.4f}" for k, v in metrics.items())
    print(log)
    return metrics


def save_checkpoint(path, model, config):
    # Config travels with the weights so generate_2d.py rebuilds the exact
    # same architecture/slice setup without having to repeat CLI flags.
    torch.save({"model": model.state_dict(), "config": config}, path)


def main():
    parser = argparse.ArgumentParser(description="RIED-Net 2.5D slice-wise CT -> MRI translation")
    parser.add_argument("-d", "--dataset", default="./dataset/fold1", type=str,
                         help="Fold directory produced by prepare_data.py "
                              "(contains train/pairs.csv, val/pairs.csv, test/pairs.csv)")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=16, type=int, help="Slices per batch")
    parser.add_argument("--model_name", default="resunet", type=str, choices=["unet", "resunet", "attunet"])
    parser.add_argument("--loss", default="l1_ssim", type=str, choices=["l1", "l1_ssim"])
    parser.add_argument("--tp_weight", default=0.0, type=float,
                         help="Weight of the through-plane consistency loss (0 = off; try 0.1-0.5 "
                              "if coronal/sagittal views show banding). Needs --out_slices >= 2.")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--context", default=2, type=int,
                         help="Neighbouring CT slices on each side fed as input channels (2 -> 5 slices)")
    parser.add_argument("--out_slices", default=3, type=int,
                         help="Neighbouring MRI slices predicted per step; overlaps are averaged at inference")
    parser.add_argument("--axes", default=(2,), type=int, nargs="+", choices=[0, 1, 2],
                         help="Slicing axes in canonical RAS order: 0=sagittal 1=coronal 2=axial. "
                              "Several axes = one model trained on all views, fused at inference.")
    parser.add_argument("--slice_size", default=(256, 256), type=int, nargs=2,
                         help="Training slices are padded/randomly cropped to this size")
    parser.add_argument("--min_foreground", default=0.01, type=float,
                         help="Skip training slices whose foreground fraction is below this (0 = keep all)")
    parser.add_argument("--ct_normalize", default="percentile", choices=["percentile", "hu_window"],
                         help="How to normalize the CT input. Use hu_window for real clinical CT in HU.")
    parser.add_argument("--ct_window", default=(-1000.0, 1000.0), type=float, nargs=2)
    parser.add_argument("--infer_batch", default=16, type=int, help="Slices per forward pass at validation")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--val_freq", default=1, type=int, help="Validate every N epochs")
    parser.add_argument("--save_freq", default=10, type=int, help="Also snapshot every N epochs")
    parser.add_argument("--output_dir", default="./runs/2d", type=str)
    args = parser.parse_args()

    if args.tp_weight > 0 and args.out_slices < 2:
        parser.error("--tp_weight > 0 needs --out_slices >= 2 (the loss compares neighbouring output slices)")
    axes = tuple(sorted(set(args.axes)))

    timestamp = datetime.datetime.now().strftime("%m%d%y%H%M%S")
    training_folder = os.path.join(args.output_dir, f"{args.model_name}_{timestamp}")

    os.makedirs(os.path.join(training_folder, "weights"), exist_ok=True)
    with open(os.path.join(training_folder, "params.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    logger = Logger(os.path.join(training_folder, "logs"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "model_name": args.model_name,
        "context": args.context,
        "out_slices": args.out_slices,
        "axes": list(axes),
        "ct_normalize": args.ct_normalize,
        "ct_window": list(args.ct_window),
    }
    model = build_model_2d(args.model_name, 2 * args.context + 1, args.out_slices, device)
    print(args)
    print("Model param:", get_n_params(model))
    print("Working on", args.dataset, "| device:", device)

    torch.backends.cudnn.benchmark = True

    dataset_train = get_slice_loader(
        os.path.join(args.dataset, "train", "pairs.csv"), axes=axes, context=args.context,
        out_slices=args.out_slices, slice_size=tuple(args.slice_size), batch_size=args.batch_size,
        min_foreground=args.min_foreground, ct_normalize=args.ct_normalize,
        ct_window=tuple(args.ct_window), num_workers=args.num_workers,
    )
    val_cases = load_cases(os.path.join(args.dataset, "val", "pairs.csv"),
                           args.ct_normalize, tuple(args.ct_window))
    print(f"Train slices: {len(dataset_train.dataset)} | val volumes: {len(val_cases)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = make_criterion(args.loss, args.tp_weight, device)

    best_ssim = -1.0
    for e in range(args.epochs):
        loss_metrics = train_one_epoch(model, dataset_train, criterion, opt, e, device)
        for cnt, v in enumerate(loss_metrics["loss"]):
            logger.scalar_summary("train/loss", v, e * len(dataset_train) + cnt + 1)

        save_checkpoint(os.path.join(training_folder, "weights", "latest.ckpt"), model, config)
        if args.save_freq and (e + 1) % args.save_freq == 0:
            save_checkpoint(os.path.join(training_folder, "weights", f"epoch{e + 1}.ckpt"), model, config)

        if (e + 1) % args.val_freq != 0 and e + 1 != args.epochs:
            continue

        val_metrics = evaluate(model, val_cases, axes, args.context, args.out_slices,
                               device, e, args.infer_batch, mode="val")
        for tag, values in val_metrics.items():
            logger.scalar_summary(f"val/{tag}", np.mean(values), e + 1)

        mean_ssim = float(np.mean(val_metrics["ssim"]))
        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            save_checkpoint(os.path.join(training_folder, "weights", "best.ckpt"), model, config)
            print(f"New best val SSIM = {best_ssim:.4f} -> saved best.ckpt")

    print(f"Training complete. Checkpoints in {os.path.join(training_folder, 'weights')}")


if __name__ == "__main__":
    main()
