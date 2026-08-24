"""Train RIED-Net (as a MONAI 3D UNet/AttentionUnet) to translate CT -> MRI
on paired 3D NIfTI volumes.

Volumes are trained on random 3D patches (for GPU memory and augmentation
diversity) and validated on the full volume via sliding-window inference.
"""
import argparse
import datetime
import json
import os

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet, BasicUNet, UNet

from calculate_metrics import PSNR, SSIM3D
from data_loaders import get_loader
from logger import Logger


def get_n_params(model):
    return sum(p.numel() for p in model.parameters())


def build_model(model_name, device):
    if model_name == "unet":
        model = BasicUNet(
            spatial_dims=3,
            features=(16, 32, 64, 128, 256, 32),
            in_channels=1,
            out_channels=1,
        )
    elif model_name == "resunet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif model_name == "attunet":
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    return model.to(device)


def make_criterion(loss_name, device):
    l1 = torch.nn.L1Loss()
    if loss_name == "l1":
        return lambda y_hat, y: l1(y_hat, y)

    ssim = SSIM3D(window_size=7).to(device)

    def l1_ssim(y_hat, y):
        return l1(y_hat, y) + 0.5 * (1 - ssim(y_hat, y))

    return l1_ssim


def train_one_epoch(model, loader, criterion, opt, epoch, device):
    model.train()
    losses = []
    for it, (x, y, _, _) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        print(f"Epoch {epoch + 1}, Iter {it + 1}/{len(loader)}: loss = {np.mean(losses):.4f}")

    return {"loss": losses}


def evaluate(model, loader, criterion, criterion_mse, patch_size, device, epoch, mode="val"):
    model.eval()
    ssim_fn = SSIM3D(window_size=7).to(device)
    metrics = {"loss": [], "psnr": [], "ssim": []}

    with torch.no_grad():
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            y_hat = sliding_window_inference(x, roi_size=patch_size, sw_batch_size=1,
                                              predictor=model, overlap=0.25)

            loss = criterion(y_hat, y)
            data_range = (torch.max(y) - torch.min(y)).item()
            psnr = PSNR(y, y_hat, criterion_mse, data_range).item()
            ssim = ssim_fn(y, y_hat).item()

            metrics["loss"].append(loss.item())
            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)

    log = f"{mode.upper()} @ epoch {epoch + 1}: "
    log += " ".join(f"{k}={np.mean(v):.4f}" for k, v in metrics.items())
    print(log)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="RIED-Net 3D CT -> MRI translation")
    parser.add_argument("-d", "--dataset", default="./dataset/fold1", type=str,
                         help="Fold directory produced by prepare_data.py "
                              "(contains train/pairs.csv, val/pairs.csv, test/pairs.csv)")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=1, type=int,
                         help="Increase once your dataset has more than a handful of cases "
                              "(kept at 1 by default so tiny demo folds don't drop samples)")
    parser.add_argument("--model_name", default="resunet", type=str, choices=["unet", "resunet", "attunet"])
    parser.add_argument("--loss", default="l1_ssim", type=str, choices=["l1", "l1_ssim"])
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--patch_size", default=(96, 96, 96), type=int, nargs=3)
    parser.add_argument("--target_shape", default=(240, 240, 160), type=int, nargs=3,
                         help="Canonical shape all volumes are pad/cropped to before patching")
    parser.add_argument("--ct_normalize", default="percentile", choices=["percentile", "hu_window"],
                         help="How to normalize the CT input. Use hu_window for real clinical CT in HU.")
    parser.add_argument("--ct_window", default=(-1000.0, 1000.0), type=float, nargs=2)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--save_freq", default=10, type=int, help="Also snapshot every N epochs")
    parser.add_argument("--output_dir", default="./runs", type=str)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m%d%y%H%M%S")
    training_folder = os.path.join(args.output_dir, f"{args.model_name}_{timestamp}")

    os.makedirs(os.path.join(training_folder, "weights"), exist_ok=True)
    with open(os.path.join(training_folder, "params.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    logger = Logger(os.path.join(training_folder, "logs"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model_name, device)
    print(args)
    print("Model param:", get_n_params(model))
    print("Working on", args.dataset, "| device:", device)

    torch.backends.cudnn.benchmark = True

    dataset_train = get_loader(
        os.path.join(args.dataset, "train", "pairs.csv"), mode="train",
        batch_size=args.batch_size, patch_size=tuple(args.patch_size),
        target_shape=tuple(args.target_shape), ct_normalize=args.ct_normalize,
        ct_window=tuple(args.ct_window), num_workers=args.num_workers,
    )
    dataset_val = get_loader(
        os.path.join(args.dataset, "val", "pairs.csv"), mode="val",
        batch_size=1, target_shape=tuple(args.target_shape),
        ct_normalize=args.ct_normalize, ct_window=tuple(args.ct_window),
        num_workers=args.num_workers,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = make_criterion(args.loss, device)
    criterion_mse = torch.nn.MSELoss()

    best_ssim = -1.0
    for e in range(args.epochs):
        loss_metrics = train_one_epoch(model, dataset_train, criterion, opt, e, device)
        for cnt, v in enumerate(loss_metrics["loss"]):
            logger.scalar_summary("train/loss", v, e * len(dataset_train) + cnt + 1)

        val_metrics = evaluate(model, dataset_val, criterion, criterion_mse,
                                tuple(args.patch_size), device, e, mode="val")
        for tag, values in val_metrics.items():
            logger.scalar_summary(f"val/{tag}", np.mean(values), e + 1)

        latest_path = os.path.join(training_folder, "weights", "latest.ckpt")
        torch.save(model.state_dict(), latest_path)

        if args.save_freq and (e + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(training_folder, "weights", f"epoch{e + 1}.ckpt"))

        mean_ssim = float(np.mean(val_metrics["ssim"]))
        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            torch.save(model.state_dict(), os.path.join(training_folder, "weights", "best.ckpt"))
            print(f"New best val SSIM = {best_ssim:.4f} -> saved best.ckpt")

    print(f"Training complete. Checkpoints in {os.path.join(training_folder, 'weights')}")


if __name__ == "__main__":
    main()
