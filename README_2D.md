# RIED-Net 2.5D (slice-wise) CT → MRI

A slice-wise alternative to the 3D pipeline. A 2D network synthesizes the MRI slice by slice, and the slices are reassembled into a 3D volume. It is fully independent of the 3D code: separate scripts, checkpoints and outputs. It shares only the fold CSVs from `prepare_data.py` and the helper functions in `calculate_metrics.py`, `logger.py` and `data_loaders.py`, which it imports but doesn't change.

| | 3D | 2.5D |
|---|---|---|
| Entry point | `run.sh` | `run_2d.sh` |
| Data / train / generate | `data_loaders.py`, `train.py`, `generate.py` | `data_loaders_2d.py`, `train_2d.py`, `generate_2d.py` |
| Checkpoints | `runs/<model>_<time>/` | `runs/2d/<model>_<time>/` |
| Synthetic volumes | `results/` | `results/2d/` |

## Quick start
```bash
./run_2d.sh                 # prepare -> train -> generate
./run_2d.sh train
./run_2d.sh generate
AXES="0 1 2" TP_WEIGHT=0.2 EPOCHS=200 ./run_2d.sh train   # multi-view + through-plane loss
```

## How slice order and 3D consistency are kept
- **Fixed orientation.** Volumes are reoriented to closest-canonical RAS on load, so axis 0/1/2 means sagittal/coronal/axial for every case.
- **Order by construction.** A training sample is (case, axis, slice k). CT and MRI slices come from the same indices of co-registered volumes. At inference slice k is written back into position k of the output volume, so nothing has to be sorted afterwards.
- **Whole-volume normalization.** Intensities are normalized per volume before slicing, never per slice.
- **2.5D input (`--context`).** The network sees 2·context+1 neighbouring CT slices as channels (default 5).
- **Multi-slice output (`--out_slices`).** The network predicts several neighbouring MRI slices (default 3). Every position is predicted several times and the predictions are averaged.
- **Multi-view fusion (`--axes 0 1 2`).** One model is trained on slices from all three views, and at inference the three rebuilt volumes are averaged. With a multi-view checkpoint, `generate_2d.py --axes` can also run single views.
- **Through-plane loss (`--tp_weight`, default 0 = off).** An L1 loss between the slice-to-slice differences of the prediction and those of the real MRI. It suppresses banding without blurring real edges. Try 0.1–0.5 if the coronal/sagittal metrics are poor.

## Metrics
Validation and `generate_2d.py` score the **rebuilt 3D volume** (MAE, PSNR, SSIM3D), so the numbers compare directly with the 3D model. They also report:
- `ssim_{sagittal,coronal,axial}`: mean 2D SSIM over slices in each view. If axial is good but coronal/sagittal are worse, the volume has banding.
- `tp_err_{view}`: through-plane error along each axis (lower is better). It measures flicker directly.

## Output geometry
Unlike the 3D pipeline, outputs are not padded or cropped. They have the CT's own voxel grid, in canonical orientation with the matching affine, so they overlay on the source CT and real MRI in any NIfTI viewer.
