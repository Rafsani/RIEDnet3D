#!/usr/bin/env bash
# Run the slice-wise (2.5D) RIED-Net CT -> MRI pipeline: prepare data -> train -> generate.
# Independent of run.sh (3D): writes to ./runs/2d and ./results/2d by default.
#
# Usage:
#   ./run_2d.sh              # run all 3 steps with the defaults below
#   ./run_2d.sh prepare      # run just the data-prep step (same splits as run.sh)
#   ./run_2d.sh train        # run just training
#   ./run_2d.sh generate     # run just generation (needs a checkpoint, see below)
#
# Override via env, e.g. AXES="0 1 2" TP_WEIGHT=0.2 EPOCHS=200 ./run_2d.sh train

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

module load mamba/latest
source activate fazle_env

CT_DIR="${CT_DIR:-./CT}"
MRI_DIR="${MRI_DIR:-./MRI}"
DATASET_DIR="${DATASET_DIR:-./dataset}"
FOLD_NAME="${FOLD_NAME:-fold1}"
FOLD_DIR="${DATASET_DIR}/${FOLD_NAME}"

MODEL_NAME="${MODEL_NAME:-resunet}"       # unet | resunet | attunet
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"            # slices per batch
CONTEXT="${CONTEXT:-2}"                   # CT neighbours each side (2 -> 5 input slices)
OUT_SLICES="${OUT_SLICES:-3}"             # MRI slices predicted per step
AXES="${AXES:-2}"                         # 0=sagittal 1=coronal 2=axial; "0 1 2" = multi-view
TP_WEIGHT="${TP_WEIGHT:-0}"               # through-plane consistency loss weight (0 = off)
RUNS_DIR="${RUNS_DIR:-./runs/2d}"
RESULTS_DIR="${RESULTS_DIR:-./results/2d}"

# Checkpoint for generation. Defaults to the best checkpoint of the most
# recently started 2D training run under $RUNS_DIR.
CHECKPOINT="${CHECKPOINT:-}"

prepare() {
    echo "== prepare_data: pairing ${CT_DIR} <-> ${MRI_DIR} into ${FOLD_DIR} =="
    python3 prepare_data.py \
        --ct_dir "${CT_DIR}" --mri_dir "${MRI_DIR}" \
        --out_dir "${DATASET_DIR}" --fold_name "${FOLD_NAME}" \
        --val_frac 0.2 --test_frac 0.2 --seed 42
}

train() {
    echo "== train_2d: ${MODEL_NAME} on ${FOLD_DIR}, axes=[${AXES}] for ${EPOCHS} epochs =="
    # shellcheck disable=SC2086
    python3 train_2d.py \
        --dataset "${FOLD_DIR}" --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" \
        --model_name "${MODEL_NAME}" --context "${CONTEXT}" --out_slices "${OUT_SLICES}" \
        --axes ${AXES} --tp_weight "${TP_WEIGHT}" --output_dir "${RUNS_DIR}"
}

generate() {
    if [[ -z "${CHECKPOINT}" ]]; then
        set +e
        CHECKPOINT="$(ls -td "${RUNS_DIR}"/${MODEL_NAME}_*/weights/best.ckpt 2>/dev/null | head -n1)"
        set -e
    fi
    if [[ -z "${CHECKPOINT}" ]]; then
        echo "No checkpoint found under ${RUNS_DIR}/${MODEL_NAME}_*/weights/best.ckpt." >&2
        echo "Run './run_2d.sh train' first, or set CHECKPOINT=/path/to/best.ckpt." >&2
        exit 1
    fi
    echo "== generate_2d: ${CHECKPOINT} on ${FOLD_DIR}/test -> ${RESULTS_DIR} =="
    python3 generate_2d.py \
        --checkpoint "${CHECKPOINT}" --dataset "${FOLD_DIR}" --split test \
        --output_dir "${RESULTS_DIR}"
}

case "${1:-all}" in
    prepare)  prepare ;;
    train)    train ;;
    generate) generate ;;
    all)      prepare; train; generate ;;
    *)        echo "Usage: $0 [prepare|train|generate|all]" >&2; exit 1 ;;
esac
