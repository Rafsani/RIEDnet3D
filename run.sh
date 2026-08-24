#!/usr/bin/env bash
# Run the full RIED-Net CT -> MRI pipeline: prepare data -> train -> generate.
#
# Usage:
#   ./run.sh                 # run all 3 steps with the defaults below
#   ./run.sh prepare         # run just the data-prep step
#   ./run.sh train           # run just training
#   ./run.sh generate        # run just generation (needs a checkpoint, see below)
#
# Edit the variables below (or override via env, e.g. EPOCHS=200 ./run.sh train)
# to point at your real CT/MRI folders once you're past the demo data.

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
BATCH_SIZE="${BATCH_SIZE:-1}"
RUNS_DIR="${RUNS_DIR:-./runs}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

# Checkpoint to use for generation. Defaults to the best checkpoint of the
# most recently started training run under $RUNS_DIR (resolved lazily inside
# generate(), so prepare/train don't need $RUNS_DIR to exist yet).
CHECKPOINT="${CHECKPOINT:-}"

prepare() {
    echo "== prepare_data: pairing ${CT_DIR} <-> ${MRI_DIR} into ${FOLD_DIR} =="
    python3 prepare_data.py \
        --ct_dir "${CT_DIR}" --mri_dir "${MRI_DIR}" \
        --out_dir "${DATASET_DIR}" --fold_name "${FOLD_NAME}" \
        --val_frac 0.2 --test_frac 0.2 --seed 42
}

train() {
    echo "== train: ${MODEL_NAME} on ${FOLD_DIR} for ${EPOCHS} epochs =="
    python3 train.py \
        --dataset "${FOLD_DIR}" --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" \
        --model_name "${MODEL_NAME}" --output_dir "${RUNS_DIR}"
}

generate() {
    if [[ -z "${CHECKPOINT}" ]]; then
        set +e
        CHECKPOINT="$(ls -td "${RUNS_DIR}"/${MODEL_NAME}_*/weights/best.ckpt 2>/dev/null | head -n1)"
        set -e
    fi
    if [[ -z "${CHECKPOINT}" ]]; then
        echo "No checkpoint found under ${RUNS_DIR}/${MODEL_NAME}_*/weights/best.ckpt." >&2
        echo "Run './run.sh train' first, or set CHECKPOINT=/path/to/best.ckpt." >&2
        exit 1
    fi
    echo "== generate: ${CHECKPOINT} on ${FOLD_DIR}/test -> ${RESULTS_DIR} =="
    python3 generate.py \
        --model_name "${MODEL_NAME}" --checkpoint "${CHECKPOINT}" \
        --dataset "${FOLD_DIR}" --split test --output_dir "${RESULTS_DIR}"
}

case "${1:-all}" in
    prepare)  prepare ;;
    train)    train ;;
    generate) generate ;;
    all)      prepare; train; generate ;;
    *)        echo "Usage: $0 [prepare|train|generate|all]" >&2; exit 1 ;;
esac
