#!/usr/bin/env bash
set -euo pipefail

DYNAMIC_ROOT=${1:-dynamic_data}
OUT_DATA=${2:-data/predypocket_dynamic}
OUT_MODEL=${3:-runs/predypocket}
MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-weights/predypocket_model}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
WORKERS=${WORKERS:-8}
MAX_RESIDUES=${MAX_RESIDUES:-1000}

mkdir -p "${OUT_DATA}" "${OUT_MODEL}"

PYTHONPATH=src python -u src/prepare_predypocket_data.py \
  --dynamic-root "${DYNAMIC_ROOT}" \
  --out-dir "${OUT_DATA}" \
  --label-methods local_contact,homolog_contact,expanded_contact \
  --enable-rcsb \
  --homolog-identity 0.70 \
  --homolog-coverage 0.80 \
  --max-homologs 25 \
  --workers "${WORKERS}" \
  --worker-timeout 1200 \
  --contact-cutoff 4.5 \
  --buffer-cutoff 6.0

PYTHONPATH=src python -u src/train_predypocket.py \
  --dataset-csv "${OUT_DATA}/dataset_expanded_contact.csv" \
  --checkpoint "${MODEL_CHECKPOINT}" \
  --out-dir "${OUT_MODEL}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --val-fraction 0.1 \
  --learning-rate 1e-4 \
  --freeze-backbone \
  --manual-gpu-replicas \
  --max-residues "${MAX_RESIDUES}"
