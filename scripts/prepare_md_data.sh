#!/usr/bin/env bash
set -euo pipefail

DYNAMIC_ROOT=${1:-dynamic_data}
OUT_DIR=${2:-data/predypocket_dynamic}
WORKERS=${WORKERS:-8}

PYTHONPATH=src python -u src/prepare_predypocket_data.py \
  --dynamic-root "${DYNAMIC_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --label-methods local_contact,homolog_contact,expanded_contact \
  --enable-rcsb \
  --homolog-identity 0.70 \
  --homolog-coverage 0.80 \
  --max-homologs 25 \
  --workers "${WORKERS}" \
  --worker-timeout 1200 \
  --contact-cutoff 4.5 \
  --buffer-cutoff 6.0
