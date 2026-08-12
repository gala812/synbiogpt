#!/usr/bin/env bash
set -euo pipefail

if [[ -d /qiannanhu01/synbiogpt ]]; then
  storage_root=/qiannanhu01
else
  storage_root=/qiannanhu01_nfs
fi

PROJECT_ROOT="${PROJECT_ROOT:-${storage_root}/synbiogpt}"
PYTHON_BIN="${PYTHON_BIN:-/persist_data/home/qiannanhu/.conda/envs/mineru/bin/python}"
CHUNKS_DIR="${CHUNKS_DIR:-${PROJECT_ROOT}/backend/data/PDF/chunks}"
RECOVERED_CHUNKS_DIR="${RECOVERED_CHUNKS_DIR:-${PROJECT_ROOT}/backend/data/PDF/image_index_integration_v1/recovered_chunks}"
MAPPING_DB="${MAPPING_DB:-${PROJECT_ROOT}/data/id_mapping/pmid_pmcid_full.sqlite3}"
MODEL_DIR="${MODEL_DIR:-${storage_root}/models/MedCPT/Article-Encoder}"
STATE_FILE="${STATE_FILE:-${PROJECT_ROOT}/backend/data/PDF/medcpt_ip_index_manifest.json}"
QDRANT_URL="${QDRANT_URL:-http://58.19.38.186:6333}"
OPENSEARCH_URL="${OPENSEARCH_URL:-http://58.19.38.186:9200}"
PATCHES_DIR="${PATCHES_DIR:-${PROJECT_ROOT}/backend/data/PDF/image_index_integration_v1/payload_patches}"
PATCH_STATE_FILE="${PATCH_STATE_FILE:-${PROJECT_ROOT}/backend/data/PDF/medcpt_ip_image_patch_manifest.json}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-256}"
UPLOAD_BATCH_SIZE="${UPLOAD_BATCH_SIZE:-2048}"
LIMIT_SHARDS="${LIMIT_SHARDS:-0}"

args=(
  "${PROJECT_ROOT}/scripts/index_medcpt_fulltext.py"
  --chunks-dir "${CHUNKS_DIR}"
  --mapping-db "${MAPPING_DB}"
  --model "${MODEL_DIR}"
  --state-file "${STATE_FILE}"
  --collection fulltext_medcpt_ip_v1
  --bm25-index fulltext_bm25_v1
  --vector-only
  --device cuda
  --dtype float16
  --encode-batch-size "${ENCODE_BATCH_SIZE}"
  --upload-batch-size "${UPLOAD_BATCH_SIZE}"
  --qdrant-url "${QDRANT_URL}"
  --opensearch-url "${OPENSEARCH_URL}"
  --local-files-only
)

if [[ -d "${RECOVERED_CHUNKS_DIR}" ]]; then
  args+=(--chunks-dir "${RECOVERED_CHUNKS_DIR}")
fi

if (( LIMIT_SHARDS > 0 )); then
  args+=(--limit-shards "${LIMIT_SHARDS}")
fi

"${PYTHON_BIN}" "${args[@]}"

if [[ "${LIMIT_SHARDS}" -eq 0 ]]; then
  if [[ -d "${PATCHES_DIR}" ]]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/apply_medcpt_image_patches.py" \
      --patches-dir "${PATCHES_DIR}" \
      --state-file "${PATCH_STATE_FILE}" \
      --qdrant-url "${QDRANT_URL}" \
      --collection fulltext_medcpt_ip_v1 \
      --vector-only
  fi
fi
