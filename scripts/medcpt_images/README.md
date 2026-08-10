# Stable paper image access

MinerU image paths remain on the 117 data server. The image pipeline assigns
each `(PMCID, relative path)` pair a deterministic SHA-256 `asset_key`, records
it in `image_assets.sqlite3`, and serves the file without exposing its absolute
storage path.

Build or refresh the manifest:

```bash
python scripts/build_medcpt_image_assets.py \
  --documents-jsonl backend/data/PDF/documents.jsonl \
  --output-dir backend/data/PDF/image_access_v1 \
  --source-root /qiannanhu01_nfs/pdf_parse/jsonl_backup/output \
  --workers 8
```

Start the read-only service:

```bash
python scripts/serve_medcpt_image_assets.py \
  --manifest backend/data/PDF/image_access_v1/image_assets.sqlite3 \
  --port 8011
```

Clients fetch an image with `GET /assets/{asset_key}`. Metadata and per-paper
inspection are available at `/assets/{asset_key}/metadata` and
`/papers/{pmcid}/assets`. Set `PAPER_ASSET_BASE_URL` in the WebUI process so
retrieval metadata also contains ready-to-use `image_urls`.

The recovery pass classifies every Markdown image as `existing_bound`,
`recovered`, `excluded`, or `review`. It repairs missing panels and caption
associations, and creates low-confidence context-only assets for usable images
that have no reliable caption. It never performs OCR or invents visual content.

After a full build, validate all rows and source files:

```bash
python scripts/audit_medcpt_image_assets.py \
  --image-access-dir backend/data/PDF/image_access_v1 \
  --verify-files \
  --output backend/data/PDF/image_access_v1/audit.json
```
