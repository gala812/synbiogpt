#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" >/dev/null 2>&1 && pwd)
cd "$SCRIPT_DIR" || exit

PORT="${PORT:-8080}"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/dev_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "$LOG_DIR"

export VECTOR_DB="${VECTOR_DB:-qdrant}"
export QDRANT_URI="${QDRANT_URI:-http://localhost:6333}"
export QDRANT_API_KEY="${QDRANT_API_KEY:-}"
export QDRANT_COLLECTION_PREFIX="${QDRANT_COLLECTION_PREFIX:-}"

export OPENSEARCH_URI="${OPENSEARCH_URI:-http://localhost:9200}"
export OPENSEARCH_SSL="${OPENSEARCH_SSL:-false}"
export OPENSEARCH_USERNAME="${OPENSEARCH_USERNAME:-}"
export OPENSEARCH_PASSWORD="${OPENSEARCH_PASSWORD:-}"
export OPENSEARCH_CERT_VERIFY="${OPENSEARCH_CERT_VERIFY:-false}"
export OPENSEARCH_BM25_INDEX="${OPENSEARCH_BM25_INDEX:-open_webui_bm25}"

echo "Vector DB: $VECTOR_DB"
if [ "$VECTOR_DB" = "qdrant" ]; then
  echo "Checking Qdrant: $QDRANT_URI"
  python - <<'PY'
import os
import sys

from qdrant_client import QdrantClient

uri = os.environ.get("QDRANT_URI")
api_key = os.environ.get("QDRANT_API_KEY") or None

try:
    client = QdrantClient(url=uri, api_key=api_key, timeout=30)
    collections = client.get_collections().collections
    print(f"Qdrant ok: {uri}, collections={len(collections)}")
except Exception as e:
    print(f"Qdrant check failed: {e}", file=sys.stderr)
    sys.exit(1)
PY
fi

echo "Checking OpenSearch BM25: $OPENSEARCH_URI"
python - <<'PY'
import os
import sys
import urllib.request

uri = (os.environ.get("OPENSEARCH_URI") or "").rstrip("/")
username = os.environ.get("OPENSEARCH_USERNAME") or None
password = os.environ.get("OPENSEARCH_PASSWORD") or None

try:
    request = urllib.request.Request(uri)
    if username and password:
        import base64

        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")

    with urllib.request.urlopen(request, timeout=10) as response:
        print(f"OpenSearch ok: {uri}, status={response.status}")
except Exception as e:
    print(f"OpenSearch check failed: {e}", file=sys.stderr)
    sys.exit(1)
PY

echo "Writing backend logs to $LOG_FILE"
echo "Starting backend on port $PORT ..."

nohup uvicorn open_webui.main:app \
  --host 127.0.0.1 \
  --port "$PORT" \
  --forwarded-allow-ips '*' \
  >> "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$LOG_DIR/dev.pid"

echo "Backend started in background. PID: $SERVER_PID"
echo "Log file: $LOG_FILE"
echo "Follow logs: tail -f \"$LOG_FILE\""
echo "Stop backend: kill \$(cat \"$LOG_DIR/dev.pid\")"
