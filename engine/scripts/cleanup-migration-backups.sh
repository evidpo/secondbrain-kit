#!/bin/bash
# Prune LightRAG JSON-to-Postgres migration leftovers older than 7 days.
# Cron: 0 4 * * 0 (weekly, Sunday 04:00 UTC)
#
# Removes:
#   $VAULT/.lightrag/kv_store_*.json.migrated-YYYY-MM-DD          (renamed sources)
#   $VAULT/.lightrag-json-backup-YYYY-MM-DD/                       (pre-migration copies)
#
# Selection by file mtime (-mtime +7), so a fresh migration's files are kept
# until they age past the safety window — without us hardcoding any dates.
set -euo pipefail

VAULT="${VAULT:-/home/miki_xbot/SecondBrain}"
LIGHTRAG_DIR="$VAULT/.lightrag"
DAYS="${RETENTION_DAYS:-7}"

if [ ! -d "$LIGHTRAG_DIR" ]; then
  echo "vault dir missing: $LIGHTRAG_DIR" >&2
  exit 0
fi

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[$(ts)] scanning for migration leftovers older than ${DAYS}d"

# Renamed JSON sources (one file per namespace)
find "$LIGHTRAG_DIR" -maxdepth 1 -type f \
  -name 'kv_store_*.json.migrated-*' \
  -mtime +"$DAYS" -print -delete

# Backup directories (one per migration date)
find "$VAULT" -maxdepth 1 -type d \
  -name '.lightrag-json-backup-*' \
  -mtime +"$DAYS" -print -exec rm -rf {} +

echo "[$(ts)] cleanup complete"
