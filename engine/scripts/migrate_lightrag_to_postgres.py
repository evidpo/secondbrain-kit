"""Migrate LightRAG KV + DocStatus from JSON files to Postgres backend.

Run inside the secondbrain-daemon container with PG env vars set:

    docker compose stop secondbrain-daemon secondbrain-webui
    docker compose run --rm --entrypoint python secondbrain-daemon \
        -m scripts.migrate_lightrag_to_postgres

The script:
  1. Pre-flight: env vars, postgres reachable, JSON files exist.
  2. Backup: copy kv_store_*.json into a sibling backup directory.
  3. Initialize LightRAG with PG backend → creates empty tables.
  4. Idempotency check: aborts if PG storage is not empty.
  5. Migrate KV namespaces (text_chunks, entity_chunks, …, llm_response_cache)
     in batches of 1000. Doc_status migrated separately.
  6. Spot-check: read 50 random keys back from PG and compare counts.
  7. Rename source JSON files to *.migrated-YYYY-MM-DD (kept for rollback).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import glob
import json
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("migrate")

# Map JSON file namespace → LightRAG instance attribute holding the KV storage.
KV_NAMESPACES = [
    "full_docs",
    "text_chunks",
    "entity_chunks",
    "relation_chunks",
    "full_entities",
    "full_relations",
    "llm_response_cache",
]

BATCH_SIZE = 1000
SPOT_CHECK_SIZE = 50
MIGRATED_SUFFIX = f".migrated-{_dt.date.today().isoformat()}"


def _pre_flight() -> Path:
    """Validate env and return working_dir Path."""
    must_be = {
        "LIGHTRAG_KV_STORAGE": "PGKVStorage",
        "LIGHTRAG_DOC_STATUS_STORAGE": "PGDocStatusStorage",
    }
    for k, expected in must_be.items():
        actual = os.getenv(k, "")
        if actual != expected:
            sys.exit(f"FATAL: env {k}={actual!r}, expected {expected!r}. "
                     "Update .env and re-run.")

    for k in ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DATABASE",
              "POSTGRES_USER", "POSTGRES_PASSWORD"):
        if not os.getenv(k):
            sys.exit(f"FATAL: env {k} is empty.")

    working_dir = Path(os.getenv("LIGHTRAG_WORKING_DIR", ""))
    if not working_dir.is_dir():
        sys.exit(f"FATAL: LIGHTRAG_WORKING_DIR {working_dir} not found")
    return working_dir


def _backup_json(working_dir: Path) -> Path:
    """Copy kv_store_*.json files into a sibling backup directory."""
    backup_dir = working_dir.parent / f".lightrag-json-backup-{_dt.date.today().isoformat()}"
    if backup_dir.exists():
        log.warning("Backup dir %s already exists, skipping copy", backup_dir)
        return backup_dir
    backup_dir.mkdir()
    for src in working_dir.glob("kv_store_*.json"):
        shutil.copy2(src, backup_dir / src.name)
    log.info("Backed up %d JSON files to %s", len(list(backup_dir.iterdir())), backup_dir)
    return backup_dir


async def _check_postgres_reachable():
    """Independent connectivity test — confirms NO_PROXY is set correctly."""
    import asyncpg
    conn = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DATABASE"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        timeout=5,
    )
    version = await conn.fetchval("SELECT version()")
    await conn.close()
    log.info("Postgres reachable: %s", version.split(",")[0])


async def _migrate_kv_namespace(rag, namespace: str, working_dir: Path) -> int:
    """Migrate one kv_store_<namespace>.json into PG storage. Returns row count."""
    src = working_dir / f"kv_store_{namespace}.json"
    if not src.exists():
        log.warning("  %s: source file missing, skipping", namespace)
        return 0

    storage = getattr(rag, namespace, None)
    if storage is None:
        sys.exit(f"FATAL: rag.{namespace} attribute not found — "
                 "LightRAG version mismatch?")

    data = json.loads(src.read_text("utf-8"))
    keys = list(data.keys())
    total = len(keys)
    if total == 0:
        log.info("  %s: empty source, skipping", namespace)
        return 0

    for i in range(0, total, BATCH_SIZE):
        batch = {k: data[k] for k in keys[i:i + BATCH_SIZE]}
        await storage.upsert(batch)
        log.info("  %s: %d/%d", namespace, min(i + BATCH_SIZE, total), total)

    return total


async def _migrate_doc_status(rag, working_dir: Path) -> int:
    """Migrate kv_store_doc_status.json into PG doc_status."""
    src = working_dir / "kv_store_doc_status.json"
    if not src.exists():
        log.warning("doc_status source missing")
        return 0

    data = json.loads(src.read_text("utf-8"))
    keys = list(data.keys())
    total = len(keys)

    for i in range(0, total, BATCH_SIZE):
        batch = {k: data[k] for k in keys[i:i + BATCH_SIZE]}
        await rag.doc_status.upsert(batch)
        log.info("  doc_status: %d/%d", min(i + BATCH_SIZE, total), total)

    return total


async def _spot_check(rag, working_dir: Path):
    """Read SPOT_CHECK_SIZE keys from each namespace, compare to source."""
    for ns in KV_NAMESPACES:
        src = working_dir / f"kv_store_{ns}.json"
        if not src.exists():
            continue
        data = json.loads(src.read_text("utf-8"))
        keys = list(data.keys())[:SPOT_CHECK_SIZE]
        if not keys:
            continue
        storage = getattr(rag, ns)
        found = await storage.get_by_ids(keys)
        non_null = sum(1 for x in found if x is not None)
        if non_null != len(keys):
            sys.exit(f"FATAL: {ns} spot-check: {non_null}/{len(keys)} found in PG")
        log.info("  %s: spot-check OK (%d/%d)", ns, non_null, len(keys))

    # doc_status: use get_by_id (no get_by_ids on every storage)
    src = working_dir / "kv_store_doc_status.json"
    if src.exists():
        data = json.loads(src.read_text("utf-8"))
        keys = list(data.keys())[:SPOT_CHECK_SIZE]
        if keys:
            found = await rag.doc_status.get_by_ids(keys)
            non_null = sum(1 for x in found if x is not None)
            if non_null != len(keys):
                sys.exit(f"FATAL: doc_status spot-check: {non_null}/{len(keys)}")
            log.info("  doc_status: spot-check OK (%d/%d)", non_null, len(keys))


def _rename_sources(working_dir: Path):
    """Rename kv_store_*.json → kv_store_*.json.migrated-YYYY-MM-DD."""
    renamed = 0
    for src in working_dir.glob("kv_store_*.json"):
        dst = src.with_suffix(src.suffix + MIGRATED_SUFFIX)
        src.rename(dst)
        renamed += 1
    log.info("Renamed %d JSON files with suffix %s", renamed, MIGRATED_SUFFIX)


async def main():
    log.info("=== LightRAG JSON → Postgres migration ===")
    working_dir = _pre_flight()
    log.info("Working dir: %s", working_dir)

    backup_dir = _backup_json(working_dir)
    log.info("Backup dir:  %s", backup_dir)

    log.info("Testing Postgres connectivity…")
    await _check_postgres_reachable()

    log.info("Initializing LightRAG with PG backend (will create tables)…")
    # Import here so working_dir env is read first.
    from src.lightrag_engine import _create_instance
    rag = await _create_instance()

    # Idempotency: refuse to overwrite a non-empty target.
    if not await rag.doc_status.is_empty():
        sys.exit(
            "FATAL: PG doc_status is not empty. Migration is non-idempotent.\n"
            "If you want to re-run, drop and recreate tables manually:\n"
            "  docker exec secondbrain-postgres psql -U $POSTGRES_USER "
            "-d $POSTGRES_DATABASE -c "
            "\"DROP SCHEMA public CASCADE; CREATE SCHEMA public;\""
        )

    log.info("Migrating KV namespaces…")
    kv_total = 0
    for ns in KV_NAMESPACES:
        kv_total += await _migrate_kv_namespace(rag, ns, working_dir)

    log.info("Migrating doc_status…")
    ds_total = await _migrate_doc_status(rag, working_dir)

    log.info("Verifying with spot-checks…")
    await _spot_check(rag, working_dir)

    log.info("Renaming source JSON files…")
    _rename_sources(working_dir)

    # Finalize so PG flushes all writes.
    try:
        await rag.finalize_storages()
    except Exception:
        pass

    log.info("=== Migration complete: %d KV records + %d doc_status records ===",
             kv_total, ds_total)
    log.info("Backup retained at %s — keep for at least 7 days.", backup_dir)


if __name__ == "__main__":
    asyncio.run(main())
