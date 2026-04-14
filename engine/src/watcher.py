"""File watcher: monitors inbox/ for new notes, vault/ for renames, title changes,
creates and deletes — reactive graph sync on file changes."""

import logging
import os
import sys
import time
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from .approval import APPROVAL_MODE, handle_callback, cleanup_stale
from .processor import process_file
from .telegram import notify_inbox, notify_orphans, cleanup_system_notifications
from .lightrag_engine import (
    get_instance, shutdown, sync_with_vault,
    insert as lightrag_insert, delete_doc, strip_frontmatter,
    find_doc_id_by_path,
)
from .path_sync import (
    FrontmatterCache,
    handle_move,
    handle_modify,
    sync_paths,
    _is_vault_note,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/secondbrain-watcher.log"),
    ],
)
logger = logging.getLogger(__name__)

VAULT_PATH = os.getenv("VAULT_PATH", "/app/vault")
INBOX_DIR_NAME = os.getenv("INBOX_DIR_NAME", "_inbox")
INBOX_DIR = os.path.join(VAULT_PATH, INBOX_DIR_NAME)
RETRY_DELAY = 300  # 5 minutes
SYNC_INTERVAL = 300  # 5 minutes — check graph for orphans (notify, no auto-delete)
PATH_SYNC_INTERVAL = 30  # 30 seconds — check for renames/title changes
REINDEX_CHANGE_THRESHOLD = int(os.getenv("REINDEX_CHANGE_THRESHOLD", "20"))  # auto-reindex after N changes
STALE_CHECK_INTERVAL = 3600  # 1 hour — check for stale approvals
DAILY_CLEANUP_INTERVAL = 86400  # 24 hours — purge accumulated system notifications


class InboxHandler(FileSystemEventHandler):
    """Handle new .md files in Inbox."""

    def __init__(self):
        self._retry_queue: list[tuple[str, int]] = []

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        if not event.src_path.endswith(".md"):
            return
        if os.path.basename(event.src_path).startswith("."):
            return

        logger.info(f"New file detected: {event.src_path}")

        # Small delay to let file writing complete
        time.sleep(1)

        self._process(event.src_path)

    def _process(self, file_path: str) -> None:
        try:
            created = process_file(file_path, source="inbox")
            if created:
                logger.info(f"Processed → {len(created)} notes")
            else:
                logger.info(f"No notes created from {file_path}")
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            self._retry_queue.append((file_path, int(time.time())))

    def retry_failed(self) -> None:
        """Retry failed files after delay."""
        now = int(time.time())
        remaining = []
        for file_path, failed_at in self._retry_queue:
            if now - failed_at >= RETRY_DELAY:
                logger.info(f"Retrying: {file_path}")
                if Path(file_path).exists():
                    self._process(file_path)
            else:
                remaining.append((file_path, failed_at))
        self._retry_queue = remaining


def _delete_by_path(rel_path: str) -> bool:
    """Delete a doc from LightRAG by its vault-relative file path."""
    doc_id = find_doc_id_by_path(rel_path)
    if doc_id:
        return delete_doc(doc_id)
    logger.debug("No indexed doc found for path: %s (will catch on next sync)", rel_path)
    return False


class VaultHandler(FileSystemEventHandler):
    """Watch entire vault for renames, moves, title changes, creates and deletes."""

    _DEBOUNCE_SEC = 2.0
    _BATCH_SEC = 3.0  # batch window for create/delete events

    def __init__(self, cache: FrontmatterCache):
        self._cache = cache
        self._debounce: dict[str, float] = {}
        self._lock = threading.Lock()
        self._pending_creates: dict[str, float] = {}  # path -> timestamp
        self._pending_deletes: dict[str, float] = {}  # path -> timestamp

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        if not _is_vault_note(event.dest_path, VAULT_PATH):
            return
        logger.info("Vault move: %s -> %s", event.src_path, event.dest_path)
        handle_move(event.src_path, event.dest_path, self._cache)

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        if not _is_vault_note(event.src_path, VAULT_PATH):
            return

        # Debounce: editors trigger multiple modify events per save
        now = time.time()
        last = self._debounce.get(event.src_path, 0)
        if now - last < self._DEBOUNCE_SEC:
            return
        self._debounce[event.src_path] = now

        handle_modify(event.src_path, self._cache)

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        if not _is_vault_note(event.src_path, VAULT_PATH):
            return
        with self._lock:
            # Cancel pending delete for same path (rename = delete + create)
            self._pending_deletes.pop(event.src_path, None)
            self._pending_creates[event.src_path] = time.time()

    def on_deleted(self, event) -> None:
        if event.is_directory:
            return
        # Can't call _is_vault_note on deleted file — check path manually
        if not event.src_path.endswith(".md"):
            return
        try:
            rel = os.path.relpath(event.src_path, VAULT_PATH)
        except ValueError:
            return
        parts = Path(rel).parts
        # Must be inside a subdirectory (root-level files are system files)
        if len(parts) < 2:
            return
        skip_dirs = {"templates", ".obsidian", ".git", ".lightrag", ".entire", ".trash"}
        inbox = os.getenv("INBOX_DIR_NAME", "_inbox")
        if any(p in skip_dirs or p.startswith(".") for p in parts):
            return
        if parts and parts[0] == inbox:
            return
        with self._lock:
            # Cancel pending create for same path
            self._pending_creates.pop(event.src_path, None)
            self._pending_deletes[event.src_path] = time.time()

    def flush_pending(self) -> dict:
        """Process batched create/delete events. Call from main loop.

        Returns: {inserted: int, deleted: int}
        """
        now = time.time()
        to_create = []
        to_delete = []

        with self._lock:
            # Collect events older than batch window
            for path, ts in list(self._pending_creates.items()):
                if now - ts >= self._BATCH_SEC:
                    to_create.append(path)
                    del self._pending_creates[path]
            for path, ts in list(self._pending_deletes.items()):
                if now - ts >= self._BATCH_SEC:
                    to_delete.append(path)
                    del self._pending_deletes[path]

        inserted = 0
        deleted = 0

        for path in to_create:
            try:
                fp = Path(path)
                if not fp.exists():
                    continue
                raw = fp.read_text(encoding="utf-8").strip()
                body = strip_frontmatter(raw)
                if not body or len(body) < 20:
                    continue
                rel_path = str(fp.relative_to(Path(VAULT_PATH)))
                lightrag_insert(raw, file_path=rel_path)
                inserted += 1
                logger.info("Reactive insert: %s", rel_path)
            except Exception as e:
                logger.warning("Reactive insert failed for %s: %s", path, e)

        for path in to_delete:
            try:
                rel_path = os.path.relpath(path, VAULT_PATH)
                logger.info("File removed from vault (not deleting from graph): %s", rel_path)
                notify_inbox(
                    f"🗑 <b>Файл удалён из vault</b>\n"
                    f"<code>{rel_path}</code>\n"
                    f"Запись в графе сохранена. Удалить из графа вручную: "
                    f"<code>POST /reindex-sync</code>"
                )
                deleted += 1
            except Exception as e:
                logger.warning("Delete notify failed for %s: %s", path, e)

        return {"inserted": inserted, "deleted": deleted}


def process_existing_inbox() -> None:
    """Process any .md files already in Inbox on startup."""
    inbox = Path(INBOX_DIR)
    if not inbox.exists():
        return

    files = sorted(inbox.glob("*.md"))
    if files:
        logger.info(f"Found {len(files)} existing files in Inbox")
        for f in files:
            if f.name.startswith("."):
                continue
            try:
                process_file(str(f), source="inbox")
            except Exception as e:
                logger.error(f"Failed to process {f.name}: {e}")


def start_watcher() -> None:
    """Start watching Inbox directory and vault for path changes."""
    logger.info("SecondBrain Daemon starting...")
    logger.info(f"Vault: {VAULT_PATH}")
    logger.info(f"Inbox: {INBOX_DIR}")

    # Ensure directories exist
    os.makedirs(INBOX_DIR, exist_ok=True)

    # Init LightRAG
    get_instance()
    logger.info("LightRAG engine ready")

    # Build frontmatter cache for path sync
    fm_cache = FrontmatterCache()
    if not fm_cache.all():
        fm_cache.build(VAULT_PATH)
    logger.info("Frontmatter cache ready (%d entries)", len(fm_cache.all()))

    # Process existing inbox files
    process_existing_inbox()

    # Full vault reindex on startup (background thread — non-blocking)
    def _startup_reindex():
        from .lightrag_engine import get_instance
        from .index_generator import write_index

        vault = Path(VAULT_PATH)
        skip = {"templates", ".obsidian", ".git", ".lightrag", ".entire", ".trash", "_system"}
        indexed = 0
        skipped = 0
        errors = 0

        # Load current doc_status to skip already-processed docs (by file path)
        try:
            rag = get_instance()
            processed_paths = {
                info.get("file_path")
                for info in (rag.doc_status._data or {}).values()
                if isinstance(info, dict) and info.get("status") == "processed"
                and info.get("file_path")
            }
        except Exception:
            processed_paths = set()

        for d in vault.iterdir():
            if not d.is_dir() or d.name in skip or d.name.startswith("."):
                continue
            if d.name == INBOX_DIR_NAME:
                continue
            for f in d.rglob("*.md"):
                if f.name.startswith("."):
                    continue
                try:
                    rel_path = str(f.relative_to(vault))
                    if rel_path in processed_paths:
                        skipped += 1
                        continue
                    content = f.read_text(encoding="utf-8")
                    lightrag_insert(content, file_path=rel_path)
                    indexed += 1
                except Exception as e:
                    logger.warning("Startup reindex failed for %s: %s", f, e)
                    errors += 1

        logger.info(
            "Startup reindex: %d new, %d already indexed, %d errors",
            indexed, skipped, errors,
        )
        if indexed:
            notify_inbox(f"🔄 Реиндекс при старте: {indexed} новых заметок")

        # Sync wiki-links after reindex so Obsidian graph stays in sync
        try:
            from .api import _sync_all_links
            link_result = _sync_all_links()
            added = link_result.get("total_links_added", 0)
            if added:
                logger.info("Startup link sync: %d links added to %d notes",
                            added, len(link_result.get("notes_updated", [])))
        except Exception as e:
            logger.warning("Startup link sync failed: %s", e)

        # Regenerate _index.md after reindex + link sync
        try:
            write_index(VAULT_PATH)
            logger.info("_index.md updated")
        except Exception as e:
            logger.warning("Index write failed: %s", e)

    threading.Thread(target=_startup_reindex, daemon=True, name="startup-reindex").start()

    # Inbox watcher
    inbox_handler = InboxHandler()
    observer = Observer()
    observer.schedule(inbox_handler, INBOX_DIR, recursive=False)

    # Vault watcher — renames, moves, title changes
    vault_handler = VaultHandler(fm_cache)
    observer.schedule(vault_handler, VAULT_PATH, recursive=True)

    observer.start()
    logger.info("Watching inbox + vault for changes...")

    # Callbacks arrive via HTTP POST /telegram/callback from openclaw-gateway.
    # Do NOT start a local getUpdates poll — it conflicts with the main bot (sa_bot).

    last_graph_sync = time.time()
    last_path_sync = time.time()
    last_stale_check = time.time()
    last_daily_cleanup = time.time()
    change_counter = 0  # track vault changes for threshold-based reindex
    _last_orphan_set: frozenset[str] = frozenset()  # dedup: skip if same orphans as last time

    try:
        while True:
            time.sleep(2)
            inbox_handler.retry_failed()

            # Flush batched create/delete events (reactive graph sync)
            pending = vault_handler.flush_pending()
            if pending["inserted"] or pending["deleted"]:
                change_counter += pending["inserted"] + pending["deleted"]
                logger.info(
                    "Reactive sync: %d inserted, %d removed (total changes: %d)",
                    pending["inserted"], pending["deleted"], change_counter,
                )

            # Threshold-based reindex: when enough changes accumulate
            if change_counter >= REINDEX_CHANGE_THRESHOLD:
                logger.info("Change threshold reached (%d), triggering reindex", change_counter)
                change_counter = 0
                def _threshold_reindex():
                    from .api import _reindex_vault
                    try:
                        result = _reindex_vault()
                        logger.info("Threshold reindex: %d indexed", result["indexed"])
                        notify_inbox(f"🔄 Авто-реиндекс ({REINDEX_CHANGE_THRESHOLD}+ изменений): "
                                  f"{result['indexed']} заметок")
                    except Exception as e:
                        logger.warning("Threshold reindex failed: %s", e)
                threading.Thread(target=_threshold_reindex, daemon=True).start()

            now = time.time()

            # Periodic path sync: detect renames/title changes (git pull scenarios)
            if now - last_path_sync >= PATH_SYNC_INTERVAL:
                try:
                    result = sync_paths(fm_cache, VAULT_PATH)
                    renames = len(result.get("renames", []))
                    changes = len(result.get("title_changes", []))
                    if renames or changes:
                        change_counter += renames + changes
                        logger.info(
                            "Path sync: %d renames, %d title changes", renames, changes
                        )
                except Exception as e:
                    logger.warning(f"Path sync failed: {e}")
                last_path_sync = now

            # Periodic graph sync: detect orphans + TTL cleanup of old notifications
            if now - last_graph_sync >= SYNC_INTERVAL:
                try:
                    result = sync_with_vault(VAULT_PATH, dry_run=True)
                    orphans = result.get("orphans", [])
                    current_set = frozenset(orphans)
                    if orphans and current_set != _last_orphan_set:
                        logger.info("Graph sync: %d orphan docs found (not deleted)", len(orphans))
                        notify_orphans(orphans)
                        _last_orphan_set = current_set
                    elif not orphans and _last_orphan_set:
                        logger.info("Graph sync: all orphans resolved")
                        _last_orphan_set = frozenset()
                except Exception as e:
                    logger.warning(f"Graph sync failed: {e}")

                # TTL cleanup: delete notifications older than NOTIF_TTL
                try:
                    from .telegram import NOTIF_TTL
                    cleanup_system_notifications(max_age=NOTIF_TTL)
                except Exception as e:
                    logger.warning("TTL cleanup failed: %s", e)

                last_graph_sync = now

            # Daily cleanup: force-delete ALL remaining notifications, reset state
            if now - last_daily_cleanup >= DAILY_CLEANUP_INTERVAL:
                try:
                    deleted = cleanup_system_notifications(max_age=None)
                    logger.info("Daily cleanup: removed %d old notifications", deleted)
                    _last_orphan_set = frozenset()
                except Exception as e:
                    logger.warning("Daily cleanup failed: %s", e)
                last_daily_cleanup = now

            # Stale approval cleanup
            if APPROVAL_MODE == "approve" and now - last_stale_check >= STALE_CHECK_INTERVAL:
                try:
                    refreshed = cleanup_stale()
                    if refreshed:
                        logger.info("Stale approvals refreshed: %d", refreshed)
                except Exception as e:
                    logger.warning("Stale cleanup failed: %s", e)
                last_stale_check = now
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        observer.stop()
    observer.join()
    shutdown()
    logger.info("Daemon stopped.")


if __name__ == "__main__":
    start_watcher()
