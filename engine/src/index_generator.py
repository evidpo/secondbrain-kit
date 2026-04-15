"""Generate _index.md — vault catalog for fast Claude Code context injection."""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .lightrag_engine import strip_frontmatter
from .link_integrity import WIKI_LINK_RE, _vault_md_files, _extract_title

logger = logging.getLogger(__name__)

VAULT_PATH = os.getenv("VAULT_PATH", "/app/vault")
MAX_INDEX_CHARS = int(os.getenv("INDEX_MAX_CHARS", "20000"))

# Types excluded from the index (noise, not knowledge)
EXCLUDED_TYPES = {"channel-post", "unknown"}

# Filename prefix patterns that imply channel-post type
_TG_POST_PREFIXES = ("tg ", "tg-")


def _infer_type(note_type: str, rel_path: str) -> str:
    """Infer type for notes that have no type set in frontmatter."""
    if note_type != "unknown":
        return note_type
    fname = Path(rel_path).stem.lower()
    if any(fname.startswith(p) for p in _TG_POST_PREFIXES) or "tg-channel" in rel_path:
        return "channel-post"
    return "unknown"


def generate_index(vault_path: str | None = None) -> str:
    """Generate vault index as markdown string.

    Lists notes sorted by incoming link count (most-referenced first).
    Excludes noise types (channel-post, unknown).
    Capped at MAX_INDEX_CHARS for session-start hook limit.
    """
    vp = vault_path or VAULT_PATH
    vault = Path(vp)

    notes = []
    for f in _vault_md_files(vp):
        try:
            raw = f.read_text(encoding="utf-8").strip()
            rel = str(f.relative_to(vault))
            title = _extract_title(raw, f)

            note_type = "unknown"
            tags = []
            if raw.startswith("---"):
                end = raw.find("---", 3)
                if end != -1:
                    try:
                        meta = yaml.safe_load(raw[3:end])
                        if isinstance(meta, dict):
                            note_type = meta.get("type", "unknown")
                            tags = meta.get("tags", [])
                            if isinstance(tags, str):
                                tags = [tags]
                    except Exception:
                        pass

            note_type = _infer_type(note_type, rel)
            if note_type in EXCLUDED_TYPES:
                continue

            out_links = len(WIKI_LINK_RE.findall(raw))

            notes.append({
                "title": title,
                "type": note_type,
                "tags": tags[:5],
                "out_links": out_links,
                "path": rel,
            })
        except Exception:
            continue

    if not notes:
        return "# Vault Index\n\nNo notes found.\n"

    # Count incoming links
    incoming: dict[str, int] = {}
    for f in _vault_md_files(vp):
        try:
            content = f.read_text(encoding="utf-8")
            for m in WIKI_LINK_RE.finditer(content):
                base = m.group(1).split("#")[0].split("|")[0].strip().lower()
                incoming[base] = incoming.get(base, 0) + 1
        except Exception:
            continue

    for note in notes:
        note["in_links"] = incoming.get(note["title"].lower(), 0)

    # Sort by incoming links first (hub notes), then outgoing (overview notes)
    notes.sort(key=lambda n: (n["in_links"], n["out_links"]), reverse=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    total_in_vault = len(notes)
    lines = [
        "# Vault Index (auto-generated)",
        f"Updated: {now}",
        f"Total: {total_in_vault} notes",
        "",
        "| Title | Type | Tags | In | Out |",
        "|-------|------|------|----|-----|",
    ]

    for note in notes:
        tags_str = ", ".join(str(t) for t in note["tags"]) if note["tags"] else ""
        lines.append(
            f"| {note['title']} | {note['type']} | {tags_str} | "
            f"{note['in_links']} | {note['out_links']} |"
        )

    content = "\n".join(lines) + "\n"

    if len(content) > MAX_INDEX_CHARS:
        header_end = content.index("|-------|")
        header_end = content.index("\n", header_end) + 1
        header = content[:header_end]
        remaining = MAX_INDEX_CHARS - len(header) - 50

        rows = content[header_end:].split("\n")
        kept = []
        total_len = 0
        for row in rows:
            if not row.strip():
                continue
            if total_len + len(row) + 1 > remaining:
                break
            kept.append(row)
            total_len += len(row) + 1

        dropped = total_in_vault - len(kept)
        logger.warning("Index truncated: %d notes dropped (MAX_INDEX_CHARS=%d)", dropped, MAX_INDEX_CHARS)
        footer = f"\n... and {dropped} more notes\n"
        content = header + "\n".join(kept) + footer

    return content


def write_index(vault_path: str | None = None) -> str:
    """Generate and write _index.md to vault root. Returns file path."""
    vp = vault_path or VAULT_PATH
    content = generate_index(vp)
    index_path = Path(vp) / "_index.md"
    index_path.write_text(content, encoding="utf-8")
    logger.info("Index written: %s (%d chars)", index_path, len(content))
    return str(index_path)
