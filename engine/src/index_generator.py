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
MAX_INDEX_CHARS = 20000


def generate_index(vault_path: str | None = None) -> str:
    """Generate vault index as markdown string.

    Lists all notes sorted by link count (most-linked first).
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

            out_links = len(WIKI_LINK_RE.findall(raw))
            body = strip_frontmatter(raw)
            word_count = len(body.split()) if body else 0

            notes.append({
                "title": title,
                "type": note_type,
                "tags": tags[:5],
                "out_links": out_links,
                "words": word_count,
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

    notes.sort(key=lambda n: n["in_links"] + n["out_links"], reverse=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    lines = [
        "# Vault Index (auto-generated)",
        f"Updated: {now}",
        f"Total: {len(notes)} notes",
        "",
        "| Title | Type | Tags | In | Out | Words |",
        "|-------|------|------|----|-----|-------|",
    ]

    for note in notes:
        tags_str = ", ".join(str(t) for t in note["tags"]) if note["tags"] else ""
        lines.append(
            f"| {note['title']} | {note['type']} | {tags_str} | "
            f"{note['in_links']} | {note['out_links']} | {note['words']} |"
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

        footer = f"\n... and {len(notes) - len(kept)} more notes\n"
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
