"""Vault / LightRAG integrity checks with optional auto-heal."""

import json
import logging
import os
import re
from pathlib import Path

import yaml

from .lightrag_engine import (
    compute_doc_id,
    strip_frontmatter,
    get_indexed_doc_ids,
    get_indexed_paths,
    insert as lightrag_insert,
    delete_doc,
)
from .link_integrity import WIKI_LINK_RE, _vault_md_files, _extract_title, clean_broken_links
from .path_sync import VAULT_SKIP_DIRS

logger = logging.getLogger(__name__)

VAULT_PATH = os.getenv("VAULT_PATH", "/app/vault")
INBOX_DIR_NAME = os.getenv("INBOX_DIR_NAME", "_inbox")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_vault_notes(vault_path: str) -> dict[str, dict]:
    """Collect all vault notes with metadata.

    Returns {rel_path: {title, content, body, doc_id, headings, path}}.
    """
    vault = Path(vault_path)
    notes: dict[str, dict] = {}
    for d in vault.iterdir():
        if not d.is_dir() or d.name in VAULT_SKIP_DIRS or d.name.startswith(".") or d.name == INBOX_DIR_NAME:
            continue
        for f in d.rglob("*.md"):
            try:
                raw = f.read_text(encoding="utf-8").strip()
                rel = str(f.relative_to(vault))
                title = _extract_title(raw, f)
                body = strip_frontmatter(raw)
                headings = re.findall(r"^##\s+(.+)$", raw, re.MULTILINE)
                doc_id = compute_doc_id(raw) if body and len(body) >= 20 else None
                notes[rel] = {
                    "title": title,
                    "content": raw,
                    "body": body,
                    "doc_id": doc_id,
                    "headings": [h.strip() for h in headings],
                    "path": f,
                }
            except Exception:
                continue
    return notes


def _collect_all_links(vault_path: str) -> dict[str, list[dict]]:
    """Collect all wiki-links from vault.

    Returns {source_rel_path: [{raw, base_title, anchor, display}]}.
    """
    vault = Path(vault_path)
    links: dict[str, list[dict]] = {}
    for f in _vault_md_files(vault_path):
        try:
            content = f.read_text(encoding="utf-8")
        except Exception:
            continue
        rel = str(f.relative_to(vault))
        file_links = []
        for m in WIKI_LINK_RE.finditer(content):
            raw = m.group(1).strip()
            parts_hash = raw.split("#", 1)
            base_and_display = parts_hash[0]
            anchor = parts_hash[1].split("|")[0].strip() if len(parts_hash) > 1 else None
            parts_pipe = base_and_display.split("|", 1)
            base_title = parts_pipe[0].strip()
            display = parts_pipe[1].strip() if len(parts_pipe) > 1 else None
            file_links.append({
                "raw": raw,
                "base_title": base_title,
                "anchor": anchor,
                "display": display,
            })
        if file_links:
            links[rel] = file_links
    return links


def _slugify(text: str) -> str:
    """Simple slug: lowercase, strip non-alnum, collapse dashes."""
    slug = re.sub(r"[^\w\s-]", "", text.lower().strip())
    return re.sub(r"[\s_]+", "-", slug).strip("-")


# ---------------------------------------------------------------------------
# CHECK 1: Dead links
# ---------------------------------------------------------------------------

def check_dead_links(
    vault_notes: dict[str, dict],
    all_links: dict[str, list[dict]],
    vault_path: str,
    fix: bool = False,
) -> dict:
    """Links pointing to notes that don't exist in the vault.

    Resolves links by three strategies (Obsidian-compatible):
    1. Title match: [[Профиль здоровья]] → note with that title
    2. Path-qualified: [[health/profile]] → health/profile.md exists
    3. Stem-only: [[profile]] → any */profile.md exists
    4. Relative path: [[../goals]] → normalized to [[goals]] then stem match
    """
    title_set = {n["title"].lower() for n in vault_notes.values() if n.get("title")}
    # "health/profile" from "health/profile.md"
    path_qualified = {rel[:-3].lower() for rel in vault_notes}
    # "profile" from any */profile.md
    stem_set = {Path(rel).stem.lower() for rel in vault_notes}

    dead: list[dict] = []
    for src, links in all_links.items():
        for link in links:
            base = link["base_title"]
            if not base:
                continue
            base_lower = base.lower()
            # Normalize relative paths: "../goals" → "goals", "./goals" → "goals"
            normalized = re.sub(r"^\.\.?/", "", base_lower)
            if (
                base_lower not in title_set
                and base_lower not in path_qualified
                and normalized not in stem_set
                and normalized not in path_qualified
            ):
                dead.append({"source": src, "target": base})

    fixed = 0
    if fix and dead:
        # Group by source file for clean_broken_links
        broken_map: dict[str, list[str]] = {}
        for d in dead:
            src_path = str(Path(vault_path) / d["source"])
            broken_map.setdefault(src_path, []).append(d["target"])
        # Deduplicate targets per file
        broken_map = {k: list(set(v)) for k, v in broken_map.items()}
        fixed = clean_broken_links(broken_map)

    return {"count": len(dead), "details": dead, "fixed": fixed}


# ---------------------------------------------------------------------------
# CHECK 2: Orphan notes
# ---------------------------------------------------------------------------

def check_orphan_notes(
    vault_notes: dict[str, dict],
    all_links: dict[str, list[dict]],
) -> dict:
    """Notes with 0 incoming links AND 0 outgoing links.

    Exempts notes with needs_review: true in frontmatter (per CLAUDE.md policy).
    """
    import yaml as _yaml

    # Build incoming map: title_lower -> set of source rel_paths
    incoming: dict[str, set] = {}
    for src, links in all_links.items():
        for link in links:
            t = link["base_title"].lower()
            incoming.setdefault(t, set()).add(src)

    has_outgoing = set(all_links.keys())
    structural_prefixes = ("templates/", "goals/")

    orphans: list[dict] = []
    exempt: list[dict] = []

    for rel, note in vault_notes.items():
        if rel.startswith(structural_prefixes):
            continue
        title_lower = note["title"].lower() if note.get("title") else ""
        has_in = title_lower in incoming
        has_out = rel in has_outgoing
        if not has_in and not has_out:
            # Check if note has needs_review: true (exempt orphan)
            is_exempt = False
            raw = note.get("content", "")
            if raw.startswith("---"):
                end = raw.find("---", 3)
                if end != -1:
                    try:
                        meta = _yaml.safe_load(raw[3:end])
                        if isinstance(meta, dict) and meta.get("needs_review"):
                            is_exempt = True
                    except Exception:
                        pass
            entry = {"path": rel, "title": note.get("title", "")}
            if is_exempt:
                exempt.append(entry)
            else:
                orphans.append(entry)

    return {
        "count": len(orphans),
        "details": orphans,
        "exempt_count": len(exempt),
        "exempt_details": exempt,
    }


def _fix_orphan_notes(orphans: list[dict], vault_path: str) -> int:
    """Orphan notes are report-only, no auto-fix. Returns 0."""
    return 0


# ---------------------------------------------------------------------------
# CHECK 3: LightRAG orphans
# ---------------------------------------------------------------------------

def check_lightrag_orphans(vault_notes: dict[str, dict], fix: bool = False) -> dict:
    """Docs in LightRAG index whose file_path no longer exists in the vault.

    Uses file_path matching (not content hash) so file edits don't create
    false orphan reports.
    """
    vault_paths = set(vault_notes.keys())
    indexed_paths = get_indexed_paths()  # {file_path: doc_id}

    orphan_doc_ids = [
        doc_id for fp, doc_id in indexed_paths.items()
        if fp not in vault_paths
    ]

    fixed = 0
    if fix:
        for did in orphan_doc_ids:
            if delete_doc(did):
                fixed += 1

    return {"count": len(orphan_doc_ids), "details": orphan_doc_ids, "fixed": fixed}


# ---------------------------------------------------------------------------
# CHECK 4: Vault orphans (not in LightRAG)
# ---------------------------------------------------------------------------

def check_vault_orphans(vault_notes: dict[str, dict], fix: bool = False) -> dict:
    """Vault notes not indexed in LightRAG (by file_path, not content hash).

    Content-hash matching breaks whenever a file is edited (e.g. daemon adds
    wiki-links). File_path matching is stable across edits.
    """
    indexed_paths = get_indexed_paths()  # {file_path: doc_id}

    orphans: list[dict] = []
    for rel, note in vault_notes.items():
        if note.get("doc_id") and rel not in indexed_paths:
            orphans.append({"path": rel, "title": note.get("title", ""), "doc_id": note["doc_id"]})

    fixed = 0
    if fix:
        for o in orphans:
            note = vault_notes[o["path"]]
            if note.get("content"):
                try:
                    lightrag_insert(note["content"], file_path=o["path"])
                    fixed += 1
                except Exception as e:
                    logger.warning("Failed to insert vault orphan %s: %s", o["path"], e)

    return {"count": len(orphans), "details": orphans, "fixed": fixed}


# ---------------------------------------------------------------------------
# CHECK 5: Stale anchors
# ---------------------------------------------------------------------------

def check_stale_anchors(
    vault_notes: dict[str, dict],
    all_links: dict[str, list[dict]],
    vault_path: str,
    fix: bool = False,
) -> dict:
    """Links with anchor (#Section) where target note exists but heading doesn't."""
    title_to_note: dict[str, dict] = {}
    for note in vault_notes.values():
        if note.get("title"):
            title_to_note[note["title"].lower()] = note

    stale: list[dict] = []
    for src, links in all_links.items():
        for link in links:
            if not link.get("anchor"):
                continue
            target = title_to_note.get(link["base_title"].lower())
            if not target:
                continue  # dead link, handled by check 1
            heading_set = {h.lower() for h in target.get("headings", [])}
            if link["anchor"].lower() not in heading_set:
                stale.append({
                    "source": src,
                    "target": link["base_title"],
                    "anchor": link["anchor"],
                })

    fixed = 0
    if fix and stale:
        # Remove anchor portion from links in source files
        fixes_by_file: dict[str, list[dict]] = {}
        for s in stale:
            src_path = str(Path(vault_path) / s["source"])
            fixes_by_file.setdefault(src_path, []).append(s)

        for file_path, items in fixes_by_file.items():
            try:
                content = Path(file_path).read_text(encoding="utf-8")
                original = content
                for item in items:
                    # Replace [[Title#Anchor]] with [[Title]], also handle |display
                    old_pattern = re.compile(
                        r"\[\["
                        + re.escape(item["target"])
                        + r"#"
                        + re.escape(item["anchor"])
                        + r"(\|[^\]]*)?"
                        + r"\]\]"
                    )
                    def _repl(m):
                        display = m.group(1) or ""
                        return f"[[{item['target']}{display}]]"
                    content = old_pattern.sub(_repl, content)
                if content != original:
                    Path(file_path).write_text(content, encoding="utf-8")
                    fixed += sum(1 for i in items)
            except Exception as e:
                logger.warning("Failed to fix stale anchors in %s: %s", file_path, e)

    return {"count": len(stale), "details": stale, "fixed": fixed}


# ---------------------------------------------------------------------------
# CHECK 6: Title-path mismatch
# ---------------------------------------------------------------------------

def check_title_path_mismatch(vault_notes: dict[str, dict]) -> dict:
    """Slugified title doesn't match filename stem.

    Skips by design: English filename stem (ASCII) + Russian/non-ASCII title.
    Per CLAUDE.md rule: "Filenames: kebab-case, English. Title in frontmatter, not filename."
    """
    mismatches: list[dict] = []
    for rel, note in vault_notes.items():
        title = note.get("title", "")
        if not title:
            continue
        stem = Path(rel).stem
        title_slug = _slugify(title)
        if not title_slug or stem == title_slug:
            continue
        # By design: English stem + non-ASCII (Russian) slug → skip
        if stem.replace("-", "").replace("_", "").isascii() and not title_slug.replace("-", "").isascii():
            continue
        mismatches.append({"path": rel, "title": title, "slug": title_slug, "stem": stem})

    return {"count": len(mismatches), "details": mismatches}


# ---------------------------------------------------------------------------
# CHECK 7: Unlinked entities (best-effort)
# ---------------------------------------------------------------------------

def check_unlinked_entities() -> dict:
    """Try to find entities with 0 relations from LightRAG graph."""
    try:
        from .lightrag_engine import get_instance
        rag = get_instance()
        # Try to access the graph through storage
        # Best-effort: return empty if graph not accessible
        return {"count": 0, "details": [], "note": "graph inspection not available in current mode"}
    except Exception:
        return {"count": 0, "details": [], "note": "LightRAG not available"}


# ---------------------------------------------------------------------------
# CHECK 8: Missing definitions (warnings only, does not affect total_issues)
# ---------------------------------------------------------------------------

def _load_definition_titles_lint(vault_path: str) -> list[tuple[str, list[str]]]:
    """Return (title, aliases) pairs from knowledge/definitions/*.md."""
    defs_dir = Path(vault_path) / "knowledge" / "definitions"
    if not defs_dir.exists():
        return []
    result = []
    for md_file in defs_dir.glob("*.md"):
        text = md_file.read_text("utf-8")
        fm = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
        if not fm:
            continue
        body = fm.group(1)
        title_m = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', body, re.MULTILINE)
        if not title_m:
            continue
        title = title_m.group(1).strip()
        aliases_m = re.search(r'^aliases:\s*\[(.+?)\]', body, re.MULTILINE)
        aliases = []
        if aliases_m:
            aliases = [a.strip().strip('"\'') for a in aliases_m.group(1).split(",")]
        result.append((title, aliases))
    return result


def _entity_matches_def(name: str, defs: list[tuple[str, list[str]]]) -> bool:
    name_lower = name.lower()
    for title, aliases in defs:
        if title.lower() == name_lower:
            return True
        if any(a.lower() == name_lower for a in aliases):
            return True
    return False


def check_missing_definitions(vault_path: str | None = None) -> dict:
    """Entities present in 3+ docs that have no knowledge/definitions/ file.

    Returns warnings only — count is always 0 so it never contributes to
    total_issues and never blocks the pipeline.
    """
    vp = vault_path or VAULT_PATH
    from .lightrag_engine import _get_config
    cfg = _get_config()
    ec_file = Path(cfg["working_dir"]) / "kv_store_entity_chunks.json"
    if not ec_file.exists():
        return {"count": 0, "details": [], "warnings": [], "warnings_count": 0}

    try:
        entity_chunks = json.loads(ec_file.read_text("utf-8"))
    except Exception:
        return {"count": 0, "details": [], "warnings": [], "warnings_count": 0}

    defs = _load_definition_titles_lint(vp)
    warnings = []
    for name, data in entity_chunks.items():
        occurrences = len(data.get("chunk_ids", []))
        if occurrences >= 3 and not _entity_matches_def(name, defs):
            warnings.append({"entity": name, "occurrences": occurrences})

    warnings.sort(key=lambda x: -x["occurrences"])
    return {"count": 0, "details": [], "warnings": warnings, "warnings_count": len(warnings)}


def check_duplicate_entities(vault_path: str | None = None, fix: bool = False) -> dict:
    """Check for duplicate entity names in the LightRAG graph.

    AUTO clusters (case variants, definition aliases, slug variants) are
    counted as issues and fixed when fix=True. WARN clusters (fuzzy matches)
    are always report-only.

    Returns:
        {
          "count": int,        # number of AUTO clusters (0 after fix)
          "details": [...],    # WARN clusters (always)
          "fixed": int,        # AUTO clusters merged (only when fix=True)
          "merged": [...],     # merge details
        }
    """
    vp = vault_path or VAULT_PATH
    try:
        from .graph_dedup import run_dedup
        result = run_dedup(vault_path=vp, dry_run=not fix)
    except Exception as e:
        logger.warning("check_duplicate_entities failed: %s", e)
        return {"count": 0, "details": [], "fixed": 0, "merged": []}

    auto_count = result.get("auto_clusters", 0)
    merged = result.get("merged", [])
    warn = result.get("warnings", [])

    return {
        "count": 0 if fix else auto_count,
        "details": warn,
        "fixed": len(merged) if fix else 0,
        "merged": merged,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_lint(vault_path: str | None = None, fix: bool = False) -> dict:
    """Run all 7 integrity checks. If fix=True, auto-heal where safe."""
    vp = vault_path or VAULT_PATH
    results: dict = {}
    fixed_count = 0

    # Collect vault data once
    vault_notes = _collect_vault_notes(vp)
    all_links = _collect_all_links(vp)

    # CHECK 1: Dead links
    r1 = check_dead_links(vault_notes, all_links, vp, fix=fix)
    results["dead_links"] = r1
    fixed_count += r1.get("fixed", 0)

    # CHECK 2: Orphan notes
    r2 = check_orphan_notes(vault_notes, all_links)
    results["orphan_notes"] = r2
    if fix and r2["details"]:
        fixed_count += _fix_orphan_notes(r2["details"], vp)
        r2["fixed"] = 0

    # CHECK 3: LightRAG orphans
    r3 = check_lightrag_orphans(vault_notes, fix=fix)
    results["lightrag_orphans"] = r3
    fixed_count += r3.get("fixed", 0)

    # CHECK 4: Vault orphans
    r4 = check_vault_orphans(vault_notes, fix=fix)
    results["vault_orphans"] = r4
    fixed_count += r4.get("fixed", 0)

    # CHECK 5: Stale anchors
    r5 = check_stale_anchors(vault_notes, all_links, vp, fix=fix)
    results["stale_anchors"] = r5
    fixed_count += r5.get("fixed", 0)

    # CHECK 6: Title-path mismatch
    r6 = check_title_path_mismatch(vault_notes)
    results["title_mismatches"] = r6

    # CHECK 7: Unlinked entities (report only)
    r7 = check_unlinked_entities()
    results["unlinked_entities"] = r7

    # CHECK 8: Missing definitions (warnings only, excluded from total_issues)
    r8 = check_missing_definitions(vp)
    results["missing_definitions"] = r8

    # CHECK 9: Duplicate entities in graph (AUTO fixed when fix=True)
    r9 = check_duplicate_entities(vault_path=vp, fix=fix)
    results["duplicate_entities"] = r9
    fixed_count += r9.get("fixed", 0)

    results["total_issues"] = sum(
        r.get("count", 0) for k, r in results.items()
        if isinstance(r, dict) and "count" in r and k not in ("missing_definitions",)
    )
    results["fixed"] = fixed_count
    return results
