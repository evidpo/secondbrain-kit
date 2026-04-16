"""Graph entity deduplication via LightRAG WebUI API.

Three rules for AUTO-merge (no confirmation needed):
  Rule 1 — Case variants: "content loop" == "Content Loop"
  Rule 2 — Definition anchors: aliases in knowledge/definitions/*.md
  Rule 3 — Path/slug normalization: "mihailov-flow" == "@mihailov_flow"

Rule 4 — Fuzzy name overlap (Jaccard >= 0.70) → WARN only, no auto-merge.

Entry point: run_dedup(vault_path, dry_run=False) -> dict
"""

import logging
import os
import re
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Abbreviations: single tokens <= 5 chars with no spaces are skipped by Rule 4
_ABBREV_RE = re.compile(r"^[A-Z0-9]{2,5}$")


class GraphDeduplicator:
    def __init__(self, webui_url: str, api_key: str, vault_path: str):
        self.webui_url = webui_url.rstrip("/")
        self.api_key = api_key
        self.vault_path = vault_path
        self._headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # WebUI API calls
    # ------------------------------------------------------------------

    def get_all_labels(self) -> list[str]:
        """GET /graph/label/list → all entity names in the graph."""
        try:
            resp = requests.get(
                f"{self.webui_url}/graph/label/list",
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("graph_dedup: failed to get labels: %s", e)
            return []

    def entity_exists(self, name: str) -> bool:
        """GET /graph/entity/exists?name=... → bool."""
        try:
            resp = requests.get(
                f"{self.webui_url}/graph/entity/exists",
                params={"name": name},
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("exists", False)
        except Exception as e:
            logger.warning("graph_dedup: entity_exists failed: %s", e)
            return False

    def merge(self, sources: list[str], target: str) -> dict:
        """POST /graph/entities/merge → merge source entities into target."""
        payload = {"entities_to_change": sources, "entity_to_change_into": target}
        try:
            resp = requests.post(
                f"{self.webui_url}/graph/entities/merge",
                json=payload,
                headers=self._headers,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("graph_dedup: merge failed %s → %s: %s", sources, target, e)
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Definition anchors (Rule 2)
    # ------------------------------------------------------------------

    def _load_definitions(self) -> dict[str, list[str]]:
        """Load knowledge/definitions/*.md → {canonical_title: [alias, ...]}.

        Returns lowercase keys for case-insensitive matching.
        """
        defs: dict[str, list[str]] = {}
        defs_dir = Path(self.vault_path) / "knowledge" / "definitions"
        if not defs_dir.exists():
            return defs

        for md_file in sorted(defs_dir.glob("*.md")):
            text = md_file.read_text("utf-8")
            fm_match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
            if not fm_match:
                continue
            fm = fm_match.group(1)
            title_m = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', fm, re.MULTILINE)
            if not title_m:
                continue
            canonical = title_m.group(1).strip()
            aliases_m = re.search(r'^aliases:\s*\[(.+?)\]', fm, re.MULTILINE)
            aliases: list[str] = []
            if aliases_m:
                raw = aliases_m.group(1)
                aliases = [a.strip().strip("\"'") for a in raw.split(",") if a.strip()]
            defs[canonical] = aliases

        return defs

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_case(label: str) -> str:
        """Lowercase + strip for Rule 1 grouping."""
        return label.lower().strip()

    @staticmethod
    def _normalize_slug(label: str) -> str:
        """Strip @, replace -_/ with space, lowercase — for Rule 3."""
        s = label.lstrip("@")
        s = re.sub(r"[-_/]", " ", s)
        return s.lower().strip()

    @staticmethod
    def _is_path_like(label: str) -> bool:
        """True if label looks like a path/slug: contains /  or starts with @ or contains - without spaces."""
        if label.startswith("@"):
            return True
        if "/" in label:
            return True
        if "-" in label and " " not in label:
            return True
        return False

    @staticmethod
    def _canonical_case(group: list[str]) -> str:
        """Pick the canonical form from a case-variant group.

        Preference order:
          1. The form that already uses TitleCase (each word capitalized)
          2. The longest form
          3. First alphabetically
        """
        def score(s: str) -> tuple:
            is_title = s == s.title()
            return (is_title, len(s))
        return max(group, key=score)

    @staticmethod
    def _tokens(label: str) -> set[str]:
        """Tokenize label into lowercase words >= 3 chars (for Jaccard)."""
        return {w.lower() for w in re.split(r"\W+", label) if len(w) >= 3}

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # ------------------------------------------------------------------
    # Core: find_clusters
    # ------------------------------------------------------------------

    def find_clusters(self, labels: list[str]) -> dict:
        """Analyse labels and return merge clusters.

        Returns:
            {
              "auto": [(canonical, [sources_to_merge]), ...],
              "warn": [(canonical, [sources], reason), ...],
            }
        """
        auto: list[tuple[str, list[str]]] = []
        warn: list[tuple[str, list[str], str]] = []
        used: set[str] = set()  # labels already assigned to a cluster

        defs = self._load_definitions()

        # --- Rule 2: Definition anchors (highest priority) ---
        for canonical, aliases in defs.items():
            all_aliases_lower = {a.lower() for a in aliases}
            all_aliases_lower.add(canonical.lower())

            matches = [
                lbl for lbl in labels
                if lbl.lower() in all_aliases_lower and lbl != canonical
            ]
            if matches:
                # Ensure canonical exists in the graph before merging
                if canonical not in labels:
                    # canonical might not be in graph — skip (entity may not be indexed yet)
                    logger.debug("graph_dedup Rule2: canonical %r not in graph, skipping", canonical)
                    continue
                sources = [m for m in matches if m not in used and m != canonical]
                if sources:
                    auto.append((canonical, sources))
                    used.update(sources)
                    used.add(canonical)

        # --- Rule 1: Case variants ---
        case_groups: dict[str, list[str]] = {}
        for lbl in labels:
            if lbl in used:
                continue
            key = self._normalize_case(lbl)
            case_groups.setdefault(key, []).append(lbl)

        for key, group in case_groups.items():
            if len(group) < 2:
                continue
            canonical = self._canonical_case(group)
            sources = [g for g in group if g != canonical and g not in used]
            if sources:
                auto.append((canonical, sources))
                used.update(sources)
                used.add(canonical)

        # --- Rule 3: Path/slug normalization ---
        # Build a lookup: normalized_slug → label (for non-path labels)
        slug_map: dict[str, str] = {}
        for lbl in labels:
            if lbl in used or self._is_path_like(lbl):
                continue
            slug_map[self._normalize_slug(lbl)] = lbl

        for lbl in labels:
            if lbl in used or not self._is_path_like(lbl):
                continue
            norm = self._normalize_slug(lbl)
            if norm in slug_map:
                # path-like label is a duplicate of a plain label
                canonical = slug_map[norm]
                # If canonical itself is path-like, prefer the @ form
                if lbl.startswith("@") and not canonical.startswith("@"):
                    canonical, lbl = lbl, canonical
                if lbl not in used and canonical not in used:
                    auto.append((canonical, [lbl]))
                    used.add(lbl)
                    used.add(canonical)
            else:
                # Check if two path-like labels normalize to the same slug
                pass

        # Also: two path-like labels with same normalized slug
        path_slug_groups: dict[str, list[str]] = {}
        for lbl in labels:
            if lbl in used or not self._is_path_like(lbl):
                continue
            norm = self._normalize_slug(lbl)
            path_slug_groups.setdefault(norm, []).append(lbl)

        for norm, group in path_slug_groups.items():
            if len(group) < 2:
                continue
            # Prefer @ form, else longest
            canonical = next((g for g in group if g.startswith("@")), max(group, key=len))
            sources = [g for g in group if g != canonical and g not in used]
            if sources and canonical not in used:
                auto.append((canonical, sources))
                used.update(sources)
                used.add(canonical)

        # --- Rule 4: Fuzzy name overlap (WARN only) ---
        remaining = [lbl for lbl in labels if lbl not in used]
        # Skip single-token abbreviations
        candidates = [
            lbl for lbl in remaining
            if not _ABBREV_RE.match(lbl) and " " in lbl  # only multi-word
        ]

        checked: set[frozenset] = set()
        for i, a in enumerate(candidates):
            tok_a = self._tokens(a)
            if len(tok_a) < 2:
                continue
            for b in candidates[i + 1:]:
                pair = frozenset([a, b])
                if pair in checked:
                    continue
                checked.add(pair)
                if b in used:
                    continue
                tok_b = self._tokens(b)
                if len(tok_b) < 2:
                    continue
                score = self._jaccard(tok_a, tok_b)
                if score >= 0.70:
                    # Prefer longer as canonical (more specific)
                    canonical = a if len(a) >= len(b) else b
                    other = b if canonical == a else a
                    warn.append((canonical, [other], f"Jaccard={score:.2f}"))

        return {"auto": auto, "warn": warn}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_dedup(vault_path: str, dry_run: bool = False) -> dict:
    """Run entity deduplication against the LightRAG WebUI.

    Args:
        vault_path: Path to the vault (for loading definitions).
        dry_run: If True, find clusters but don't execute merges.

    Returns:
        {
          "total_labels": int,
          "auto_clusters": int,
          "warn_clusters": int,
          "merged": [...],     # actual merge results (empty if dry_run)
          "skipped": [...],    # clusters skipped in dry_run
          "warnings": [...],   # WARN-only clusters (never merged)
          "dry_run": bool,
        }
    """
    webui_url = os.getenv("LIGHTRAG_WEBUI_URL", "http://secondbrain-webui:9621")
    api_key = os.getenv("SECONDBRAIN_API_KEY", "")

    d = GraphDeduplicator(webui_url=webui_url, api_key=api_key, vault_path=vault_path)
    labels = d.get_all_labels()
    if not labels:
        return {
            "total_labels": 0,
            "auto_clusters": 0,
            "warn_clusters": 0,
            "merged": [],
            "skipped": [],
            "warnings": [],
            "dry_run": dry_run,
        }

    clusters = d.find_clusters(labels)
    merged = []
    skipped = []

    for canonical, sources in clusters["auto"]:
        if dry_run:
            skipped.append({"canonical": canonical, "sources": sources})
            continue
        result = d.merge(sources, canonical)
        status = result.get("status", "error")
        merged.append({
            "canonical": canonical,
            "sources": sources,
            "status": status,
            "message": result.get("message", ""),
        })
        if status != "success":
            logger.warning("graph_dedup: merge failed: %s → %s: %s", sources, canonical, result)
        else:
            logger.info("graph_dedup: merged %s → %s", sources, canonical)
        time.sleep(0.3)  # avoid overwhelming WebUI

    return {
        "total_labels": len(labels),
        "auto_clusters": len(clusters["auto"]),
        "warn_clusters": len(clusters["warn"]),
        "merged": merged,
        "skipped": skipped,
        "warnings": [
            {"canonical": c, "sources": s, "reason": r}
            for c, s, r in clusters["warn"]
        ],
        "dry_run": dry_run,
    }
