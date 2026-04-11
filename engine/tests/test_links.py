"""Tests for wiki-link validation and cleanup."""

import re
import sys
import types
from unittest.mock import MagicMock

# Stub heavy dependencies before importing src modules
_yaml_stub = types.ModuleType("yaml")
_yaml_stub.safe_load = MagicMock(return_value={})
sys.modules.setdefault("yaml", _yaml_stub)

_google = types.ModuleType("google")
_genai = types.ModuleType("google.genai")
_genai_types = types.ModuleType("google.genai.types")
_google.genai = _genai
_genai.Client = MagicMock
_genai.types = _genai_types
_genai_types.GenerateContentConfig = MagicMock
sys.modules.setdefault("google", _google)
sys.modules.setdefault("google.genai", _genai)
sys.modules.setdefault("google.genai.types", _genai_types)

_httpx = types.ModuleType("httpx")
_httpx.post = MagicMock()
sys.modules.setdefault("httpx", _httpx)


def test_link_validation_filters_nonexistent(tmp_path):
    """Links to nonexistent notes should be filtered out."""
    # Create a mock vault with one note
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "existing-note.md").write_text("---\ntitle: Existing Note\n---\nContent")

    from src.linker import get_existing_note_titles, invalidate_cache
    invalidate_cache()  # clear any cached state from prior tests
    titles = get_existing_note_titles(str(tmp_path))
    existing_lower = {t.lower() for t in titles}

    links = ["existing note", "nonexistent note", "another missing"]
    filtered = [l for l in links if l.lower() in existing_lower]

    assert filtered == ["existing note"]
    assert "nonexistent note" not in filtered
    invalidate_cache()


def test_scan_broken_links_with_anchor():
    """[[Note#Section]] should match when Note is deleted."""
    from src.link_integrity import WIKI_LINK_RE

    content = "See [[Deleted Note#Overview]] and [[Deleted Note|alias]] for details."
    matches = WIKI_LINK_RE.findall(content)

    # Extract base titles (same logic as scan_broken_links)
    base_titles = []
    for raw_link in matches:
        base = raw_link.split("#")[0].split("|")[0].strip()
        base_titles.append(base)

    assert "Deleted Note" in base_titles
    assert len(base_titles) == 2


def test_scan_broken_links_finds_anchored_refs(tmp_path):
    """scan_broken_links should detect [[Title#Section]] and [[Title|alias]] as broken."""
    from src.link_integrity import scan_broken_links

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "note.md").write_text(
        "# Test\n\nSee [[Gone#Overview]] and [[Gone|alias]] here.\n"
    )

    result = scan_broken_links(str(tmp_path), ["Gone"])
    assert len(result) == 1
    file_path = list(result.keys())[0]
    assert result[file_path] == ["Gone"]


def test_clean_broken_links_with_anchor(tmp_path):
    """Clean should remove [[Title#anchor]] and [[Title|display]] links."""
    from src.link_integrity import clean_broken_links

    test_file = tmp_path / "test.md"
    test_file.write_text(
        "# Test\n\nSee [[Gone#Section]] for details.\n\n## Links\n- [[Gone|display]]\n- [[Kept]]\n"
    )

    cleaned = clean_broken_links({str(test_file): ["Gone"]})
    content = test_file.read_text()

    assert "[[Gone" not in content
    assert "[[Kept]]" in content
    assert cleaned >= 2
