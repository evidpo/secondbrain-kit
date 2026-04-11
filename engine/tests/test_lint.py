"""Tests for vault lint checks."""

import sys
import types
from unittest.mock import MagicMock, patch

# Stub heavy dependencies
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


def _make_vault(tmp_path, notes: dict[str, str]):
    """Create vault structure with notes. notes = {rel_path: content}."""
    for rel, content in notes.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def test_dead_links_detected(tmp_path):
    _make_vault(tmp_path, {
        "knowledge/a.md": '---\ntitle: "Note A"\n---\n# Note A\n\nSee [[Nonexistent]].\n',
        "knowledge/b.md": '---\ntitle: "Note B"\n---\n# Note B\n\nSee [[Note A]].\n',
    })
    from src.lint import _collect_vault_notes, _collect_all_links, check_dead_links
    notes = _collect_vault_notes(str(tmp_path))
    links = _collect_all_links(str(tmp_path))
    result = check_dead_links(notes, links, str(tmp_path))

    assert result["count"] >= 1
    targets = [d["target"] for d in result["details"]]
    assert "Nonexistent" in targets


def test_orphan_notes_detected(tmp_path):
    _make_vault(tmp_path, {
        "knowledge/linked.md": '---\ntitle: "Linked"\n---\n# Linked\n\nSee [[Other]].\n',
        "knowledge/other.md": '---\ntitle: "Other"\n---\n# Other\n\nContent.\n',
        "knowledge/orphan.md": '---\ntitle: "Orphan"\n---\n# Orphan\n\nNo links here.\n',
    })
    from src.lint import _collect_vault_notes, _collect_all_links, check_orphan_notes
    notes = _collect_vault_notes(str(tmp_path))
    links = _collect_all_links(str(tmp_path))
    result = check_orphan_notes(notes, links)

    orphan_titles = [o["title"].lower() for o in result["details"]]
    assert "orphan" in orphan_titles


def test_title_path_mismatch(tmp_path):
    import yaml as _yaml_mod
    _orig = _yaml_mod.safe_load
    _yaml_mod.safe_load = lambda s: {"title": "New Name"} if "New Name" in str(s) else {}

    _make_vault(tmp_path, {
        "knowledge/old-name.md": '---\ntitle: "New Name"\n---\n# New Name\n\nContent.\n',
    })
    from src.lint import _collect_vault_notes, check_title_path_mismatch
    notes = _collect_vault_notes(str(tmp_path))
    result = check_title_path_mismatch(notes)

    _yaml_mod.safe_load = _orig

    assert result["count"] >= 1
    assert result["details"][0]["stem"] == "old-name"
    assert result["details"][0]["slug"] == "new-name"


def test_run_lint_returns_totals(tmp_path):
    _make_vault(tmp_path, {
        "knowledge/a.md": '---\ntitle: "A"\n---\n# A\n\nJust a simple note with enough words to pass checks.\n',
    })
    from src.lint import run_lint
    with patch("src.lint.get_indexed_doc_ids", return_value={}), \
         patch("src.lint.check_unlinked_entities", return_value={"count": 0, "details": []}):
        result = run_lint(str(tmp_path))

    assert "total_issues" in result
    assert "fixed" in result
    assert isinstance(result["total_issues"], int)
