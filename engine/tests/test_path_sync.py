"""Regression tests for path_sync ambiguity guard.

Two notes can legitimately share a frontmatter title (e.g. a concept
definition and a codebase card). When one of them is renamed, the
daemon must NOT rewrite [[old_title]] occurrences across the vault,
because some of those links still legitimately resolve to the OTHER
note holding the title.
"""
import re
import shutil
import tempfile
from pathlib import Path

import pytest

from src import path_sync
from src.path_sync import FrontmatterCache


def _real_extract_title(content: str) -> str:
    """Regex-based frontmatter title extractor.

    Other test files (e.g. test_content_type.py) replace `yaml` in
    sys.modules with a MagicMock at collection time; by the time our
    tests run, path_sync.yaml.safe_load returns mocks. We monkeypatch
    path_sync._extract_title with this regex implementation so our
    frontmatter parsing stays deterministic regardless of test order.
    """
    if not content.startswith("---"):
        return ""
    end = content.find("---", 3)
    if end == -1:
        return ""
    for line in content[3:end].splitlines():
        m = re.match(r"\s*title:\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip().strip("\"'")
    return ""


@pytest.fixture
def vault(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setattr(path_sync, "VAULT_PATH", str(tmp))
    monkeypatch.setattr(path_sync, "_notify_telegram", lambda msg: None)
    monkeypatch.setattr(path_sync, "_reindex_in_lightrag", lambda *a, **kw: None)
    monkeypatch.setattr(path_sync, "_extract_title", _real_extract_title)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def _write_note(path: Path, title: str, body: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: {title}\n---\n# {title}\n{body}\n", encoding="utf-8")


def _build_cache(vault: Path) -> FrontmatterCache:
    cache = FrontmatterCache(cache_file=str(vault / ".fc.json"))
    cache.build(str(vault))
    return cache


def test_paths_with_title_returns_all_holders(vault):
    cache = FrontmatterCache(cache_file=str(vault / ".fc.json"))
    cache.set("a.md", "Same", "h1")
    cache.set("b.md", "Same", "h2")
    cache.set("c.md", "Other", "h3")

    assert sorted(cache.paths_with_title("Same")) == ["a.md", "b.md"]
    assert cache.paths_with_title("Other") == ["c.md"]
    assert cache.paths_with_title("Missing") == []


def test_handle_modify_skips_rewrite_when_old_title_still_held(vault):
    _write_note(vault / "concepts" / "sb.md", "SecondBrain", "concept body")
    _write_note(vault / "codebase" / "cb-sb.md", "SecondBrain", "card body")
    _write_note(vault / "concepts" / "linker.md", "Linker", "See [[SecondBrain]] for context.")
    cache = _build_cache(vault)

    cb = vault / "codebase" / "cb-sb.md"
    cb.write_text("---\ntitle: cb-secondbrain\n---\n# cb\ncard body\n", encoding="utf-8")
    path_sync.handle_modify(str(cb), cache)

    linker_text = (vault / "concepts" / "linker.md").read_text(encoding="utf-8")
    assert "[[SecondBrain]]" in linker_text, "wiki-link to concept must be preserved"
    assert "[[cb-secondbrain]]" not in linker_text, "must not redirect link to renamed note"
    assert cache.get("codebase/cb-sb.md")["title"] == "cb-secondbrain"
    # Concept note's title and cache entry must be untouched.
    assert cache.get("concepts/sb.md")["title"] == "SecondBrain"


def test_handle_modify_proceeds_when_old_title_unique(vault):
    _write_note(vault / "concepts" / "alpha.md", "Alpha", "body")
    _write_note(vault / "concepts" / "linker.md", "Linker", "See [[Alpha]].")
    cache = _build_cache(vault)

    a = vault / "concepts" / "alpha.md"
    a.write_text("---\ntitle: Beta\n---\n# Beta\nbody\n", encoding="utf-8")
    path_sync.handle_modify(str(a), cache)

    linker_text = (vault / "concepts" / "linker.md").read_text(encoding="utf-8")
    assert "[[Beta]]" in linker_text, "unique-old-title rename must propagate"
    assert "[[Alpha]]" not in linker_text
    assert cache.get("concepts/alpha.md")["title"] == "Beta"


def test_handle_modify_skips_when_new_title_already_taken(vault):
    _write_note(vault / "concepts" / "alpha.md", "Alpha", "body")
    _write_note(vault / "concepts" / "beta.md", "Beta", "body")
    _write_note(vault / "concepts" / "linker.md", "Linker", "See [[Alpha]].")
    cache = _build_cache(vault)

    a = vault / "concepts" / "alpha.md"
    a.write_text("---\ntitle: Beta\n---\n# Beta\nbody\n", encoding="utf-8")
    path_sync.handle_modify(str(a), cache)

    linker_text = (vault / "concepts" / "linker.md").read_text(encoding="utf-8")
    assert "[[Alpha]]" in linker_text, "new-title-already-taken guard must keep links"
    assert cache.get("concepts/alpha.md")["title"] == "Beta"
