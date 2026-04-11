"""Tests for compute_doc_id and strip_frontmatter hash unification."""

import sys
import types

# Stub lightrag and its submodules so lightrag_engine can import without the real package.
for mod_name in ("lightrag", "lightrag.llm", "lightrag.llm.gemini", "lightrag.utils"):
    sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

# Provide the names that lightrag_engine imports at module level.
_lr = sys.modules["lightrag"]
if not hasattr(_lr, "LightRAG"):
    _lr.LightRAG = type("LightRAG", (), {})
if not hasattr(_lr, "QueryParam"):
    _lr.QueryParam = type("QueryParam", (), {})

_gemini = sys.modules["lightrag.llm.gemini"]
if not hasattr(_gemini, "gemini_model_complete"):
    _gemini.gemini_model_complete = None
if not hasattr(_gemini, "gemini_embed"):
    _gemini.gemini_embed = None

_utils = sys.modules["lightrag.utils"]
if not hasattr(_utils, "EmbeddingFunc"):
    _utils.EmbeddingFunc = type("EmbeddingFunc", (), {})

from src.lightrag_engine import compute_doc_id, strip_frontmatter


def test_same_body_same_id():
    note1 = "---\ntitle: A\n---\n\nBody text here"
    note2 = "---\ntitle: B\ntags: [x]\n---\n\nBody text here"
    assert compute_doc_id(note1) == compute_doc_id(note2)


def test_different_body_different_id():
    note1 = "---\ntitle: A\n---\n\nBody one"
    note2 = "---\ntitle: A\n---\n\nBody two"
    assert compute_doc_id(note1) != compute_doc_id(note2)


def test_no_frontmatter():
    note = "Just plain text"
    doc_id = compute_doc_id(note)
    assert doc_id.startswith("doc-")


def test_doc_id_format():
    doc_id = compute_doc_id("some content")
    assert doc_id.startswith("doc-")
    assert len(doc_id) == 4 + 32  # "doc-" + md5 hex
