"""Tests for src/gate.py — input quality gates."""

import os
import tempfile
import pytest

# Set VAULT_PATH before importing gate
_tmpdir = tempfile.mkdtemp()
os.environ["VAULT_PATH"] = _tmpdir

from src.gate import (
    check_file_hash,
    check_size,
    check_content_quality,
    check_title_exists,
    mark_processed,
    run_all_gates,
    _hash_text,
)


class TestFileHashDedup:
    def test_first_seen_passes(self):
        ok, reason = check_file_hash("unique text that has never been seen")
        assert ok is True
        assert reason == ""

    def test_after_mark_rejects(self):
        text = "this exact text will be marked as processed"
        mark_processed(text)
        ok, reason = check_file_hash(text)
        assert ok is False
        assert "file_hash_duplicate" in reason

    def test_different_text_passes(self):
        mark_processed("text A for hash dedup")
        ok, _ = check_file_hash("text B completely different")
        assert ok is True

    def test_hash_deterministic(self):
        assert _hash_text("hello") == _hash_text("hello")
        assert _hash_text("hello") != _hash_text("world")


class TestSizeGate:
    def test_too_short_rejects(self):
        ok, reason = check_size("too short")
        assert ok is False
        assert "too_short" in reason

    def test_normal_passes(self):
        text = " ".join(["word"] * 50)
        ok, reason = check_size(text)
        assert ok is True

    def test_too_long_rejects(self):
        text = " ".join(["word"] * 6000)
        ok, reason = check_size(text)
        assert ok is False
        assert "too_long" in reason

    def test_exact_minimum_passes(self):
        text = " ".join(["word"] * 20)
        ok, _ = check_size(text)
        assert ok is True


class TestContentQuality:
    def test_normal_text_passes(self):
        text = """
        Сегодня мы обсуждали архитектуру нового проекта.
        Решили использовать микросервисный подход с Docker.
        Основной стек будет Python и FastAPI для бэкенда.
        Фронтенд на React с TypeScript.
        """
        ok, _ = check_content_quality(text)
        assert ok is True

    def test_code_dump_rejects(self):
        text = "\n".join([
            "import os",
            "from pathlib import Path",
            "def main():",
            "    return None",
            "class Foo:",
            "    def bar(self):",
            "        return 42",
            "if __name__ == '__main__':",
            "    main()",
            "import sys",
        ])
        ok, reason = check_content_quality(text)
        assert ok is False
        assert "code_or_logs" in reason

    def test_log_dump_rejects(self):
        text = "\n".join([
            "2026-04-06T14:00:00 ERROR Failed to connect",
            "2026-04-06T14:00:01 WARNING Retrying...",
            "2026-04-06T14:00:02 INFO Connected",
            "2026-04-06T14:00:03 DEBUG Sending request",
            "Traceback (most recent call last):",
            'File "/app/main.py", line 42',
            "Exception: connection timeout",
            "2026-04-06T14:00:04 ERROR Fatal",
        ])
        ok, reason = check_content_quality(text)
        assert ok is False

    def test_mixed_content_passes(self):
        text = """
        Мы решили переписать авторизацию потому что старая не поддерживает OAuth.
        Дима начнёт с бэкенда на следующей неделе.
        Нужно использовать JWT токены с ротацией каждые 24 часа.
        Виктория сделает UI для логина и регистрации.
        Дедлайн — конец апреля, но может сдвинуться.
        """
        ok, _ = check_content_quality(text)
        assert ok is True


class TestTitleDedup:
    def test_new_title_passes(self):
        ok, _ = check_title_exists("completely-new-unique-slug-xyz")
        assert ok is True

    def test_existing_title_rejects(self):
        # Create a file first
        from pathlib import Path
        # Notes live in vault root now
        (Path(_tmpdir) / "existing-note.md").write_text("content")

        ok, reason = check_title_exists("existing-note")
        assert ok is False
        assert "title_exists" in reason


class TestRunAllGates:
    def test_good_text_passes(self):
        text = " ".join(["Хорошая заметка о проекте номер"] * 10)
        ok, reason = run_all_gates(text, "test.md")
        assert ok is True

    def test_empty_rejects_at_size(self):
        ok, reason = run_all_gates("hi", "test.md")
        assert ok is False
        assert "too_short" in reason

    def test_rejection_logged(self):
        from pathlib import Path
        run_all_gates("short", "test-reject.md")
        log_path = Path(_tmpdir) / "rejected.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "test-reject.md" in content
