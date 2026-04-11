#!/usr/bin/env python3
"""Claude Code session-start hook: inject vault context into every session.

Reads _index.md from vault and outputs it as hookSpecificOutput.
Zero API calls, pure file I/O, under 1 second.
"""

import json
import os
from pathlib import Path


def main():
    vault_path = os.environ.get("VAULT_PATH", os.path.expanduser("~/coding/SecondBrain"))
    index_path = Path(vault_path) / "_index.md"

    if not index_path.exists():
        return

    try:
        content = index_path.read_text(encoding="utf-8")
        if len(content) > 20000:
            content = content[:20000] + "\n... (truncated)"
    except Exception:
        return

    print(json.dumps({"hookSpecificOutput": content}))


if __name__ == "__main__":
    main()
