# SecondBrain Hooks for Claude Code

## Setup

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{
      "command": "python3 ~/coding/secondbrain-engine/hooks/secondbrain-session-start.py",
      "timeout": 15000
    }],
    "SessionEnd": [{
      "command": "python3 ~/coding/secondbrain-engine/hooks/secondbrain-session-end.py",
      "timeout": 10000
    }],
    "PreCompact": [{
      "command": "python3 ~/coding/secondbrain-engine/hooks/secondbrain-session-compact.py",
      "timeout": 10000
    }]
  }
}
```

## How it works

- **session-start**: Reads `_index.md` from vault, injects into Claude Code context. Zero API calls.
- **session-end**: Captures last 30 turns from transcript, writes to vault `_inbox/` as session note.
- **pre-compact**: Same as session-end but fires before context auto-compaction. Min 5 turns.

## Environment

- `VAULT_PATH`: Path to Obsidian vault (default: `~/coding/SecondBrain`)
- `INBOX_DIR_NAME`: Inbox folder name (default: `_inbox`)
- `CLAUDE_INVOKED_BY`: If set, hooks exit immediately (recursion guard)
