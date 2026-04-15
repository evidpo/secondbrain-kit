"""Codebase-info sync: update vault notes when session repo metadata changes.

Called at the start of session processing (once per session file).
Text generation delegated to local Ollama (qwen3:14b) — zero API cost.

On VPS the repo paths (/Users/...) don't exist → function silently skips.
On local dev VAULT_PATH = ~/coding/SecondBrain and repos are accessible.
"""

import hashlib
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import URLError

import yaml

logger = logging.getLogger(__name__)

METADATA_FILES = ["CLAUDE.md", "pyproject.toml", "package.json"]
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:14b"
OLLAMA_TIMEOUT = 300  # seconds — qwen3 is slow


def _extract_working_dir(session_text: str) -> str | None:
    """Extract 'Project: /path' line from session content."""
    match = re.search(r'^Project:\s*(.+)$', session_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def _repo_name(working_dir: str) -> str:
    return Path(working_dir).name


def _compute_hashes(repo_path: Path) -> dict[str, str]:
    """MD5 hashes of known metadata files in repo root."""
    hashes: dict[str, str] = {}
    for fname in METADATA_FILES:
        fp = repo_path / fname
        if fp.exists() and fp.is_file():
            try:
                hashes[fname] = hashlib.md5(fp.read_bytes()).hexdigest()
            except Exception:
                pass
    return hashes


def _stored_hashes(note_path: Path) -> dict[str, str]:
    """Read metadata_hashes from note frontmatter."""
    try:
        content = note_path.read_text(encoding="utf-8")
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                meta = yaml.safe_load(content[3:end]) or {}
                return meta.get("metadata_hashes") or {}
    except Exception:
        pass
    return {}


def _extract_links_section(content: str) -> str:
    """Preserve ## Связи section from existing note."""
    match = re.search(r'(## Связи\s*\n(?:.*\n)*?)(?=\n## |\Z)', content)
    if match:
        return match.group(1).rstrip("\n")
    return "## Связи"


def _read_meta(note_path: Path) -> dict:
    """Read frontmatter dict from existing note (for 'created' field preservation)."""
    try:
        content = note_path.read_text(encoding="utf-8")
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                return yaml.safe_load(content[3:end]) or {}
    except Exception:
        pass
    return {}


def _read_metadata_files(repo_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for fname in METADATA_FILES:
        fp = repo_path / fname
        if fp.exists():
            try:
                result[fname] = fp.read_text(encoding="utf-8", errors="replace")[:2000]
            except Exception:
                pass
    return result


def _read_top_dirs(repo_path: Path) -> list[str]:
    dirs: list[str] = []
    try:
        for item in sorted(repo_path.iterdir()):
            if item.name.startswith("."):
                continue
            dirs.append(item.name + "/" if item.is_dir() else item.name)
            if len(dirs) >= 15:
                break
    except Exception:
        pass
    return dirs


def _get_remote(repo_path: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_path, capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _call_ollama(prompt: str) -> str:
    """POST to local Ollama, return response text or empty string on error."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3},
    }).encode()
    req = urllib_request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "")
    except URLError as e:
        logger.warning("Ollama unavailable: %s", e)
        return ""
    except Exception as e:
        logger.warning("Ollama error: %s", e)
        return ""


def _generate_sections(repo_name: str, metadata: dict[str, str], top_dirs: list[str]) -> dict:
    """Ask qwen3:14b for note sections. Returns dict or {} on failure."""
    meta_str = "\n\n".join(
        f"=== {fname} ===\n{content}"
        for fname, content in metadata.items()
    )[:3000]

    prompt = f"""You are generating a codebase-info note for an Obsidian vault. Write in Russian.

Repository: {repo_name}
Top-level entries: {', '.join(top_dirs[:12])}

Metadata files:
{meta_str}

Return ONLY a JSON object (no markdown fences, no thinking) with these keys:
- "purpose": 1-2 sentence description of what this repo does (Russian)
- "stack": one line listing the tech stack, comma-separated (Russian OK for labels, English for names)
- "key_paths": markdown bullet list (3-6 items) of important directories/files with descriptions (Russian)
- "stack_list": JSON array of stack item strings for frontmatter (e.g. ["python", "fastapi"])
"""

    raw = _call_ollama(prompt)
    if not raw:
        return {}

    # Strip <think>...</think> blocks (qwen3 extended thinking)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Extract first JSON object
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        logger.warning("No JSON in Ollama response for %s", repo_name)
        return {}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed for %s: %s", repo_name, e)
        return {}


def _render_note(
    repo_name: str,
    repo_path: Path,
    generated: dict,
    hashes: dict[str, str],
    links_section: str,
    created: str,
    remote: str,
) -> str:
    """Render full codebase-info note with frontmatter."""
    stack_list = generated.get("stack_list", [])
    stack_fm = ", ".join(stack_list)

    hashes_yaml = "\n".join(f'  {k}: "{v}"' for k, v in sorted(hashes.items()))

    repo_dir = str(repo_path).replace(str(Path.home()), "~")

    purpose = generated.get("purpose", "").strip()
    stack_text = generated.get("stack", "").strip()
    key_paths = generated.get("key_paths", "").strip()

    frontmatter = (
        f'---\n'
        f'title: "{repo_name}"\n'
        f'type: project\n'
        f'tags: [codebase]\n'
        f'created: {created}\n'
        f'source: scan\n'
        f'path: "{repo_dir}"\n'
        f'repo: "{remote}"\n'
        f'stack: [{stack_fm}]\n'
        f'status: active\n'
        f'metadata_hashes:\n'
        f'{hashes_yaml}\n'
        f'---'
    )

    return (
        f"{frontmatter}\n\n"
        f"# {repo_name}\n\n"
        f"## Назначение\n\n{purpose}\n\n"
        f"## Стек\n\n{stack_text}\n\n"
        f"## Ключевые пути\n\n{key_paths}\n\n"
        f"{links_section}\n"
    )


def _update_index(vault: Path, repo_name: str, stack_text: str) -> None:
    """Add or update row for this repo in codebase-info/index.md."""
    index_path = vault / "codebase-info" / "index.md"
    if not index_path.exists():
        return

    content = index_path.read_text(encoding="utf-8")
    key = repo_name.lower()
    note_link = f"[[codebase-{key}|{repo_name}]]"
    new_row = f"| {note_link} | {stack_text} | active |"

    if f"[[codebase-{key}" in content:
        content = re.sub(
            rf'\| \[\[codebase-{re.escape(key)}[^\n]+',
            new_row,
            content,
        )
    else:
        # Append before trailing newline
        content = content.rstrip("\n") + "\n" + new_row + "\n"

    index_path.write_text(content, encoding="utf-8")
    logger.info("index.md updated for %s", repo_name)


def maybe_sync_codebase_info(session_text: str, vault_path: str) -> None:
    """Check if codebase-info note needs update and update/create it.

    Called once per session file at the start of _process_session.
    - Extracts working_directory from 'Project: /path' line.
    - Computes MD5 hashes of CLAUDE.md / pyproject.toml / package.json.
    - Skips if hashes unchanged (no-op).
    - Delegates text generation to local Ollama (qwen3:14b).
    - Updates note in place (preserving ## Связи).
    - Creates note directly in codebase-info/ if it does not exist.
    - Updates index.md.
    Silently skips when repo path is inaccessible (e.g. on VPS).
    """
    working_dir = _extract_working_dir(session_text)
    if not working_dir:
        return

    repo_path = Path(working_dir)
    if not repo_path.exists() or not repo_path.is_dir():
        logger.debug("Repo path inaccessible, skipping codebase-info sync: %s", working_dir)
        return

    repo_name = _repo_name(working_dir)
    vault = Path(vault_path)
    note_path = vault / "codebase-info" / f"codebase-{repo_name.lower()}.md"

    # Compute current hashes
    new_hashes = _compute_hashes(repo_path)
    if not new_hashes:
        logger.debug("No metadata files for %s, skipping", repo_name)
        return

    note_exists = note_path.exists()

    # Check if update is needed
    if note_exists:
        if new_hashes == _stored_hashes(note_path):
            logger.debug("codebase-info up to date: %s", repo_name)
            return
        logger.info("codebase-info metadata changed, updating: %s", repo_name)
    else:
        logger.info("codebase-info missing, creating: %s", repo_name)

    # Preserve existing links section and created date
    links_section = "## Связи"
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if note_exists:
        existing = note_path.read_text(encoding="utf-8")
        links_section = _extract_links_section(existing)
        meta = _read_meta(note_path)
        created = meta.get("created", created) or created

    # Read repo metadata
    metadata = _read_metadata_files(repo_path)
    top_dirs = _read_top_dirs(repo_path)
    remote = _get_remote(repo_path)

    # Delegate generation to Ollama
    generated = _generate_sections(repo_name, metadata, top_dirs)
    if not generated:
        logger.warning("Ollama generation empty for %s, skipping update", repo_name)
        return

    # Render and write
    content = _render_note(repo_name, repo_path, generated, new_hashes,
                           links_section, created, remote)

    if note_exists:
        note_path.write_text(content, encoding="utf-8")
        logger.info("Updated codebase-info: %s", note_path.name)
    else:
        # Create directly in codebase-info/ (already well-formed, no pipeline needed)
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(content, encoding="utf-8")
        logger.info("Created codebase-info: %s", note_path.name)

    # Update index
    _update_index(vault, repo_name, generated.get("stack", ""))
