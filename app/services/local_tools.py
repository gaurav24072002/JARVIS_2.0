"""Local PC-control tools for JARVIS.

A fixed, whitelisted set of filesystem tools — no raw shell, no arbitrary
commands. Every path is resolved and must live under one of the allowed
base directories (default: the user's home directory), so a mis-parsed
brain query cannot write or delete outside the user's own space.

Add new tools by adding a function and a TOOL_HANDLERS entry; the brain
prompt, decision types, and executor are the only other touch points.
"""
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Dict

logger = logging.getLogger("J.A.R.V.I.S")

# Where tool operations are allowed. Home covers the JARVIS project itself
# (/home/<user>/JARVIS) and everything the user owns day-to-day.
ALLOWED_BASES = [
    Path.home().resolve(),
]

# Keep brain-supplied paths from smuggling in shell metacharacters/newlines.
_VALID_PATH_RE = re.compile(r"^[A-Za-z0-9._\-/ ~]+$")

# Common folder names the user might say instead of a real path.
COMMON_DIRS = {
    "home": "", "my home": "", "my home folder": "", "my home directory": "",
    "documents": "Documents", "docs": "Documents",
    "my documents": "Documents", "my documents folder": "Documents",
    "downloads": "Downloads", "my downloads": "Downloads", "my downloads folder": "Downloads",
    "desktop": "Desktop", "my desktop": "Desktop", "my desktop folder": "Desktop",
    "pictures": "Pictures", "my pictures": "Pictures", "my pictures folder": "Pictures",
    "music": "Music", "my music": "Music", "my music folder": "Music",
    "videos": "Videos", "my videos": "Videos", "my videos folder": "Videos",
}


class ToolError(Exception):
    """Raised when a tool refuses or fails an operation."""


def _resolve(path: str) -> Path:
    """Expand, absolutize, and validate a path against the allowed bases."""
    raw = (path or "").strip().strip('"').strip("'").strip()
    if not raw:
        raise ToolError("No path was given.")

    # "my documents" / "downloads" / "desktop" → well-known home subfolders.
    common = COMMON_DIRS.get(raw.lower().strip())
    if common is not None:
        target = (Path.home() / common).resolve()
        logger.info("[TOOL] resolved common dir %r -> %s", raw, target)
        return target

    if not _VALID_PATH_RE.match(raw):
        raise ToolError(f"Path contains characters I won't touch: {raw[:60]!r}")

    expanded = os.path.expanduser(raw)
    if not os.path.isabs(expanded):
        expanded = str(Path.cwd() / expanded)

    resolved = Path(expanded).resolve()

    for base in ALLOWED_BASES:
        if resolved == base or base in resolved.parents:
            return resolved

    bases = ", ".join(str(b) for b in ALLOWED_BASES)
    raise ToolError(
        f"I can only work inside: {bases}. "
        f"({resolved} is outside that.)"
    )


def create_folder(path: str) -> str:
    target = _resolve(path)
    target.mkdir(parents=True, exist_ok=True)
    logger.info("[TOOL] create_folder -> %s", target)
    return f"Created folder {target}."


def create_file(path: str) -> str:
    target = _resolve(path)
    if target.exists():
        return f"{target} already exists — I left it untouched."
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    logger.info("[TOOL] create_file -> %s", target)
    return f"Created file {target}."


def list_folder(path: str) -> str:
    target = _resolve(path)
    if not target.is_dir():
        if target.exists():
            return f"{target} is a file, not a folder. Use delete_file to remove it."
        return f"Folder {target} doesn't exist yet."
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    shown = entries[:40]
    lines = [p.name + ("/" if p.is_dir() else "") for p in shown]
    suffix = f" (and {len(entries) - len(shown)} more)" if len(entries) > len(shown) else ""
    if not lines:
        return f"Folder {target} is empty."
    return f"In {target}:\n" + "\n".join(f"- {l}" for l in lines) + suffix


def delete_file(path: str) -> str:
    target = _resolve(path)
    if not target.exists():
        return f"{target} doesn't exist — nothing to delete."
    if target.is_dir():
        return f"{target} is a folder — I won't remove folders with the file tool. Say 'delete folder {path}' instead."
    target.unlink()
    logger.info("[TOOL] delete_file -> %s", target)
    return f"Deleted {target}."


def delete_folder(path: str) -> str:
    target = _resolve(path)
    if not target.exists():
        return f"{target} doesn't exist — nothing to delete."
    if not target.is_dir():
        return f"{target} is a file — I won't remove files with the folder tool. Say 'delete file {path}' instead."
    shutil.rmtree(target)
    logger.info("[TOOL] delete_folder -> %s", target)
    return f"Deleted folder {target} and everything inside it."


TOOL_HANDLERS: Dict[str, Callable[[str], str]] = {
    "create_folder": create_folder,
    "create_file": create_file,
    "list_folder": list_folder,
    "delete_file": delete_file,
    "delete_folder": delete_folder,
}


def available_tools() -> str:
    return ", ".join(TOOL_HANDLERS.keys())


def run_tool(tool: str, path: str) -> str:
    """Run a whitelisted tool and return a human-readable result string.

    Raises ToolError for unknown tools or refused paths.
    """
    handler = TOOL_HANDLERS.get(tool)
    if handler is None:
        raise ToolError(f"Unknown tool '{tool}'. Available: {available_tools()}")
    return handler(path)
