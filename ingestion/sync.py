"""Doc sync — discover source documents in a corpus directory.

For the practice build the corpus is a set of local Markdown / HTML / PDF
files under ``data/corpus/``. A ``sync_from_git`` helper is provided for
future use (clone a docs repo into the corpus dir), but the default path
is local-folder discovery so the pipeline is testable without network
access.

The discovered files are returned as absolute paths sorted by relative
path for deterministic ingestion order.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".html", ".htm", ".pdf"})
"""File extensions the parser can handle."""


def discover_files(corpus_dir: str | Path) -> list[Path]:
    """Return all supported source files under ``corpus_dir``, sorted.

    Hidden files and common VCS / build dirs (``.git``, ``node_modules``)
    are skipped so a cloned repo's docs folder can be pointed at directly.
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")

    skip_dirs = {".git", "node_modules", "__pycache__", ".venv"}
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(path)
    return found


def sync_from_git(repo_url: str, dest: str | Path, *, sparse: str | None = None) -> Path:
    """Clone a git repo into ``dest`` and return the docs path.

    Args:
        repo_url: HTTPS git URL.
        dest: parent directory to clone into.
        sparse: optional subdirectory inside the repo to keep (sparse checkout).
            Useful for cloning a large repo but keeping only ``docs/``.

    The clone is skipped if ``dest/<repo-name>`` already exists.
    """
    dest = Path(dest)
    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    target = dest / repo_name
    if target.exists():
        logger.info("Repo already cloned at %s; skipping", target)
        return target

    dest.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = ["git", "clone", "--depth", "1", repo_url, str(target)]
    logger.info("Cloning %s → %s", repo_url, target)
    subprocess.run(cmd, check=True, capture_output=True)

    if sparse:
        subprocess.run(
            ["git", "-C", str(target), "sparse-checkout", "set", sparse],
            check=True,
            capture_output=True,
        )
    return target
