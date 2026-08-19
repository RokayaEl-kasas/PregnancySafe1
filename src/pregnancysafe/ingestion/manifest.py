"""Parses a download log (like the one produced by the user's batch
downloader) into a {relative_path: source_url} manifest.

Expected log line formats (see the sample the user uploaded):
    DOWNLOADED: data\\raw\\uti\\medications\\cephalexin\\pdf.pdf <- https://...
    EXISTS: data\\raw\\gestational_diabetes\\medications\\glyburide\\pdf.pdf
    FAILED: https://pubmed.ncbi.nlm.nih.gov/31302868/ | HTTP 203
    SKIPPED / UNCLASSIFIED: ...

Only DOWNLOADED lines carry a usable path->url mapping; EXISTS lines repeat
a path already seen (no new URL, safe to ignore); FAILED/SKIPPED lines have
no local file at all. Windows-style backslashes in the log are normalized
to forward slashes so paths compare correctly regardless of the OS the
ingestion script itself runs on.
"""

from __future__ import annotations

import re
from pathlib import Path

_DOWNLOADED_RE = re.compile(r"^DOWNLOADED:\s*(?P<path>\S+)\s*<-\s*(?P<url>\S+)\s*$")


def _normalize(path_str: str) -> str:
    """Windows backslashes -> forward slashes, so a log written on Windows
    matches files found via pathlib on any OS."""
    return path_str.replace("\\", "/").strip()


def load_download_manifest(log_path: Path) -> dict[str, str]:
    """Parse a download log into {normalized_relative_path: source_url}.

    Returns an empty dict (never raises) if the log doesn't exist — ingestion
    should still work without a manifest, just with less precise citations.
    """
    if not log_path.exists():
        return {}

    manifest: dict[str, str] = {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            match = _DOWNLOADED_RE.match(line.strip())
            if not match:
                continue
            manifest[_normalize(match.group("path"))] = match.group("url")
    return manifest


def lookup_source_url(manifest: dict[str, str], file_path: Path, repo_root: Path) -> str | None:
    """Look up a file's source URL in the manifest by its path relative to
    repo_root, tolerant of the manifest being recorded relative to a
    slightly different root (matches on path suffix if an exact match fails)."""
    try:
        rel = _normalize(str(file_path.relative_to(repo_root)))
    except ValueError:
        rel = _normalize(str(file_path))

    if rel in manifest:
        return manifest[rel]

    # Fallback: match by suffix (e.g. manifest recorded "data/raw/..." but
    # file_path is an absolute path) — helps when the log was generated on a
    # different machine/drive than where ingestion now runs.
    for logged_path, url in manifest.items():
        if rel.endswith(logged_path) or logged_path.endswith(rel):
            return url
    return None
