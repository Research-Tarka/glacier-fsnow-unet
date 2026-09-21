"""Generic download/extract helpers for one-shot external data sources.

Purpose
-------
Static feature sources (Koppen-Geiger, WorldClim, permafrost) each need to
fetch one external archive or raster once and cache it locally, unlike the
per-year Earth Engine sources in ``gee_runner.py``. This module is the shared
HTTP/FTP fetch-and-extract layer both static and any future direct-download
temporal source can reuse, so each source module only has to describe *what*
to fetch, not *how* to fetch it reliably (retries, resuming past a completed
download, archive extraction).

Inputs / outputs
-----------------
Pure filesystem + network I/O; no Earth Engine calls.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import time
import zipfile
from ftplib import FTP
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

HTTP_TIMEOUT_S = 180
HTTP_CHUNK_BYTES = 1024 * 1024
HTTP_RETRIES = 3


def guess_filename(url: str, *, default_name: str = "download.bin") -> str:
    name = Path(urlparse(str(url)).path).name.strip()
    return name or default_name


def _ftp_download(url: str, dest: Path) -> None:
    parsed = urlparse(url)
    with FTP(parsed.hostname, timeout=HTTP_TIMEOUT_S) as ftp:
        ftp.login(parsed.username or "anonymous", parsed.password or "")
        with dest.open("wb") as handle:
            ftp.retrbinary(f"RETR {parsed.path}", handle.write, blocksize=HTTP_CHUNK_BYTES)


def download_file(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download ``url`` to ``dest``, retrying transient failures.

    A cached file at ``dest`` is reused unless ``force``. Writes to a
    ``.part`` sibling first so a crash mid-download cannot leave a
    truncated file mistaken for a complete one on the next run.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log.info("Cache hit, skipping download: %s", dest.name)
        return dest

    scheme = urlparse(url).scheme.lower()
    last_error: Exception | None = None
    for attempt in range(1, HTTP_RETRIES + 1):
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.unlink(missing_ok=True)
        try:
            log.info("Downloading %s -> %s (attempt %d/%d)", url, dest.name, attempt, HTTP_RETRIES)
            if scheme == "ftp":
                _ftp_download(url, tmp)
            else:
                with requests.get(url, timeout=HTTP_TIMEOUT_S, stream=True) as response:
                    response.raise_for_status()
                    content_type = str(response.headers.get("Content-Type", "")).lower()
                    if "text/html" in content_type and dest.suffix.lower() not in {".html", ".htm"}:
                        raise RuntimeError(f"Server returned HTML instead of data for {url}")
                    with tmp.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=HTTP_CHUNK_BYTES):
                            if chunk:
                                handle.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt >= HTTP_RETRIES:
                break
            wait_s = min(10, attempt * 2)
            log.warning("Download attempt %d/%d failed (%s); retrying in %ds", attempt, HTTP_RETRIES, exc, wait_s)
            time.sleep(wait_s)
    raise RuntimeError(f"Download failed after {HTTP_RETRIES} attempts: {url}") from last_error


def extract_archive(archive_path: Path, dest_dir: Path, *, force: bool = False) -> Path:
    """Extract a zip/tar archive into ``dest_dir``, skipping if already done."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / f".extracted-{archive_path.name}.ok"
    if marker.exists() and not force:
        return dest_dir
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    marker.write_text("ok\n", encoding="utf-8")
    return dest_dir


def fetch_and_extract(
    urls: Sequence[str],
    cache_root: Path,
    *,
    force: bool = False,
) -> Path:
    """Download every URL into ``cache_root/downloads`` and extract archives
    into ``cache_root/extracted``, returning the extraction directory.

    Non-archive downloads (e.g. a bare ``.tif``) are copied straight into the
    extraction directory so callers only ever need to glob one place.
    """
    download_dir = cache_root / "downloads"
    extract_dir = cache_root / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    any_ready = False
    for url in urls:
        path = download_file(url, download_dir / guess_filename(url), force=force)
        if zipfile.is_zipfile(path) or tarfile.is_tarfile(path):
            extract_archive(path, extract_dir, force=force)
        else:
            target = extract_dir / path.name
            if force or not target.exists():
                shutil.copy2(path, target)
        any_ready = True

    if not any_ready:
        raise FileNotFoundError(f"No files were fetched into {extract_dir}")
    return extract_dir


def first_match(root: Path, *patterns: str) -> Path | None:
    """Return the first file under ``root`` matching any of ``patterns``."""
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None
