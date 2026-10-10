"""Resumable download and one-time extraction of the BBBC041 archive."""
from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

CHUNK = 1 << 20  # 1 MiB


def remote_size(url: str) -> int:
    """Return Content-Length of `url` via a HEAD request."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def download(url: str, dest: Path) -> Path:
    """Download `url` to `dest`, resuming a partial `<dest>.part` via HTTP Range.

    Skips the download if `dest` already exists with the expected size.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)

    if dest.exists():
        if dest.stat().st_size == total:
            log.info("Archive already present (%d bytes): %s", total, dest)
            return dest
        raise RuntimeError(f"{dest} exists but has size {dest.stat().st_size} != {total}")

    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    if done > total:
        log.warning("Partial file larger than remote, restarting: %s", part)
        part.unlink()
        done = 0

    if done < total:
        log.info("Downloading %s -> %s (resume at %d / %d bytes)", url, part, done, total)
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        with urllib.request.urlopen(req, timeout=60) as resp:
            if done and resp.status != 206:
                raise RuntimeError(f"Server did not honour Range request (status {resp.status})")
            with part.open("ab") as f:
                last_report = done
                while chunk := resp.read(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if done - last_report >= 100 * CHUNK:
                        log.info("  %.1f%% (%d MiB)", 100 * done / total, done >> 20)
                        last_report = done

    size = part.stat().st_size
    if size != total:
        raise RuntimeError(f"Size mismatch after download: {size} != {total}; rerun to resume")
    part.replace(dest)
    log.info("Download complete: %s (%d bytes)", dest, total)
    return dest


def extract_once(zip_path: Path, extract_dir: Path) -> Path:
    """Extract `zip_path` into `extract_dir` unless a `.extracted` marker is present."""
    extract_dir = Path(extract_dir)
    marker = extract_dir / f".{Path(zip_path).stem}.extracted"
    if marker.exists():
        log.info("Already extracted (marker %s)", marker)
        return extract_dir
    log.info("Extracting %s -> %s", zip_path, extract_dir)
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt member in zip: {bad}")
        # skip macOS resource-fork junk (__MACOSX/) shipped inside the BBBC041 zip
        members = [m for m in zf.infolist() if not m.filename.startswith("__MACOSX/")]
        for member in members:
            # guard against path traversal in archive member names
            target = (extract_dir / member.filename).resolve()
            if not target.is_relative_to(extract_dir.resolve()):
                raise RuntimeError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(extract_dir, members=members)
        log.info("Extracted %d members (%d __MACOSX entries skipped)",
                 len(members), len(zf.infolist()) - len(members))
    marker.write_text("ok\n", encoding="utf-8")
    return extract_dir


def fetch_dataset(url: str, zip_path: Path, extract_dir: Path) -> Path:
    """Download (resumable) and extract the dataset archive; idempotent."""
    zip_path = Path(zip_path)
    download(url, zip_path)
    return extract_once(zip_path, Path(extract_dir))

