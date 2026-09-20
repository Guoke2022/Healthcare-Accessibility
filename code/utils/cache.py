# -*- coding: utf-8 -*-
"""Module utilities for cache."""


from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Iterable

_FILE_STAMP_CACHE: dict[tuple, dict] = {}
_FILE_STAMP_CACHE_LOCK = threading.Lock()

def json_load(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        path.name + f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def stable_hash(obj) -> str:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _strong_hash_threshold_bytes() -> int:
    raw = os.environ.get("NC_CACHE_STRONG_HASH_MAX_MB", "32")
    try:
        mb = max(0.0, float(raw))
    except Exception:
        mb = 32.0
    return int(mb * 1024 * 1024)


def file_stamp(path: Path) -> dict:
    """Return a cache identity for a file.

    Small files use content SHA-256 (path+size+sha), so copying/re-writing a file
    with an unchanged logical content does not invalidate downstream caches just
    because its mtime changed.  Large files use path+size+mtime by default to
    avoid expensive full scans; set NC_CACHE_HASH_LARGE_FILES=1 for a strict
    publication/finalization run if desired.
    """
    path = Path(path)
    resolved = path.resolve()
    st = path.stat()
    force_large = os.environ.get("NC_CACHE_HASH_LARGE_FILES", "0").strip().lower() in {"1", "true", "yes", "y"}
    strong = force_large or int(st.st_size) <= _strong_hash_threshold_bytes()
    cache_key = (
        str(resolved), int(st.st_size), int(st.st_mtime_ns),
        int(getattr(st, "st_ctime_ns", 0)), strong,
    )
    with _FILE_STAMP_CACHE_LOCK:
        cached = _FILE_STAMP_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    out = {
        "path": str(resolved),
        "size": int(st.st_size),
        "identity_mode": "sha256" if strong else "size_mtime",
    }
    if strong:
        out["sha256"] = file_sha256(path)
    else:
        out["mtime_ns"] = int(st.st_mtime_ns)
    with _FILE_STAMP_CACHE_LOCK:
        _FILE_STAMP_CACHE[cache_key] = dict(out)
    return out


def build_fingerprint(*, config: dict, files: Iterable[Path] = ()) -> str:
    payload = {
        "config": config,
        "files": [file_stamp(Path(p)) for p in files],
    }
    return stable_hash(payload)


def meta_path_for(output_path: Path) -> Path:
    return Path(str(output_path) + ".meta.json")


def _cache_event(payload: dict) -> None:
    root = os.environ.get("NC_CACHE_EVENT_DIR", "").strip()
    if not root:
        return
    try:
        d = Path(root)
        d.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_ns": time.time_ns(),
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            **payload,
        }
        name = f"{event['timestamp_ns']}_{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex[:8]}.json"
        json_dump(event, d / name)
    except Exception:
        # Provenance logging must never break a scientific computation.
        pass


def cache_valid(
    output_path: Path,
    fingerprint: str,
    *,
    extra_outputs: Iterable[Path] = (),
    event_payload: dict | None = None,
) -> bool:
    output_path = Path(output_path)
    meta_path = meta_path_for(output_path)
    reason = "fingerprint_match"
    valid = True
    if not output_path.exists():
        valid, reason = False, "missing_output"
    elif not meta_path.exists():
        valid, reason = False, "missing_meta"
    else:
        missing_extra = [str(Path(p)) for p in extra_outputs if not Path(p).exists()]
        if missing_extra:
            valid, reason = False, "missing_extra_output"
        else:
            try:
                meta = json_load(meta_path)
            except Exception:
                valid, reason = False, "unreadable_meta"
            else:
                if meta.get("input_fingerprint") != fingerprint:
                    valid, reason = False, "fingerprint_changed"

    if event_payload is not None:
        _cache_event({
            **event_payload,
            "output": str(output_path),
            "cache_status": "hit" if valid else "miss",
            "reason": reason,
            "input_fingerprint": fingerprint,
        })
    return valid


def write_cache_meta(
    output_path: Path,
    fingerprint: str,
    *,
    payload: dict | None = None,
) -> None:
    meta = {
        "input_fingerprint": fingerprint,
        **(payload or {}),
    }
    json_dump(meta, meta_path_for(Path(output_path)))


def stage_manifest_path(stage_dir: Path) -> Path:
    return Path(stage_dir) / "_stage_input.json"


def prepare_stage_directory(
    stage_dir: Path,
    fingerprint: str,
    *,
    run_policy: str = "auto_clean",
    payload: dict | None = None,
) -> str:
    """Helper for prepare_stage_directory."""


    stage_dir = Path(stage_dir)
    manifest = stage_manifest_path(stage_dir)

    old_fp = None
    if manifest.exists():
        try:
            old_fp = json_load(manifest).get("input_fingerprint")
        except Exception:
            old_fp = None

    if run_policy == "force_rebuild":
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        status = "cleaned_forced"
        reason = "force_rebuild"
    elif stage_dir.exists() and old_fp == fingerprint:
        status = "reused_same_input"
        reason = "stage_fingerprint_match"
    elif stage_dir.exists():
        shutil.rmtree(stage_dir)
        status = "cleaned_input_changed"
        reason = "stage_fingerprint_changed"
    else:
        status = "created"
        reason = "stage_directory_missing"

    stage_dir.mkdir(parents=True, exist_ok=True)

    _cache_event({
        **(payload or {}),
        "output": str(stage_dir),
        "cache_status": status,
        "reason": reason,
        "input_fingerprint": fingerprint,
        "previous_fingerprint": old_fp,
    })


    if status == "reused_same_input":
        return status

    json_dump(
        {
            "input_fingerprint": fingerprint,
            "run_policy": run_policy,
            **(payload or {}),
        },
        manifest,
    )
    return status


def clean_directory(path: Path) -> None:
    """Helper for clean_directory."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
