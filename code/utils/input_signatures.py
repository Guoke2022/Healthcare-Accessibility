# -*- coding: utf-8 -*-
"""Semantic input signatures for cache dependency separation.

The project has several prepared tables that contain fields used by different
stages.  A byte-level hash of the whole table is safe but can invalidate an
expensive stage for an irrelevant change (for example, changing hospital beds
should not rebuild road routing).  This module hashes only the columns that
actually define a stage input.

Signatures are deterministic with respect to row order after sorting by the
stable ID, numeric dtype, and missing-value representation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils.cache import json_load, meta_path_for


SIGNATURE_SCHEMA_VERSION = 1


def _update_array(h: "hashlib._Hash", label: str, values, dtype: str) -> None:
    arr = np.asarray(values, dtype=np.dtype(dtype))
    arr = np.ascontiguousarray(arr)
    h.update(label.encode("utf-8"))
    h.update(b"\0")
    h.update(np.dtype(dtype).str.encode("ascii"))
    h.update(b"\0")
    h.update(str(arr.shape).encode("ascii"))
    h.update(b"\0")
    h.update(arr.tobytes(order="C"))
    h.update(b"\0")


def _canonical_float(series: pd.Series | Iterable, dtype: str = "<f8") -> np.ndarray:
    # pandas 2.x / NumPy 2.x may expose a read-only ndarray from Series.to_numpy().
    # The signature routine canonicalizes non-finite values in-place, so always
    # materialize an owned, writable float64 buffer first.
    x = pd.to_numeric(
        pd.Series(series), errors="coerce"
    ).to_numpy(dtype=np.float64, copy=True)
    # Defensive fallback for unusual extension-array / NumPy combinations.
    if not x.flags.writeable:
        x = x.copy(order="C")
    # Canonicalize all non-finite values.  This avoids different NaN payload bits
    # producing different hashes for the same logical missing value.
    x[~np.isfinite(x)] = np.nan
    return x.astype(np.dtype(dtype), copy=False)


def hospital_routing_signature(df: pd.DataFrame) -> str:
    """Hash only fields that can change the 1_1 hospital routing input/pool."""
    need = {"source_row", "lon", "lat", "province_id_spatial", "is_valid_for_routing"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"hospital routing signature missing columns: {sorted(missing)}")
    x = df[list(need)].copy()
    x["source_row"] = pd.to_numeric(x["source_row"], errors="raise").astype(np.int64)
    x = x.sort_values("source_row", kind="mergesort").reset_index(drop=True)

    province = pd.to_numeric(x["province_id_spatial"], errors="coerce").fillna(-1).astype(np.int32)
    valid = x["is_valid_for_routing"].fillna(False).astype(bool).to_numpy(dtype=np.uint8)

    h = hashlib.sha256()
    h.update(f"hospital_routing_v{SIGNATURE_SCHEMA_VERSION}".encode("ascii")); h.update(b"\0")
    _update_array(h, "source_row", x["source_row"].to_numpy(), "<i8")
    _update_array(h, "lon", _canonical_float(x["lon"]), "<f8")
    _update_array(h, "lat", _canonical_float(x["lat"]), "<f8")
    _update_array(h, "province_id_spatial", province.to_numpy(), "<i4")
    _update_array(h, "is_valid_for_routing", valid, "u1")
    return h.hexdigest()


def hospital_supply_signature(df: pd.DataFrame) -> str:
    """Hash hospital row identity + beds used by 1_2 supply calculations."""
    need = {"source_row", "beds_std"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"hospital supply signature missing columns: {sorted(missing)}")
    x = df[list(need)].copy()
    x["source_row"] = pd.to_numeric(x["source_row"], errors="raise").astype(np.int64)
    x = x.sort_values("source_row", kind="mergesort").reset_index(drop=True)
    h = hashlib.sha256()
    h.update(f"hospital_supply_v{SIGNATURE_SCHEMA_VERSION}".encode("ascii")); h.update(b"\0")
    _update_array(h, "source_row", x["source_row"].to_numpy(), "<i8")
    _update_array(h, "beds_std", _canonical_float(x["beds_std"]), "<f8")
    return h.hexdigest()


def population_routing_signature(df: pd.DataFrame) -> str:
    """Hash populated grid support/coordinates used by 1_1; ignores pop values."""
    need = {"grid_id", "lon", "lat"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"population routing signature missing columns: {sorted(missing)}")
    x = df[list(need)].copy()
    x["grid_id"] = pd.to_numeric(x["grid_id"], errors="raise").astype(np.int64)
    x = x.sort_values("grid_id", kind="mergesort").reset_index(drop=True)
    h = hashlib.sha256()
    h.update(f"population_routing_v{SIGNATURE_SCHEMA_VERSION}".encode("ascii")); h.update(b"\0")
    _update_array(h, "grid_id", x["grid_id"].to_numpy(), "<i8")
    _update_array(h, "lon", _canonical_float(x["lon"]), "<f8")
    _update_array(h, "lat", _canonical_float(x["lat"]), "<f8")
    return h.hexdigest()


def population_accessibility_signature(df: pd.DataFrame) -> str:
    """Hash populated grids + values actually used by 1_2 accessibility."""
    need = {"grid_id", "population", "lon", "lat"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"population accessibility signature missing columns: {sorted(missing)}")
    x = df[list(need)].copy()
    x["grid_id"] = pd.to_numeric(x["grid_id"], errors="raise").astype(np.int64)
    x = x.sort_values("grid_id", kind="mergesort").reset_index(drop=True)
    h = hashlib.sha256()
    h.update(f"population_accessibility_v{SIGNATURE_SCHEMA_VERSION}".encode("ascii")); h.update(b"\0")
    _update_array(h, "grid_id", x["grid_id"].to_numpy(), "<i8")
    _update_array(h, "population", _canonical_float(x["population"]), "<f8")
    _update_array(h, "lon", _canonical_float(x["lon"]), "<f8")
    _update_array(h, "lat", _canonical_float(x["lat"]), "<f8")
    return h.hexdigest()


def reference_points_signature(df: pd.DataFrame) -> str:
    """Hash the exact fixed points/weights consumed by 0_3's Rust helper."""
    need = {"id", "point_type", "lon", "lat", "weight"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"reference point signature missing columns: {sorted(missing)}")
    x = df[list(need)].copy()
    x["id"] = pd.to_numeric(x["id"], errors="raise").astype(np.int64)
    x["point_type"] = pd.to_numeric(x["point_type"], errors="raise").astype(np.uint8)
    x = x.sort_values(["point_type", "id"], kind="mergesort").reset_index(drop=True)
    h = hashlib.sha256()
    h.update(f"road_audit_reference_points_v{SIGNATURE_SCHEMA_VERSION}".encode("ascii")); h.update(b"\0")
    _update_array(h, "id", x["id"].to_numpy(), "<i8")
    _update_array(h, "point_type", x["point_type"].to_numpy(), "u1")
    _update_array(h, "lon", _canonical_float(x["lon"]), "<f8")
    _update_array(h, "lat", _canonical_float(x["lat"]), "<f8")
    _update_array(h, "weight", _canonical_float(x["weight"]), "<f8")
    return h.hexdigest()


def _signature_from_meta(path: Path, key: str) -> str | None:
    meta = meta_path_for(Path(path))
    if not meta.exists():
        return None
    try:
        value = json_load(meta).get(key)
    except Exception:
        return None
    return str(value) if value else None


def population_routing_signature_for_path(path: Path) -> str:
    """Read the prepared-part signature, with a legacy on-the-fly fallback."""
    path = Path(path)
    cached = _signature_from_meta(path, "routing_signature")
    if cached:
        return cached
    df = pd.read_parquet(path, columns=["grid_id", "lon", "lat"])
    return population_routing_signature(df)


def population_accessibility_signature_for_path(path: Path) -> str:
    """Read the prepared-part accessibility signature, with legacy fallback."""
    path = Path(path)
    cached = _signature_from_meta(path, "accessibility_signature")
    if cached:
        return cached
    df = pd.read_parquet(path, columns=["grid_id", "population", "lon", "lat"])
    return population_accessibility_signature(df)


def hospital_signatures_for_path(path: Path) -> tuple[str, str]:
    """Return (routing, supply) signatures; compute only if legacy meta lacks them."""
    path = Path(path)
    routing = _signature_from_meta(path, "routing_signature")
    supply = _signature_from_meta(path, "supply_signature")
    if routing and supply:
        return routing, supply
    df = pd.read_parquet(path)
    return routing or hospital_routing_signature(df), supply or hospital_supply_signature(df)
