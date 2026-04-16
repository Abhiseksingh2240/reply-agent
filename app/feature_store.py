"""Persist and load normalized feature tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.normalizer import normalize_bundle
from app.io_loader import RawDatasetBundle
from app.utils import ensure_dir


def _fingerprint(data_dir: Path) -> str:
    tx = data_dir / "transactions.csv"
    h = hashlib.sha256()
    if tx.exists():
        h.update(tx.read_bytes()[:2_000_000])
    for name in ["users.json", "locations.json", "mails.json", "sms.json"]:
        p = data_dir / name
        if p.exists():
            h.update(p.name.encode())
            h.update(str(p.stat().st_mtime_ns).encode())
            h.update(p.read_bytes()[:500_000])
    return h.hexdigest()[:16]


def cache_paths(cache_root: Path, data_dir: Path) -> dict[str, Path]:
    fp = _fingerprint(data_dir)
    base = cache_root / f"{data_dir.name}_{fp}"
    ensure_dir(base)
    return {
        "dir": base,
        "bundle": base / "normalized_bundle.parquet",
        "meta": base / "meta.json",
    }


def save_normalized(cache_root: Path, data_dir: Path, tables: dict[str, Any]) -> Path:
    paths = cache_paths(cache_root, data_dir)
    tables["transactions"].to_parquet(paths["dir"] / "transactions.parquet", index=False)
    tables["users"].to_parquet(paths["dir"] / "users.parquet", index=False)
    tables["locations"].to_parquet(paths["dir"] / "locations.parquet", index=False)
    tables["mails"].to_parquet(paths["dir"] / "mails.parquet", index=False)
    tables["sms"].to_parquet(paths["dir"] / "sms.parquet", index=False)
    tables["audio"].to_parquet(paths["dir"] / "audio.parquet", index=False)
    meta = {"data_dir": str(data_dir)}
    paths["meta"].write_text(json.dumps(meta), encoding="utf-8")
    return paths["dir"]


def load_normalized(cache_dir: Path) -> dict[str, Any]:
    return {
        "transactions": pd.read_parquet(cache_dir / "transactions.parquet"),
        "users": pd.read_parquet(cache_dir / "users.parquet"),
        "locations": pd.read_parquet(cache_dir / "locations.parquet"),
        "mails": pd.read_parquet(cache_dir / "mails.parquet"),
        "sms": pd.read_parquet(cache_dir / "sms.parquet"),
        "audio": pd.read_parquet(cache_dir / "audio.parquet"),
        "data_dir": Path(json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))["data_dir"]),
        "cache_dir": cache_dir,
    }


def build_or_load(cache_root: Path, bundle: RawDatasetBundle, force: bool = False) -> dict[str, Any]:
    paths = cache_paths(cache_root, bundle.data_dir)
    flag = paths["dir"] / "transactions.parquet"
    if not force and flag.exists():
        return load_normalized(paths["dir"])
    tables = normalize_bundle(bundle)
    save_normalized(cache_root, bundle.data_dir, tables)
    return load_normalized(paths["dir"])
