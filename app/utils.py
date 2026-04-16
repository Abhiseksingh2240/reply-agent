"""Shared helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"(?<!^)(?=[A-Z])", "_", str(c)).lower().replace(" ", "_") for c in out.columns]
    return out


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def stable_json_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_iso_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def is_employer_id(sender_id: str) -> bool:
    s = str(sender_id)
    return s.upper().startswith("EMP") or s.upper().startswith("EMPLOYER")


def looks_like_biotag(sender_id: str) -> bool:
    s = str(sender_id)
    if len(s) < 8:
        return False
    return "-" in s and s[0].isalpha()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clip01(x: float | pd.Series | np.ndarray) -> float | pd.Series:
    if isinstance(x, pd.Series):
        return x.clip(0.0, 1.0)
    if isinstance(x, np.ndarray):
        return np.clip(x, 0.0, 1.0)
    v = float(x)
    return float(max(0.0, min(1.0, v)))


def softmax_rows(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    m = np.max(arr)
    e = np.exp(arr - m)
    return e / (np.sum(e) + 1e-12)
