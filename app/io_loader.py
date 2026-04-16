"""Load challenge dataset files from a directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class RawDatasetBundle:
    data_dir: Path
    transactions: pd.DataFrame
    users: list[dict[str, Any]]
    locations: list[dict[str, Any]]
    mails: list[dict[str, Any]]
    sms: list[dict[str, Any]]
    audio_files: list[Path]


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def discover_audio_files(data_dir: Path) -> list[Path]:
    audio_dir = data_dir / "audio"
    if not audio_dir.is_dir():
        return []
    exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}
    out: list[Path] = []
    for p in audio_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def load_dataset(data_dir: str | Path) -> RawDatasetBundle:
    root = Path(data_dir).resolve()
    tx_path = root / "transactions.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"Missing transactions.csv under {root}")

    transactions = pd.read_csv(tx_path)
    users = _read_json_list(root / "users.json")
    locations = _read_json_list(root / "locations.json")
    mails = _read_json_list(root / "mails.json")
    sms = _read_json_list(root / "sms.json")
    audio = discover_audio_files(root)
    return RawDatasetBundle(
        data_dir=root,
        transactions=transactions,
        users=users,
        locations=locations,
        mails=mails,
        sms=sms,
        audio_files=audio,
    )
