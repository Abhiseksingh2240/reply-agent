"""Write challenge outputs and run metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.models import RunSummary


def write_predictions(ids: list[str], output_path: Path) -> None:
    """UTF-8, no BOM. One transaction_id per line; no blank trailing line (strict portals split on \\n)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [str(i).strip() for i in ids if str(i).strip()]
    # Join only — do not append an extra trailing "\\n" (that creates an empty "" line after split).
    output_path.write_text("\n".join(cleaned), encoding="utf-8")


def write_langfuse_session(session_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "langfuse_session_id.txt").write_text(session_id.strip(), encoding="utf-8")


def write_run_summary(summary: RunSummary, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_summary.json"
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def mirror_default_outputs(pred_path: Path, out_dir: Path) -> None:
    """Also write outputs/fraudulent_transactions.txt for challenge naming."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = pred_path.read_text(encoding="utf-8") if pred_path.exists() else ""
    (out_dir / "fraudulent_transactions.txt").write_text(text, encoding="utf-8")
