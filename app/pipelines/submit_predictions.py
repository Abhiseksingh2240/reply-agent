"""Helpers for packaging submission artifacts."""

from __future__ import annotations

from pathlib import Path

from app.output_writer import mirror_default_outputs


def ensure_submission_files(pred_path: Path, outputs_dir: Path) -> None:
    mirror_default_outputs(pred_path, outputs_dir)
