"""Load raw data, normalize, cache, and compute cheap scores."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings, load_settings
from app.feature_store import build_or_load
from app.io_loader import load_dataset
from app.scoring import score_transactions


def run_build_features(data_dir: Path, settings: Settings | None = None, force: bool = False) -> Path:
    settings = settings or load_settings()
    bundle = load_dataset(data_dir)
    tables = build_or_load(settings.data_cache_dir, bundle, force=force)
    scored = score_transactions(tables)
    cache_dir = Path(tables["cache_dir"])
    scored_path = cache_dir / "scored_transactions.parquet"
    scored.to_parquet(scored_path, index=False)
    return scored_path
