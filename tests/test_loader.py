from pathlib import Path

import pytest

from app.io_loader import load_dataset


def test_load_truman_train():
    root = Path(__file__).resolve().parents[2] / "The+Truman+Show+-+train" / "The Truman Show - train"
    if not (root / "transactions.csv").exists():
        pytest.skip("dataset not present")
    bundle = load_dataset(root)
    assert not bundle.transactions.empty
    assert isinstance(bundle.users, list)
