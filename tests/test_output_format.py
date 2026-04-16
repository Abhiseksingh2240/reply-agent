from pathlib import Path

from app.output_writer import write_predictions


def test_output_utf8_lines(tmp_path: Path):
    p = tmp_path / "out.txt"
    write_predictions(["a", "b"], p)
    text = p.read_text(encoding="utf-8")
    assert text.strip().splitlines() == ["a", "b"]
