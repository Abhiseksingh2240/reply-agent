from app.llm.audio_transcription import _suffix_to_format, transcription_cache_key
from pathlib import Path


def test_suffix_to_format():
    assert _suffix_to_format(".mp3") == "mp3"
    assert _suffix_to_format("wav") == "wav"
    assert _suffix_to_format(".m4a") == "aac"


def test_transcription_cache_key_stable(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("x", encoding="utf-8")
    k1 = transcription_cache_key(p)
    k2 = transcription_cache_key(p)
    assert k1 == k2
    assert len(k1) == 64
